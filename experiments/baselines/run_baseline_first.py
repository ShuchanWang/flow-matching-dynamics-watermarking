"""Frozen-protocol baseline tuning and independent-key confirmation for CIFAR RF."""
from repo_layout import ROOT, source_path

import argparse
import copy
import fcntl
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import beta
import torch
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

import run_watermark_security as security
from run_embedding_strategy_robustness import save_torch
from run_secret_carrier_pilot import pilot_fid


METHODS = ("dynamics", "weight", "trigger", "naive")
GRIDS = {"dynamics": [.5, 1.5, 3.], "weight": [.1, 1., 10.],
         "trigger": [.1, 1., 10.], "naive": [.5, 1.5, 3.]}
LAYER = "output_blocks.11.0.out_layers.3.weight"


def make_key(payload, seed, weight_shape, device):
    rng = torch.Generator(device=device).manual_seed(seed)
    bits = torch.randint(2, (payload,), generator=rng, device=device)
    projection = torch.linalg.qr(torch.randn(3072, 32, generator=rng, device=device)).Q
    codes = (torch.linalg.qr(torch.randn(32, 32, generator=rng, device=device)).Q.T
             if payload == 5 else torch.eye(32, device=device))
    public = torch.Generator().manual_seed(0)
    public_projection = torch.linalg.qr(torch.randn(3072, payload, generator=public)).Q.to(device)
    return dict(bits=bits, projection=projection, codes=codes, public_projection=public_projection,
                weight_projection=torch.randn(math.prod(weight_shape[1:]), payload, generator=rng, device=device),
                trigger_noise=torch.randn(1, 3, 32, 32, generator=rng, device=device),
                trigger_label=torch.randint(10, (1,), generator=rng, device=device),
                cells=torch.randperm(32, generator=rng, device=device)[:payload], seed=seed)


def code_for(key, method):
    bits = key["bits"]
    if method == "naive":
        return (2 * bits.float() - 1) / math.sqrt(len(bits))
    if len(bits) == 5:
        index = sum(int(bit) << i for i, bit in enumerate(bits))
        return key["codes"][index]
    return (2 * bits.float() - 1) / math.sqrt(len(bits))


def weight_logits(model, key):
    weights = dict(model.named_parameters())[LAYER].mean(0).flatten()
    return 10 * (weights @ key["weight_projection"])


def trigger_image(key):
    tiles = torch.zeros(32, device=key["bits"].device)
    tiles[key["cells"]] = .7 * (2 * key["bits"].float() - 1)
    return tiles.reshape(4, 8).repeat_interleave(8, 0).repeat_interleave(4, 1)[None, None].expand(1, 3, 32, 32)


def trigger_bits(images, key):
    tiles = images.mean(1).reshape(len(images), 4, 8, 8, 4).mean((2, 4)).flatten(1)
    return tiles[:, key["cells"]] > 0


def projected_loss(prediction, clean, target, t, key, method, strength):
    projection = key["public_projection"] if method == "naive" else key["projection"]
    code = code_for(key, method)
    carrier = torch.ones_like(t) if method == "naive" else torch.sin(2 * math.pi * t)
    desired = strength * carrier[:, None] * code
    signal = (desired @ projection.T).reshape_as(prediction)
    coordinates = (prediction - clean).flatten(1) @ projection
    correlation = (carrier * (coordinates * code).sum(1)).mean()
    reward = torch.tanh(correlation / (.5 * strength) * 2) if strength > 0 else 0.
    return (F.mse_loss(prediction, target + signal) + 10 * F.mse_loss(coordinates, desired)
            - reward)


def train(base, method, key, strength, seed, pool, labels, args, directory, marked):
    directory.mkdir(parents=True, exist_ok=True)
    final, latest = directory / "model.pt", directory / "latest.pt"
    model = copy.deepcopy(base).requires_grad_(True)
    if final.exists():
        model.load_state_dict(torch.load(final, weights_only=True, map_location=pool.device))
        return model.eval().requires_grad_(False)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    rng = torch.Generator(device=pool.device).manual_seed(seed)
    auxiliary_rng = torch.Generator(device=pool.device).manual_seed(seed + 90000)
    completed, elapsed = 0, 0.
    if latest.exists():
        state = torch.load(latest, weights_only=True, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        rng.set_state(state["rng"])
        auxiliary_rng.set_state(state["auxiliary_rng"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], pool.device)
        completed, elapsed = state["step"], state["elapsed"]
        del state
    started = time.monotonic()
    destination = trigger_image(key)
    for step in range(completed + 1, args.steps + 1):
        model.train()
        x, t, y, target = security.query_batch(pool, labels, args.batch, rng)
        prediction = security.flow.model_velocity(model, x, t, y)
        if method in ("dynamics", "naive"):
            with torch.no_grad():
                clean = security.flow.model_velocity(base, x, t, y)
            loss = projected_loss(prediction, clean, target, t, key, method, strength if marked else 0.)
        else:
            loss = F.mse_loss(prediction, target)
            if method == "weight" and marked:
                loss = loss + strength * F.binary_cross_entropy_with_logits(weight_logits(model, key), key["bits"].float())
            if method == "trigger":
                z = key["trigger_noise"] + .02 * torch.randn(args.trigger_batch, 3, 32, 32, generator=auxiliary_rng, device=x.device)
                tt = torch.rand(args.trigger_batch, generator=auxiliary_rng, device=x.device)
                xx = (1 - tt[:, None, None, None]) * z + tt[:, None, None, None] * destination
                yy = key["trigger_label"].expand(args.trigger_batch)
                if marked:
                    auxiliary_target = destination - z
                else:
                    with torch.no_grad():
                        auxiliary_target = security.flow.model_velocity(base, xx, tt, yy)
                auxiliary_prediction = security.flow.model_velocity(model, xx, tt, yy)
                loss = loss + strength * F.mse_loss(auxiliary_prediction, auxiliary_target)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite embedding loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            print(f"TRAIN {directory.relative_to(args.output)} {step}/{args.steps} loss={float(loss.detach()):.5f}", flush=True)
        if step % 100 == 0 or step == args.steps:
            spent = elapsed + time.monotonic() - started
            save_torch(latest, dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                                   rng=rng.get_state(), auxiliary_rng=auxiliary_rng.get_state(), cpu_rng=torch.get_rng_state(),
                                   cuda_rng=torch.cuda.get_rng_state(pool.device), step=step, elapsed=spent))
            security.write_json(directory / "training.json", dict(steps=step, elapsed_seconds=spent, marked=marked,
                                method=method, strength=strength, training_seed=seed, ordinary_examples=step * args.batch,
                                auxiliary_examples=step * args.trigger_batch if method == "trigger" else 0,
                                trainable_parameters=sum(p.numel() for p in model.parameters())))
    save_torch(final, model.state_dict())
    latest.unlink(missing_ok=True)
    return model.eval().requires_grad_(False)


def binomial_interval(successes, count):
    return [0. if successes == 0 else float(beta.ppf(.025, successes, count - successes + 1)),
            1. if successes == count else float(beta.ppf(.975, successes + 1, count - successes))]


@torch.no_grad()
def verify(model, method, key, pool, labels, args):
    started = time.monotonic()
    truth, trials = key["bits"].bool(), []
    if method == "weight":
        decoded = weight_logits(model, key) > 0
        trials.append(dict(decoded=decoded.long().tolist(), exact=bool(torch.equal(decoded, truth)),
                           bit_accuracy=float((decoded == truth).float().mean())))
        cost = dict(interface="white-box weights", inputs=0, velocity_calls=0, weight_reads=1)
    elif method == "trigger":
        rng = torch.Generator(device=pool.device).manual_seed(args.eval_seed + 10000)
        for _ in range(args.trials):
            x = key["trigger_noise"] + .01 * torch.randn(key["trigger_noise"].shape, device=pool.device, generator=rng)
            y = key["trigger_label"]
            for step in range(args.sample_steps):
                t = torch.full((1,), step / args.sample_steps, device=pool.device)
                x = (x + security.flow.model_velocity(model, x, t, y) / args.sample_steps).clamp(-5, 5)
            decoded = trigger_bits(x.clamp(-1, 1), key)[0]
            trials.append(dict(decoded=decoded.long().tolist(), exact=bool(torch.equal(decoded, truth)),
                               bit_accuracy=float((decoded == truth).float().mean())))
        cost = dict(interface="generated images", inputs=args.trials,
                    velocity_calls=args.trials * args.sample_steps, weight_reads=0)
    else:
        for trial in range(args.trials):
            rng = torch.Generator(device=pool.device).manual_seed(args.eval_seed + 10000 + trial)
            signature = torch.zeros(3072, device=pool.device, dtype=torch.float64)
            for start in range(0, args.queries, args.eval_batch):
                x, t, y, _ = security.query_batch(pool, labels, min(args.eval_batch, args.queries - start), rng)
                v = security.flow.model_velocity(model, x, t, y)
                carrier = torch.ones_like(t) if method == "naive" else torch.sin(2 * math.pi * t)
                signature += (carrier[:, None] * v.flatten(1)).double().sum(0)
            projection = key["public_projection"] if method == "naive" else key["projection"]
            coordinates = signature @ projection.double() / args.queries
            if method == "dynamics" and len(truth) == 5:
                scores = coordinates @ key["codes"].double().T
                index = int(scores.argmax())
                decoded = torch.tensor([(index >> i) & 1 for i in range(5)], device=pool.device).bool()
            else:
                scores = coordinates
                decoded = scores > 0
            trials.append(dict(decoded=decoded.long().tolist(), exact=bool(torch.equal(decoded, truth)),
                               bit_accuracy=float((decoded == truth).float().mean()), scores=scores.cpu().tolist()))
        cost = dict(interface="velocity queries", inputs=args.trials * args.queries,
                    velocity_calls=args.trials * args.queries, weight_reads=0)
    successes = sum(t["exact"] for t in trials)
    if pool.device.type == "cuda":
        torch.cuda.synchronize(pool.device)
    return dict(recovery_percent=100 * successes / len(trials), bit_accuracy_percent=100 * np.mean([t["bit_accuracy"] for t in trials]),
                within_model_interval=binomial_interval(successes, len(trials)) if method != "weight" else None,
                trials=trials, verification_cost=cost, verification_elapsed_seconds=time.monotonic() - started)


def kid(features_a, features_b, args, paired=False):
    """Unbiased cubic MMD; paired controls exclude the matched cross diagonal."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a, b = features_a.to(device).double(), features_b.to(device).double()
    size = min(args.kid_subset, len(a), len(b))
    rng = torch.Generator(device=device).manual_seed(args.eval_seed + 30000)
    values = []
    for _ in range(args.kid_subsets):
        ia = torch.randperm(len(a), device=device, generator=rng)[:size]
        ib = ia if paired else torch.randperm(len(b), device=device, generator=rng)[:size]
        x, y = a[ia], b[ib]
        xx, yy, xy = (x @ x.T / x.shape[1] + 1).pow(3), (y @ y.T / y.shape[1] + 1).pow(3), (x @ y.T / x.shape[1] + 1).pow(3)
        within = (xx.sum() - xx.trace() + yy.sum() - yy.trace()) / (size * (size - 1))
        between = ((xy.sum() - xy.trace()) / (size * (size - 1)) if paired else xy.mean())
        values.append(float(within - 2 * between))
    return dict(mean=float(np.mean(values)), subset_std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.,
                subsets=len(values), subset_size=size, paired_cross_diagonal_excluded=paired)


def fid(a, b):
    # Small smoke tests use the equivalent unregularized sample-space formula.
    return pilot_fid(a, b) if max(len(a), len(b)) <= 1024 else security.flow.compute_fid_from_features(a, b)


@torch.no_grad()
def quality(model, real_features, args, path):
    metadata, features_path = path / "quality.json", path / "features.pt"
    if metadata.exists() and features_path.exists():
        return json.loads(metadata.read_text()), torch.load(features_path, weights_only=True, map_location="cpu")
    print(f"QUALITY {path}: generating {args.samples} images, {args.sample_steps} Euler steps", flush=True)
    samples, _ = security.generate(model, args.samples, args.sample_steps, args.sample_batch, args.eval_seed + 20000, torch.device("cuda"))
    print(f"QUALITY {path}: extracting Inception features", flush=True)
    features = security.flow.inception_features(samples, torch.device("cuda"), args.feature_batch)
    print(f"QUALITY {path}: FID and KID", flush=True)
    result = dict(fid=fid(real_features, features), kid_real=kid(real_features, features, args),
                  samples=args.samples, sample_steps=args.sample_steps)
    save_torch(features_path, features)
    save_image(samples[:100].add(1).div(2), path / "samples.png", nrow=10)
    security.write_json(metadata, result)
    return result, features


def select_trials(rows):
    result = []
    for payload, method in sorted({(r["payload"], r["method"]) for r in rows}):
        selected = [r for r in rows if (r["payload"], r["method"]) == (payload, method)]
        qualified = [r for r in selected if r["metrics"]["recovery_percent"] >= 95 and r["metrics"]["fid_ratio_control"] <= 1.05]
        winner = (min(qualified, key=lambda r: (r["metrics"]["fid_ratio_control"], r["trial"])) if qualified else
                  min(selected, key=lambda r: (-r["metrics"]["recovery_percent"], r["metrics"]["fid_ratio_control"], r["trial"])))
        result.append(dict(payload=payload, method=method, strength=winner["strength"], qualified=bool(qualified),
                           development_trial=winner["trial"]))
    return result


def summarize(output, results):
    security.write_json(output / "results.json", results)
    lines = ["# Baseline-First Study", "", f"State: {results['state']}", "",
             "Development tuning and independent-key confirmation are separate stages. "
             "No attack-based selection. All failures are retained. Weight extraction is "
             "deterministic; repeated functional queries are within-model trials. "
             "Trigger is a generative adaptation of a classification backdoor watermark. "
             "No comparative robustness conclusion is available from embedding alone.", "",
             "| Payload | Method | Key/seed | Trial | Strength | Exact % | Bit % | FID/control | KID real |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results["rows"]:
        m = r["metrics"]
        lines.append(f"| {r['payload']} | {r['method']} | {r['seed']} | {r['trial']} | {r['strength']:g} | "
                     f"{m['recovery_percent']:.2f} | {m['bit_accuracy_percent']:.2f} | {m['fid_ratio_control']:.4f} | {m['kid_real']['mean']:.6f} |")
    groups = []
    if results["stage"] == "confirm":
        for payload, method in sorted({(r["payload"], r["method"]) for r in results["rows"]}):
            rows = [r for r in results["rows"] if (r["payload"], r["method"]) == (payload, method)]
            group = dict(payload=payload, method=method, independent_keys=len(rows))
            for metric in ("recovery_percent", "bit_accuracy_percent", "fid_ratio_control"):
                values = [r["metrics"][metric] for r in rows]
                group[metric + "_mean"] = float(np.mean(values))
                group[metric + "_sample_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            group["qualified_key_interval"] = binomial_interval(sum(r["metrics"]["recovery_percent"] >= 95 and r["metrics"]["fid_ratio_control"] <= 1.05 for r in rows), len(rows))
            groups.append(group)
    security.write_json(output / "summary.json", dict(state=results["state"], groups=groups))
    (output / "report.md").write_text('\n'.join(lines) + '\n')


def run(args):
    from huggingface_hub import hf_hub_download
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    output = args.output / args.stage
    output.mkdir(parents=True, exist_ok=True)
    cfg = security.flow.DEFAULT_CONFIGS["cifar10"]
    checkpoint = Path(hf_hub_download("WayBob/FlowMatching-Unet-Celeb-64x64", cfg.filename, local_files_only=True))
    selection = None
    if args.stage == "confirm":
        tuning = json.loads((args.output / "tune/results.json").read_text())
        if tuning["state"] != "complete":
            raise ValueError("All tuning trials must complete before confirmation")
        selection = json.loads((args.output / "selection.json").read_text())
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    protocol_sha256=security.digest(ROOT / "baseline_protocol.md"),
                    sources={str(p): security.digest(p) for p in [Path(__file__), Path(security.__file__), Path(security.flow.__file__), source_path('run_embedding_strategy_robustness.py'), source_path('run_secret_carrier_pilot.py')]},
                    base_sha256=security.digest(checkpoint), grids=GRIDS, selected=selection,
                    fidelity="protocol" if args.samples >= 5000 and args.steps == 1000 and args.keys >= 5 else "smoke/non-protocol")
    if args.stage == "confirm":
        tuning_manifest = json.loads((args.output / "tune/config.json").read_text())
        for name in ("protocol_sha256", "sources", "base_sha256", "grids"):
            if manifest[name] != tuning_manifest[name]:
                raise ValueError(f"Confirmation differs from tuning: {name}")
        for name in ("methods", "payloads", "steps", "batch", "trigger_batch", "samples", "sample_steps", "sample_batch",
                     "feature_batch", "queries", "trials", "eval_batch", "kid_subset", "kid_subsets", "eval_seed"):
            if manifest["arguments"][name] != tuning_manifest["arguments"][name]:
                raise ValueError(f"Confirmation protocol changed: {name}")
    path = output / "config.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Resume requires identical sources/settings")
    security.write_json(path, manifest)
    (output / "protocol_snapshot.md").write_bytes((ROOT / "baseline_protocol.md").read_bytes())
    path = output / "results.json"
    results = json.loads(path.read_text()) if path.exists() else dict(rows=[], stage=args.stage)
    results.update(state="running")
    results.pop("error", None)
    summarize(output, results)
    base = security.flow.build_model(cfg, device)
    security.flow.load_state_dict_flexible(base, str(checkpoint), device)
    base.eval().requires_grad_(False)
    shape = dict(base.named_parameters())[LAYER].shape
    dataset = CIFAR10(str(ROOT / "data"), train=True, download=False)
    pool = torch.from_numpy(dataset.data[:20000]).permute(0, 3, 1, 2).to(device).float() / 127.5 - 1
    labels = torch.tensor(dataset.targets[:20000], device=device)
    reference_path = args.output / "real_features.pt"
    try:
        if reference_path.exists():
            real_features = torch.load(reference_path, weights_only=True, map_location="cpu")
            if len(real_features) != args.samples:
                raise ValueError("Sample count differs from the saved reference")
        else:
            print(f"REFERENCE: extracting {args.samples} real CIFAR test features", flush=True)
            test = CIFAR10(str(ROOT / "data"), train=False, download=False)
            indices = np.concatenate([np.flatnonzero(np.asarray(test.targets) == y)[:args.samples // 10] for y in range(10)])
            real = torch.from_numpy(test.data[indices]).permute(0, 3, 1, 2).float() / 127.5 - 1
            real_features = security.flow.inception_features(real, device, args.feature_batch)
            save_torch(reference_path, real_features)
        floor_path = args.output / "noise_floor.json"
        if not floor_path.exists():
            print(f"NOISE FLOOR: generating two clean sets of {args.samples} images", flush=True)
            images, _ = security.generate(base, args.samples, args.sample_steps, args.sample_batch, args.eval_seed + 20000, device)
            a = security.flow.inception_features(images, device, args.feature_batch)
            images, _ = security.generate(base, args.samples, args.sample_steps, args.sample_batch, args.eval_seed + 20001, device)
            b = security.flow.inception_features(images, device, args.feature_batch)
            save_torch(args.output / "clean_features.pt", dict(first=a, reseed=b))
            security.write_json(floor_path, dict(fid_between_clean_reseeds=fid(a, b),
                                               kid_between_clean_reseeds=kid(a, b, args), samples=args.samples))
        for payload in args.payloads:
            for replicate in range(1 if args.stage == "tune" else args.keys):
                seed = 810000 + payload if args.stage == "tune" else 910000 + 100 * replicate + payload
                key_path = output / str(payload) / str(replicate) / "key.pt"
                key_path.parent.mkdir(parents=True, exist_ok=True)
                if key_path.exists():
                    key = torch.load(key_path, weights_only=True, map_location=device)
                else:
                    key = make_key(payload, seed, shape, device)
                    save_torch(key_path, key)
                    key_path.chmod(0o600)
                key_manifest = key_path.parent / "key_manifest.json"
                fingerprint = dict(sha256=security.digest(key_path))
                if key_manifest.exists() and json.loads(key_manifest.read_text()) != fingerprint:
                    raise ValueError("Saved owner key changed")
                security.write_json(key_manifest, fingerprint)
                for method in args.methods:
                    strengths = (GRIDS[method][:args.tuning_trials] if args.stage == "tune" else
                                 [next(s["strength"] for s in selection if s["payload"] == payload and s["method"] == method)])
                    for trial, strength in enumerate(strengths):
                        identity = dict(payload=payload, method=method, seed=seed, trial=trial, strength=strength)
                        if any(all(r[k] == v for k, v in identity.items()) for r in results["rows"]):
                            continue
                        results["current"] = identity
                        summarize(output, results)
                        directory = key_path.parent / method / f"trial_{trial}"
                        control = train(base, method, key, strength, seed, pool, labels, args, directory / "control", False)
                        control_quality, control_features = quality(control, real_features, args, directory / "control")
                        # Keep unmarked decoding controls instead of assuming successful extraction is specific.
                        control_metrics = verify(control, method, key, pool, labels, args)
                        security.write_json(directory / "control/verification.json", control_metrics)
                        del control
                        torch.cuda.empty_cache()
                        marked = train(base, method, key, strength, seed, pool, labels, args, directory / "marked", True)
                        metrics = verify(marked, method, key, pool, labels, args)
                        quality_metrics, features = quality(marked, real_features, args, directory / "marked")
                        metrics.update(quality_metrics, fid_ratio_control=quality_metrics["fid"] / control_quality["fid"],
                                       control_quality=control_quality, control_verification=control_metrics,
                                       kid_control=kid(features, control_features, args, paired=True))
                        security.write_json(directory / "metrics.json", metrics)
                        results["rows"].append(dict(**identity, metrics=metrics))
                        summarize(output, results)
                        print(f"RESULT {identity}: recovery={metrics['recovery_percent']:.1f}% FID/control={metrics['fid_ratio_control']:.4f}", flush=True)
                        del marked
                        torch.cuda.empty_cache()
        if args.stage == "tune":
            security.write_json(args.output / "selection.json", select_trials(results["rows"]))
        results.update(state="complete", current=None)
    except Exception as exc:
        results.update(state="failed", error=repr(exc))
        raise
    finally:
        summarize(output, results)
        (args.output / "baseline_gate.md").write_text(
            "# Baseline Gate\n\nEmbedding calibration/confirmation only.\n\n"
            "Comparative compression, fine-tuning, distillation and removal results are pending. "
            "No broad-win or venue conclusion is supported yet.\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "table_runs/baseline_first")
    p.add_argument("--stage", choices=["tune", "confirm"], default="tune")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--payloads", type=int, nargs="+", choices=[5, 32], default=[32, 5])
    p.add_argument("--keys", type=int, default=5)
    p.add_argument("--tuning-trials", type=int, default=3)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--trigger-batch", type=int, default=16)
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--sample-steps", type=int, default=100)
    p.add_argument("--sample-batch", type=int, default=128)
    p.add_argument("--feature-batch", type=int, default=32)
    p.add_argument("--queries", type=int, default=4096)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--eval-batch", type=int, default=512)
    p.add_argument("--kid-subset", type=int, default=1000)
    p.add_argument("--kid-subsets", type=int, default=20)
    p.add_argument("--eval-seed", type=int, default=930042)
    a = p.parse_args()
    if min(a.keys, a.steps, a.batch, a.trigger_batch, a.samples, a.sample_steps, a.sample_batch,
           a.feature_batch, a.queries, a.trials, a.eval_batch, a.kid_subset, a.kid_subsets, a.tuning_trials) < 1:
        p.error("Counts must be positive")
    if a.samples < 10 or a.samples > 10000 or a.samples % 10 or a.kid_subset < 2 or a.tuning_trials > 3:
        p.error("Invalid sample/KID/tuning settings")
    if len(set(a.methods)) != len(a.methods) or len(set(a.payloads)) != len(a.payloads):
        p.error("Duplicate method/payload settings")
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / "study.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(a)
