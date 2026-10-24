"""Keyed rare-region watermark screen; no manuscript changes or security claims."""
from repo_layout import ROOT, source_path

import argparse
import copy
import fcntl
import json
import math
from pathlib import Path
import secrets

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

import run_watermark_security as security
from run_embedding_strategy_robustness import code_from_key, prepare_model, save_torch, whole_model_attack
from run_secret_carrier_pilot import BlindRegressor, Removed, pilot_fid


class TailGate(nn.Module):
    def __init__(self, direction, mean, variance, threshold, sharpness):
        super().__init__()
        self.register_buffer("direction", direction / direction.norm())
        self.register_buffer("mean", torch.as_tensor(mean, device=direction.device))
        self.register_buffer("variance", torch.as_tensor(variance, device=direction.device))
        self.threshold, self.sharpness = threshold, sharpness

    def moments(self, t):
        return t * self.mean, ((1 - t).square() + t.square() * self.variance).sqrt().clamp_min(.05)

    def coordinate(self, x, t):
        mean, scale = self.moments(t)
        return (x.flatten(1) @ self.direction - mean) / scale

    def forward(self, x, t):
        return torch.sigmoid(self.sharpness * (self.coordinate(x, t) - self.threshold))

    def trigger(self, x, t, target):
        mean, scale = self.moments(t)
        shift = mean + scale * target - x.flatten(1) @ self.direction
        return x + (shift[:, None] * self.direction).reshape_as(x)


class GatedField(nn.Module):
    def __init__(self, base, gate, direction, epsilon):
        super().__init__()
        self.base, self.gate, self.epsilon = base, gate, epsilon
        self.register_buffer("direction", direction)

    def forward(self, x, t, y=None):
        a = torch.sin(2 * math.pi * t) * self.gate(x, t)
        return security.flow.model_velocity(self.base, x, t, y) + (self.epsilon * a[:, None] * self.direction).reshape_as(x)


def batches(pool, labels, gate, n, generator):
    x, t, y, target = security.query_batch(pool, labels, n, generator)
    position = gate.threshold + .5 + torch.rand(n, device=x.device, generator=generator)
    return x, gate.trigger(x, t, position), t, y, target


def embedding_loss(ordinary, tail, clean_ordinary, clean_tail, target, x, trigger, t,
                   gate, projection, code, epsilon, projection_weight):
    sine = torch.sin(2 * math.pi * t)
    desired_on = epsilon * (sine * gate(x, t))[:, None] * code
    desired_off = epsilon * (sine * gate(trigger, t))[:, None] * code
    signal_on = (desired_on @ projection.T).reshape_as(x)
    signal_off = (desired_off @ projection.T).reshape_as(x)
    residual_on, residual_off = ordinary - clean_ordinary, tail - clean_tail
    projection_loss = .5 * (F.mse_loss(residual_on.flatten(1) @ projection, desired_on)
                              + F.mse_loss(residual_off.flatten(1) @ projection, desired_off))
    return (F.mse_loss(ordinary, target + signal_on) + F.mse_loss(residual_off, signal_off)
            + projection_weight * projection_loss)


def train(base, gate, projection, code, config, pool, labels, args, directory, marked):
    torch.manual_seed(args.seed)
    model = prepare_model(base, config, "lora")
    path = directory / ("marked_adapter.pt" if marked else "control_adapter.pt")
    if path.exists():
        security.flow.load_adapter(model, path, pool.device)
        return model.eval().requires_grad_(False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    generator = torch.Generator(device=pool.device).manual_seed(args.seed)
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        x, trigger, t, y, target = batches(pool, labels, gate, args.train_batch, generator)
        with torch.no_grad():
            clean_ordinary = security.flow.model_velocity(base, x, t, y)
            clean_tail = security.flow.model_velocity(base, trigger, t, y)
        ordinary = security.flow.model_velocity(model, x, t, y)
        tail = security.flow.model_velocity(model, trigger, t, y)
        loss = embedding_loss(ordinary, tail, clean_ordinary, clean_tail, target, x, trigger, t,
                              gate, projection, code, args.epsilon if marked else 0., config["wm_proj_weight"])
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite embedding loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            history.append(dict(step=step, loss=float(loss.detach())))
            security.write_json(directory / ("marked_training.json" if marked else "control_training.json"), dict(history=history))
            print(f"TRAIN {directory.name}/{'marked' if marked else 'control'} {step}/{args.steps} loss={history[-1]['loss']:.5f}", flush=True)
    temporary = path.with_suffix(".tmp")
    security.flow.save_adapter(model, temporary)
    temporary.replace(path)
    return model.eval().requires_grad_(False)


@torch.no_grad()
def detection(model, base, gate, projection, codes, code, message, pool, labels, args):
    trials = []
    sums = dict(on_raw=0., on_excess=0., on_energy=0., off_raw=0., off_excess=0., off_energy=0.,
                gate=0., active=0., gate_square=0.)
    for trial in range(args.trials):
        generator = torch.Generator(device=pool.device).manual_seed(args.seed + 10000 + trial)
        total = torch.zeros(projection.shape[1], device=pool.device, dtype=torch.float64)
        energy = 0.
        for start in range(0, args.queries, args.eval_batch):
            n = min(args.eval_batch, args.queries - start)
            x, trigger, t, y, _ = batches(pool, labels, gate, n, generator)
            a = torch.sin(2 * math.pi * t) * gate(trigger, t)
            velocity = security.flow.model_velocity(model, trigger, t, y)
            total += (a[:, None] * (velocity.flatten(1) @ projection)).double().sum(0)
            energy += float(a.double().square().sum())
            if trial == 0:
                for prefix, query in (("on", x), ("off", trigger)):
                    v = security.flow.model_velocity(model, query, t, y)
                    clean = security.flow.model_velocity(base, query, t, y)
                    carrier = torch.sin(2 * math.pi * t)
                    raw = (v.flatten(1) @ projection) @ code
                    excess = ((v - clean).flatten(1) @ projection) @ code
                    sums[prefix + "_raw"] += float((carrier * raw).double().sum())
                    sums[prefix + "_excess"] += float((carrier * excess).double().sum())
                    sums[prefix + "_energy"] += float(carrier.double().square().sum())
                activation = gate(x, t)
                sums["gate"] += float(activation.double().sum())
                sums["gate_square"] += float(activation.double().square().sum())
                sums["active"] += int((activation > .5).sum())
        coordinates = total / max(energy, 1e-20)
        scores = coordinates @ codes.double().T
        decoded = ''.join(str(int(b)) for b in (scores > 0).tolist())
        trials.append(dict(decoded=decoded, exact=decoded == message,
                           bit_accuracy=sum(a == b for a, b in zip(decoded, message)) / len(message),
                           normalized_scores=scores.cpu().tolist(),
                           effective_strength_raw=float(coordinates @ code.double()) / float(code.square().sum())))
    result = dict(recovery_percent=100 * np.mean([r["exact"] for r in trials]),
                  bit_accuracy_percent=100 * np.mean([r["bit_accuracy"] for r in trials]), trials=trials,
                  ordinary_gate_mean=sums["gate"] / args.queries,
                  ordinary_gate_rms=math.sqrt(sums["gate_square"] / args.queries),
                  ordinary_activation_rate=sums["active"] / args.queries)
    for prefix in ("on", "off"):
        for kind in ("raw", "excess"):
            result[prefix + "_amplitude_" + kind] = sums[prefix + "_" + kind] / max(sums[prefix + "_energy"], 1e-20)
    result["leakage_ratio_excess"] = abs(result["on_amplitude_excess"]) / max(abs(result["off_amplitude_excess"]), 1e-12)
    return result


@torch.no_grad()
def quality(model, reference, args, device, destination):
    samples, _ = security.generate(model, args.samples, args.sample_steps, args.sample_batch,
                                   args.seed + 20000, device)
    features = security.flow.inception_features(samples, device, args.feature_batch)
    fid = pilot_fid(reference["real_features"], features)
    save_image(samples.add(1).div(2), destination.with_suffix(".png"), nrow=10)
    np.savez_compressed(destination.with_suffix(".npz"), samples=samples.numpy(), features=features.numpy())
    return dict(fid=fid, fid_ratio_clean=fid / reference["clean_fid"])


def coherent_amplitude(model, x, y):
    times = torch.tensor([.125, .375, .625, .875], device=x.device, dtype=torch.float64)
    carrier = torch.sin(2 * math.pi * times)
    nuisance = torch.stack((torch.ones_like(times), times), 1)
    carrier = carrier - nuisance @ torch.linalg.lstsq(nuisance, carrier).solution
    amplitude = torch.zeros_like(x, dtype=torch.float64)
    for time, weight in zip(times, carrier):
        t = torch.full((len(x),), float(time), device=x.device)
        amplitude = amplitude + weight * security.flow.model_velocity(model, x, t, y).double()
    return (amplitude / carrier.square().sum()).to(x.dtype)


def probe(model, args, device, gradient=False):
    """Public OOD search objective. No owner gate, key, or clean reference."""
    generator = torch.Generator(device=device).manual_seed(args.seed + 50000)
    candidates, scores, searches = [], [], 0
    for start in range(0, args.probe_candidates, args.probe_batch):
        n = min(args.probe_batch, args.probe_candidates - start)
        scale = torch.tensor([1., 2., 4.], device=device)[torch.arange(start, start + n, device=device) % 3]
        x = torch.randn(n, 3, 32, 32, device=device, generator=generator) * scale[:, None, None, None]
        y = torch.arange(start, start + n, device=device) % 10
        if gradient:
            for _ in range(args.probe_steps):
                x = x.detach().requires_grad_(True)
                objective = coherent_amplitude(model, x, y).flatten(1).square().mean(1)
                grad = torch.autograd.grad(objective.sum(), x)[0]
                with torch.no_grad():
                    grad /= grad.flatten(1).square().mean(1).sqrt().clamp_min(1e-12)[:, None, None, None]
                    x = x + .1 * grad
                    rms = x.flatten(1).square().mean(1).sqrt()
                    x *= torch.minimum(torch.ones_like(rms), scale / rms.clamp_min(1e-12))[:, None, None, None]
                searches += 4 * n
        with torch.no_grad():
            score = coherent_amplitude(model, x, y).flatten(1).square().mean(1)
        candidates.append(x.detach())
        scores.append(score.detach())
        searches += 4 * n
    x, scores = torch.cat(candidates), torch.cat(scores)
    count = min(64, len(x))
    chosen = scores.topk(count).indices
    return x[chosen], dict(forward_evaluations=searches,
                          backward_evaluations=args.probe_steps * args.probe_candidates if gradient else 0,
                          maximum_score=float(scores.max()), mean_score=float(scores.mean()),
                          gradient_access=gradient)


@torch.no_grad()
def attack_fit(model, pool, labels, args, candidates=None):
    generator = torch.Generator(device=pool.device).manual_seed(args.seed + 40000)
    xs, ts, vs = [], [], []
    for start in range(0, args.attack_queries, args.eval_batch):
        n = min(args.eval_batch, args.attack_queries - start)
        if candidates is None:
            x, t, y, _ = security.query_batch(pool, labels, n, generator)
        else:
            index = torch.randint(len(candidates), (n,), generator=generator, device=pool.device)
            x = candidates[index]
            t = torch.rand(n, device=pool.device, generator=generator)
            y = torch.randint(10, (n,), device=pool.device, generator=generator)
        xs.append(x)
        ts.append(t)
        vs.append(security.flow.model_velocity(model, x, t, y))
    return BlindRegressor("time", 3072).to(pool.device).fit(torch.cat(xs), torch.cat(ts), torch.cat(vs))


@torch.no_grad()
def probe_diagnostics(candidates, gate):
    xs = candidates.repeat_interleave(4, 0)
    ts = torch.tensor([.125, .375, .625, .875], device=xs.device).repeat(len(candidates))
    activation = gate(xs, ts)
    return dict(selected_gate_mean=float(activation.mean()), selected_activation_rate=float((activation > .5).float().mean()))


def summarize(output, results):
    security.write_json(output / "results.json", results)
    lines = ["# Tail-Gate Pilot", "", f"State: {results['state']}", "",
             "One private H32 key; 300-step mixed ordinary/trigger LoRA training by default. "
             "Controls receive identical queries and losses with zero watermark target. "
             "Owner decoding uses only marked velocities and normalized trigger demodulation. "
             "Clean-model subtraction is used only for analysis of signal strength/leakage. "
             "Each attack fits independently to marked responses without the owner key. "
             "Gradient probing requires checkpoint/input-gradient access; random probing uses velocities only. "
             "OOD search scores can reflect ordinary model behavior: matched clean-control probes are included. "
             "The 100-image paired FID is a pilot screen. Repeated trials are not independent keys.", "",
             "| Gate | Model | Condition | Exact % | Bits % | FID/clean | FID/control | Leakage ratio |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in results["rows"]:
        m = r["metrics"]
        ratio = m.get("fid_ratio_control")
        control_text = f"{ratio:.3f}" if ratio is not None else "n/a"
        lines.append(f"| {r['gate']} | {r['field']} | {r['condition']} | {m['recovery_percent']:.1f} | "
                     f"{m['bit_accuracy_percent']:.1f} | {m['fid_ratio_clean']:.3f} | "
                     f"{control_text} | {m['leakage_ratio_excess']:.4f} |")
    (output / "report.md").write_text('\n'.join(lines) + '\n')


def run(args):
    from huggingface_hub import hf_hub_download
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    reference_config = json.loads(Path("table_runs/watermark_security/config.json").read_text())
    job = next(j for j in reference_config["plan"] if j["scheme"] == "H32")
    original_key = Path(job["run_dir"]) / "watermark_key.pt"
    key = torch.load(original_key, map_location=device, weights_only=False)
    config = job["config"]
    cfg = security.flow.DEFAULT_CONFIGS["cifar10"]
    checkpoint = Path(hf_hub_download("WayBob/FlowMatching-Unet-Celeb-64x64", cfg.filename, local_files_only=True))
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    source_hashes={str(p): security.digest(p) for p in [Path(__file__), Path(security.__file__),
                                   Path(security.flow.__file__), source_path('run_secret_carrier_pilot.py'),
                                   source_path('run_embedding_strategy_robustness.py'), source_path('run_flow_unet_robustness.py')]},
                    checkpoint_sha256=security.digest(checkpoint), original_key_sha256=security.digest(original_key),
                    embedding_config=config)
    path = args.output / "config.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Resume requires identical sources/settings")
    security.write_json(path, manifest)
    result_path = args.output / "results.json"
    results = json.loads(result_path.read_text()) if result_path.exists() else dict(rows=[], probes=[])
    results.update(state="running", expected_screen_evaluations=10 * len(args.thresholds) * len(args.sharpness))
    results.pop("error", None)
    summarize(args.output, results)
    dataset = CIFAR10(str(security.ROOT / "data"), train=True, download=False)
    pool = torch.from_numpy(dataset.data[:20000]).permute(0, 3, 1, 2).to(device).float() / 127.5 - 1
    labels = torch.tensor(dataset.targets[:20000], device=device)
    base = security.flow.build_model(cfg, device)
    security.flow.load_state_dict_flexible(base, str(checkpoint), device)
    base.eval().requires_grad_(False)
    _, code = code_from_key(key, config, device)
    secret_path = args.output / "private_key.pt"
    if secret_path.exists():
        private = torch.load(secret_path, map_location=device, weights_only=True)
    else:
        generator = torch.Generator(device=device).manual_seed(secrets.randbits(63))
        direction = torch.randn(3072, generator=generator, device=device)
        direction /= direction.norm()
        projection = torch.linalg.qr(torch.randn(3072, 32, generator=generator, device=device)).Q
        private = dict(direction=direction, projection=projection)
        save_torch(secret_path, private)
        secret_path.chmod(0o600)
    fingerprint = dict(sha256=security.digest(secret_path))
    path = args.output / "secret_manifest.json"
    if path.exists() and json.loads(path.read_text()) != fingerprint:
        raise ValueError("Private key changed")
    security.write_json(path, fingerprint)
    direction, projection = private["direction"], private["projection"]
    projected_data = pool.flatten(1) @ direction
    mean, variance = projected_data.mean(), projected_data.var(unbiased=False)
    cached = torch.load("table_runs/secret_carrier_pilot/quality_reference.pt", map_location="cpu", weights_only=True)
    # Recompute the clean quality baseline if the pilot generation protocol differs.
    reference = dict(real_features=cached["real_features"][:args.samples])
    try:
        clean_samples, _ = security.generate(base, args.samples, args.sample_steps, args.sample_batch,
                                             args.seed + 20000, device)
        reference["clean_fid"] = pilot_fid(reference["real_features"], security.flow.inception_features(clean_samples, device, args.feature_batch))
        for threshold in args.thresholds:
            for sharpness in args.sharpness:
                label = f"tau{threshold:g}_slope{sharpness:g}"
                directory = args.output / label
                directory.mkdir(exist_ok=True)
                gate = TailGate(direction, mean, variance, threshold, sharpness)
                results["current"] = label
                summarize(args.output, results)
                control = train(base, gate, projection, code, config, pool, labels, args, directory, False)
                control_fid = None

                def record(model, field, condition):
                    nonlocal control_fid
                    path = directory / (field + "_" + condition + ".json")
                    if path.exists():
                        metrics = json.loads(path.read_text())
                    else:
                        metrics = detection(model, base, gate, projection, key["codes"], code, key["message"], pool, labels, args)
                        metrics.update(quality(model, reference, args, device, path.with_suffix("")))
                        if control_fid is not None:
                            metrics["fid_ratio_control"] = metrics["fid"] / control_fid
                        security.write_json(path, metrics)
                    if field == "control":
                        control_fid = metrics["fid"]
                    if not any(r["gate"] == label and r["field"] == field and r["condition"] == condition for r in results["rows"]):
                        results["rows"].append(dict(gate=label, field=field, condition=condition, metrics=metrics))
                    summarize(args.output, results)
                    print(f"EVAL {label}/{field}/{condition}: exact={metrics['recovery_percent']:.1f}% bits={metrics['bit_accuracy_percent']:.1f}%", flush=True)
                    return metrics

                record(control, "control", "baseline")
                record(base, "clean", "baseline")
                control_probes = {}
                for field in ("injected", "trained"):
                    model = (GatedField(base, gate, projection @ code, args.epsilon) if field == "injected"
                             else train(base, gate, projection, code, config, pool, labels, args, directory, True))
                    model.eval().requires_grad_(False)
                    baseline = record(model, field, "baseline")
                    estimator = attack_fit(model, pool, labels, args)
                    save_torch(directory / (field + "_ordinary_regression.pt"), estimator.state_dict())
                    record(Removed(model, estimator), field, "ordinary_regression")
                    for gradient in (False, True):
                        attack = "gradient" if gradient else "random"
                        candidates, diagnostics = probe(model, args, device, gradient)
                        diagnostics.update(probe_diagnostics(candidates, gate))
                        if gradient not in control_probes:
                            control_candidates, control_diagnostics = probe(control, args, device, gradient)
                            control_diagnostics.update(probe_diagnostics(control_candidates, gate))
                            control_probes[gradient] = control_candidates, control_diagnostics
                        control_candidates, control_diagnostics = control_probes[gradient]
                        save_torch(directory / (field + "_" + attack + "_probes.pt"), dict(candidates=candidates.cpu(), diagnostics=diagnostics,
                                   control_candidates=control_candidates.cpu(), control_diagnostics=control_diagnostics))
                        probe_row = dict(gate=label, field=field, attack=attack, marked=diagnostics, control=control_diagnostics)
                        results["probes"] = [r for r in results["probes"] if (r["gate"], r["field"], r["attack"]) != (label, field, attack)] + [probe_row]
                        estimator = attack_fit(model, pool, labels, args, candidates)
                        save_torch(directory / (field + "_" + attack + "_regression.pt"), estimator.state_dict())
                        record(Removed(model, estimator), field, "ood_" + attack + "_regression")
                    if field == "trained" and baseline["recovery_percent"] == 100 and not args.skip_followup:
                        for attack, severity in (("prune", .5), ("quant", 4)):
                            compressed = security.merge_adapters(copy.deepcopy(model))
                            scope = whole_model_attack(compressed, attack, severity)
                            security.write_json(directory / (attack + "_scope.json"), scope)
                            record(compressed, field, attack + "_" + str(severity).replace('.', 'p'))
                            del compressed
                        finetuned = security.merge_adapters(copy.deepcopy(model))
                        settings = argparse.Namespace(steps=args.ft_steps, lr=1e-4, batch_size=args.train_batch, seed=args.seed)
                        torch.manual_seed(args.seed + 80000)
                        finetuned = security.train_student(finetuned, model, pool, labels, "full_finetune", settings, directory)
                        record(finetuned, field, "full_clean_ft")
                        del finetuned
                    del model, estimator
                    torch.cuda.empty_cache()
                del control, control_probes
        results.update(state="complete", current=None)
    except Exception as exc:
        results.update(state="failed", error=repr(exc))
        raise
    finally:
        summarize(args.output, results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, default=Path("table_runs/tail_gate_pilot"))
    p.add_argument("--thresholds", type=float, nargs="+", default=[2.5, 4.])
    p.add_argument("--sharpness", type=float, nargs="+", default=[2., 4.])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--epsilon", type=float, default=3.)
    p.add_argument("--train-batch", type=int, default=64)
    p.add_argument("--queries", type=int, default=4096)
    p.add_argument("--trials", type=int, default=4)
    p.add_argument("--eval-batch", type=int, default=256)
    p.add_argument("--attack-queries", type=int, default=8192)
    p.add_argument("--probe-candidates", type=int, default=512)
    p.add_argument("--probe-batch", type=int, default=16)
    p.add_argument("--probe-steps", type=int, default=10)
    p.add_argument("--ft-steps", type=int, default=300)
    p.add_argument("--skip-followup", action="store_true")
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--sample-batch", type=int, default=100)
    p.add_argument("--feature-batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=930042)
    a = p.parse_args()
    if min(a.steps, a.train_batch, a.queries, a.trials, a.eval_batch, a.attack_queries,
           a.probe_candidates, a.probe_batch, a.probe_steps, a.ft_steps, a.samples, a.sample_steps,
           a.sample_batch, a.feature_batch, a.epsilon, *a.thresholds, *a.sharpness) <= 0:
        p.error("Counts and scales must be positive")
    if a.samples % 10 or a.samples > 100:
        p.error("This pilot supports 10..100 samples in multiples of 10")
    if len(set(a.thresholds)) != len(a.thresholds) or len(set(a.sharpness)) != len(a.sharpness):
        p.error("Duplicate sweep settings")
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / "study.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(a)
