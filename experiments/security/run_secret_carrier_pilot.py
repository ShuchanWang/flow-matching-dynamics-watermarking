"""Small CIFAR pilot: carrier estimation, learned embedding, recovery and utility.

Injected fields are mechanism controls, not trained-watermark results. Attack fits
receive only marked velocities and public inputs; owner keys are used separately.
"""
from repo_layout import ROOT, source_path

import argparse
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
from run_embedding_strategy_robustness import code_from_key, prepare_model, save_torch


def time_features(t, tones):
    phase = 2 * math.pi * t[:, None] * torch.arange(1, tones + 1, device=t.device)
    return torch.cat((phase.sin(), phase.cos()), 1)


def pilot_fid(real, generated):
    """Gaussian FID via the sample-space SVD, avoiding a 2048-square root.

    For covariance factors A and B, tr(sqrt(Ca Cb)) = ||A B^T||_*.
    This evaluates the usual unregularized FID, including singular covariances.
    """
    a, b = real.double().cpu().numpy(), generated.double().cpu().numpy()
    difference = a.mean(0) - b.mean(0)
    a = (a - a.mean(0)) / math.sqrt(len(a) - 1)
    b = (b - b.mean(0)) / math.sqrt(len(b) - 1)
    cross = np.linalg.svd(a @ b.T, compute_uv=False).sum()
    return float(max(0., difference @ difference + (a * a).sum() + (b * b).sum() - 2 * cross))


class OwnerCarrier(nn.Module):
    def __init__(self, kind, projection, codes, message, seed=72341, tones=8):
        super().__init__()
        self.kind, self.tones = kind, tones
        self.message = message
        generator = torch.Generator(device=projection.device).manual_seed(seed)
        self.register_buffer("projection", projection)
        self.register_buffer("codes", codes)
        coefficients = torch.randn(2 * tones, generator=generator, device=projection.device)
        coefficients /= coefficients.norm()
        self.register_buffer("coefficients", coefficients)
        features = torch.randn(projection.shape, generator=generator, device=projection.device)
        features /= features.norm(dim=0, keepdim=True)
        self.register_buffer("features", features)
        self.register_buffer("center", torch.zeros(projection.shape[1], device=projection.device))
        self.register_buffer("scale", torch.ones(projection.shape[1], device=projection.device))

    def gates(self, x, t):
        if self.kind == "multitone":
            a = time_features(t, self.tones) @ self.coefficients
        else:
            a = torch.sin(2 * math.pi * t)
        if self.kind != "input":
            return a[:, None].expand(-1, self.projection.shape[1])
        features = torch.tanh(x.flatten(1) @ self.features)
        return a[:, None] * (features - self.center) / self.scale

    @torch.no_grad()
    def calibrate(self, x):
        features = torch.tanh(x.flatten(1) @ self.features)
        self.center.copy_(features.mean(0))
        self.scale.copy_(features.std(0, unbiased=False).clamp_min(.05))

    def signal(self, x, t, code, epsilon):
        return (epsilon * (self.gates(x, t) * code) @ self.projection.T).reshape_as(x)

    def scores(self, velocity, x, t):
        return (velocity.flatten(1) @ self.projection * self.gates(x, t)) @ self.codes.T


class Injected(nn.Module):
    def __init__(self, base, carrier, code, epsilon):
        super().__init__()
        self.base, self.carrier, self.epsilon = base, carrier, epsilon
        self.register_buffer("code", code)

    def forward(self, x, t, y=None):
        return security.flow.model_velocity(self.base, x, t, y) + self.carrier.signal(x, t, self.code, self.epsilon)


class BlindRegressor(nn.Module):
    """Public attack basis, with no access to OwnerCarrier or the clean model."""
    def __init__(self, kind, dimension, tones=8, features=128, bins=16, rank=1, seed=90411):
        super().__init__()
        self.kind, self.tones, self.bins, self.rank = kind, tones, bins, rank
        generator = torch.Generator().manual_seed(seed)
        matrix = torch.randn(dimension, features, generator=generator)
        matrix /= matrix.norm(dim=0, keepdim=True)
        self.register_buffer("features", matrix)
        self.register_buffer("weights", torch.empty(0))

    def design(self, x, t):
        if self.kind == "bins":
            index = (t * self.bins).long().clamp(0, self.bins - 1)
            return F.one_hot(index, self.bins).to(x.dtype)
        temporal = time_features(t, self.tones)
        if self.kind == "input_ridge":
            features = torch.tanh(x.flatten(1) @ self.features)
            # Independent attacker features, modulated by public sine/cosine.
            fundamental = torch.stack((torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t)), 1)
            temporal = torch.cat((temporal, (features[:, :, None] * fundamental[:, None]).flatten(1)), 1)
        return torch.cat((torch.ones(len(x), 1, device=x.device), temporal), 1)

    @torch.no_grad()
    def fit(self, x, t, velocities, ridge=1e-3):
        design = self.design(x, t).double()
        gram = design.T @ design / len(x)
        rhs = design.T @ velocities.flatten(1).double() / len(x)
        regularizer = ridge * gram.diag().mean().clamp_min(1e-8)
        weights = torch.linalg.solve(gram + regularizer * torch.eye(len(gram), device=x.device), rhs)
        if self.kind == "bins":
            # Remove only the time-varying component, not the global velocity mean.
            counts = design.sum(0)
            weights -= (counts[:, None] * weights).sum(0) / counts.sum()
            u, s, vh = torch.linalg.svd(weights, full_matrices=False)
            self.singular_values = s.cpu().tolist()
            weights = (u[:, :self.rank] * s[:self.rank]) @ vh[:self.rank]
        else:
            # Fit the intercept as a nuisance term, but do not subtract it.
            weights[0].zero_()
        self.weights = weights.float()
        return self

    def forward(self, x, t):
        return (self.design(x, t) @ self.weights).reshape_as(x)


class Removed(nn.Module):
    def __init__(self, model, estimator):
        super().__init__()
        self.model, self.estimator = model, estimator

    def forward(self, x, t, y=None):
        return security.flow.model_velocity(self.model, x, t, y) - self.estimator(x, t)


@torch.no_grad()
def collect(model, pool, labels, count, batch, seed):
    generator = torch.Generator(device=pool.device).manual_seed(seed)
    xs, ts, velocities = [], [], []
    for start in range(0, count, batch):
        x, t, y, _ = security.query_batch(pool, labels, min(batch, count - start), generator)
        xs.append(x)
        ts.append(t)
        velocities.append(security.flow.model_velocity(model, x, t, y))
    return torch.cat(xs), torch.cat(ts), torch.cat(velocities)


def train(base, carrier, code, config, pool, labels, args, directory):
    torch.manual_seed(args.seed)
    model = prepare_model(base, config, "lora")
    final = directory / "trained_adapter.pt"
    if final.exists():
        security.flow.load_adapter(model, final, pool.device)
        return model.eval().requires_grad_(False)
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    generator = torch.Generator(device=pool.device).manual_seed(args.seed)
    history = []
    for step in range(1, args.steps + 1):
        x, t, y, target = security.query_batch(pool, labels, args.train_batch, generator)
        prediction = security.flow.model_velocity(model, x, t, y)
        with torch.no_grad():
            clean = security.flow.model_velocity(base, x, t, y)
            gates = carrier.gates(x, t)
            desired = config["wm_eps"] * gates * code
            signal = (desired @ carrier.projection.T).reshape_as(x)
        coordinates = (prediction - clean).flatten(1) @ carrier.projection
        correlation = ((coordinates * code) * gates).sum(1).mean()
        loss = (F.mse_loss(prediction, target + signal)
                + config["wm_proj_weight"] * F.mse_loss(coordinates, desired)
                - config["wm_lambda"] * torch.tanh(correlation / (.5 * config["wm_eps"]) * config["wm_tanh_scale"]))
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite embedding loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            row = dict(step=step, loss=float(loss.detach()))
            history.append(row)
            print(f"TRAIN {carrier.kind} {step}/{args.steps}: {row['loss']:.6f}", flush=True)
            security.write_json(directory / "training.json", dict(history=history))
    # Use the original adapter serializer, including the extra trained layers.
    security.flow.save_adapter(model, final)
    return model.eval().requires_grad_(False)


@torch.no_grad()
def evaluate(model, base, owner, pool, labels, reference, args, destination):
    path = destination.with_suffix(".json")
    if path.exists():
        return json.loads(path.read_text())
    trials, squared_error, clean_energy, gates_total = [], 0., 0., None
    for trial in range(args.trials):
        generator = torch.Generator(device=pool.device).manual_seed(args.seed + 10000 + trial)
        scores = torch.zeros(owner.codes.shape[0], device=pool.device, dtype=torch.float64)
        for start in range(0, args.queries, args.eval_batch):
            x, t, y, _ = security.query_batch(pool, labels, min(args.eval_batch, args.queries - start), generator)
            v = security.flow.model_velocity(model, x, t, y)
            scores += owner.scores(v, x, t).double().sum(0)
            if trial == 0:
                clean = security.flow.model_velocity(base, x, t, y)
                squared_error += float((v - clean).double().square().sum())
                clean_energy += float(clean.double().square().sum())
                g = owner.gates(x, t).double()
                gates_total = g.sum(0) if gates_total is None else gates_total + g.sum(0)
        decoded = ''.join(str(int(b)) for b in (scores > 0).tolist())
        trials.append(dict(decoded=decoded, exact=decoded == owner.message,
                           bit_accuracy=sum(a == b for a, b in zip(decoded, owner.message)) / len(owner.message),
                           scores=(scores / args.queries).cpu().tolist()))
    samples, _ = security.generate(model, args.samples, args.sample_steps, args.sample_batch,
                                   args.seed + 20000, pool.device)
    features = security.flow.inception_features(samples, pool.device, args.feature_batch)
    fid = pilot_fid(reference["real_features"], features)
    metrics = dict(recovery_percent=100 * np.mean([r["exact"] for r in trials]),
                   bit_accuracy_percent=100 * np.mean([r["bit_accuracy"] for r in trials]),
                   fid=float(fid), fid_ratio_clean=float(fid / reference["clean_fid"]),
                   velocity_relative_mse=squared_error / max(clean_energy, 1e-20),
                   gate_mean_norm=float((gates_total / args.queries).norm()), trials=trials)
    save_image(samples.add(1).div(2), destination.with_suffix(".png"), nrow=10)
    np.savez_compressed(destination.with_suffix(".npz"), features=features.numpy(), samples=samples.numpy())
    security.write_json(path, metrics)
    print(f"EVAL {destination}: exact={metrics['recovery_percent']:.1f}% "
          f"bits={metrics['bit_accuracy_percent']:.1f}% FID/clean={metrics['fid_ratio_clean']:.3f}", flush=True)
    return metrics


def summarize(output, results):
    security.write_json(output / "results.json", results)
    lines = ["# Secret Carrier Pilot", "", f"State: {results['state']}", "",
             "One H32 message/key on the CIFAR UNet. Injected = exact analytic perturbation; "
             "trained = new short LoRA embedding. These are different controls, not pooled results. "
             "FID uses a small paired pilot sample, not a publication-quality estimate. "
             "Input features are centered globally on independent calibration queries; "
             "zero mean at each time and density preservation are not assumed. "
             "Attacks use only marked responses, not owner keys or clean responses. "
             "All attack choices are reported, with no owner-guided selection.", "",
             "| Field | Carrier | Attack | Exact (%) | Bits (%) | FID/clean | Velocity relative MSE |",
             "|---|---|---|---:|---:|---:|---:|"]
    for row in results["rows"]:
        m = row["metrics"]
        lines.append(f"| {row['field']} | {row['carrier']} | {row['attack']} | "
                     f"{m['recovery_percent']:.1f} | {m['bit_accuracy_percent']:.1f} | "
                     f"{m['fid_ratio_clean']:.3f} | {m['velocity_relative_mse']:.6f} |")
    (output / "report.md").write_text('\n'.join(lines) + '\n')


def run(args):
    from huggingface_hub import hf_hub_download
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    source = Path("table_runs/watermark_security")
    original = json.loads((source / "config.json").read_text())
    job = next(j for j in original["plan"] if j["scheme"] == "H32")
    key_path = Path(job["run_dir"]) / "watermark_key.pt"
    key = torch.load(key_path, weights_only=False, map_location=device)
    c = job["config"]
    cfg = security.flow.DEFAULT_CONFIGS["cifar10"]
    checkpoint = Path(hf_hub_download("WayBob/FlowMatching-Unet-Celeb-64x64", cfg.filename, local_files_only=True))
    sources = [Path(__file__), Path(security.__file__), Path(security.flow.__file__),
               source_path('run_embedding_strategy_robustness.py')]
    config = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  sources={str(p): security.digest(p) for p in sources},
                  base_sha256=security.digest(checkpoint), key_sha256=security.digest(key_path),
                  embedding=c, attack_information="Marked velocities and public query inputs only",
                  attacks=["joint sin/cos regression, frequencies 1..8", "16-bin SVD rank 1 and 4",
                           "128 independent tanh features times sin/cos plus the temporal basis"])
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Resume requires identical settings and sources")
    security.write_json(config_path, config)
    result_path = args.output / "results.json"
    results = json.loads(result_path.read_text()) if result_path.exists() else dict(rows=[])
    results.update(state="running", expected_evaluations=len(args.carriers) * (len(args.fields) * 5 + 1))
    results.pop("error", None)
    summarize(args.output, results)
    dataset = CIFAR10(str(security.ROOT / "data"), train=True, download=False)
    pool = torch.from_numpy(dataset.data[:20000]).permute(0, 3, 1, 2).to(device).float() / 127.5 - 1
    labels = torch.tensor(dataset.targets[:20000], device=device)
    base = security.flow.build_model(cfg, device)
    security.flow.load_state_dict_flexible(base, str(checkpoint), device)
    base.eval().requires_grad_(False)
    projection, code = code_from_key(key, c, device)
    secret_path = args.output / "pilot_secret.pt"
    if secret_path.exists():
        private = torch.load(secret_path, map_location=device, weights_only=True)
    else:
        seed = secrets.randbits(63)
        generator = torch.Generator(device=device).manual_seed(seed)
        fresh_projection = torch.linalg.qr(torch.randn(projection.shape, generator=generator, device=device)).Q
        private = dict(seed=seed, projection=fresh_projection)
        save_torch(secret_path, private)
        secret_path.chmod(0o600)
    projection = private["projection"]
    manifest = args.output / "secret_manifest.json"
    fingerprint = dict(sha256=security.digest(secret_path))
    if manifest.exists() and json.loads(manifest.read_text()) != fingerprint:
        raise ValueError("Private pilot key changed; cached evaluations cannot be reused")
    security.write_json(manifest, fingerprint)
    quality_path = args.output / "quality_reference.pt"
    try:
        if quality_path.exists():
            reference = torch.load(quality_path, weights_only=True, map_location="cpu")
        else:
            test = CIFAR10(str(security.ROOT / "data"), train=False, download=False)
            indices = np.concatenate([np.flatnonzero(np.asarray(test.targets) == y)[:args.samples // 10] for y in range(10)])
            real = torch.from_numpy(test.data[indices]).permute(0, 3, 1, 2).float() / 127.5 - 1
            real_features = security.flow.inception_features(real, device, args.feature_batch)
            samples, _ = security.generate(base, args.samples, args.sample_steps, args.sample_batch,
                                           args.seed + 20000, device)
            features = security.flow.inception_features(samples, device, args.feature_batch)
            reference = dict(real_features=real_features,
                             clean_fid=pilot_fid(real_features, features))
            save_torch(quality_path, reference)
        for kind in args.carriers:
            directory = args.output / kind
            directory.mkdir(exist_ok=True)
            owner = OwnerCarrier(kind, projection, key["codes"], key["message"], seed=private["seed"]).to(device)
            generator = torch.Generator(device=device).manual_seed(args.seed + 30000)
            calibration, _, _, _ = security.query_batch(pool, labels, 4096, generator)
            owner.calibrate(calibration)
            del calibration
            save_torch(directory / "owner_key.pt", dict(state=owner.state_dict(), message=owner.message, kind=kind))
            (directory / "owner_key.pt").chmod(0o600)
            if not any(r["carrier"] == kind and r["field"] == "clean" for r in results["rows"]):
                metrics = evaluate(base, base, owner, pool, labels, reference, args, directory / "clean")
                results["rows"].append(dict(carrier=kind, field="clean", attack="none", metrics=metrics))
                summarize(args.output, results)
            for field in args.fields:
                output = directory / field
                output.mkdir(exist_ok=True)
                results["current"] = dict(carrier=kind, field=field)
                summarize(args.output, results)
                model = (Injected(base, owner, code, c["wm_eps"]) if field == "injected"
                         else train(base, owner, code, c, pool, labels, args, directory))
                model.eval().requires_grad_(False)
                conditions = [("marked", None), ("joint_spectrum", ("time", 0)),
                              ("time_svd_rank1", ("bins", 1)), ("time_svd_rank4", ("bins", 4)),
                              ("input_ridge", ("input_ridge", 0))]
                attack_data = None
                for name, specification in conditions:
                    if any(r["carrier"] == kind and r["field"] == field and r["attack"] == name for r in results["rows"]):
                        continue
                    print(f"START {kind}/{field}/{name}", flush=True)
                    attacked = model
                    if specification is not None:
                        if attack_data is None:
                            attack_data = collect(model, pool, labels, args.attack_queries,
                                                  args.eval_batch, args.seed + 40000)
                        estimator = BlindRegressor(specification[0], 3072, rank=specification[1]).to(device)
                        estimator.fit(*attack_data)
                        save_torch(output / (name + "_estimate.pt"), dict(state=estimator.state_dict(),
                                   singular_values=getattr(estimator, "singular_values", [])))
                        attacked = Removed(model, estimator).eval()
                    metrics = evaluate(attacked, base, owner, pool, labels, reference, args, output / name)
                    results["rows"].append(dict(carrier=kind, field=field, attack=name, metrics=metrics))
                    summarize(args.output, results)
                    del attacked
                del model, attack_data
                torch.cuda.empty_cache()
        results.update(state="complete", current=None)
    except Exception as exc:
        results.update(state="failed", error=repr(exc))
        raise
    finally:
        summarize(args.output, results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, default=Path("table_runs/secret_carrier_pilot"))
    p.add_argument("--fields", nargs="+", choices=["injected", "trained"], default=["injected", "trained"])
    p.add_argument("--carriers", nargs="+", choices=["sine", "multitone", "input"], default=["sine", "multitone", "input"])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--train-batch", type=int, default=64)
    p.add_argument("--attack-queries", type=int, default=8192)
    p.add_argument("--queries", type=int, default=4096)
    p.add_argument("--trials", type=int, default=4)
    p.add_argument("--eval-batch", type=int, default=256)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--sample-batch", type=int, default=100)
    p.add_argument("--feature-batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=930042)
    args = p.parse_args()
    if min(args.steps, args.train_batch, args.attack_queries, args.queries, args.trials,
           args.eval_batch, args.samples, args.sample_steps, args.sample_batch, args.feature_batch) < 1:
        p.error("Counts must be positive")
    if args.samples % 10 or args.samples > 10000:
        p.error("Samples must be divisible by 10 and at most 10000")
    if len(set(args.fields)) != len(args.fields) or len(set(args.carriers)) != len(args.carriers):
        p.error("Duplicate field/carrier choices")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "study.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
