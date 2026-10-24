"""Paired unwatermarked fine-tuning, FTSS, replication, and exposure measurements."""

import argparse
import ast
import fcntl
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import rankdata
import torch
from torchvision.datasets import MNIST

from repair_mlp_watermarks import FlowModel
from run_ftss_finetuning import digest, load_ftss, write_json


CHECKPOINTS = {
    "mlp": Path("table_runs/main_architectures/mlp/orthogonal5/clean_model_0.pt"),
    "unet": Path("table_runs/checkpoints/classic_unet_mnist/clean_mnist.pt"),
}


def load_model(name, device):
    checkpoint = torch.load(CHECKPOINTS[name], map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    if name == "mlp":
        return FlowModel(state).to(device)
    # The original training module executes on import; load only its model class.
    tree = ast.parse(Path("flow_watermark_unet_lora.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SimpleUNet")
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"torch": torch, "nn": torch.nn}
    exec(compile(module, "flow_watermark_unet_lora.py", "exec"), namespace)
    model = namespace["SimpleUNet"](28, 1, state["enc1.0.weight"].shape[0])
    model.load_state_dict(state, strict=True)
    return model.to(device)


def balanced_split(labels, size, seed, calibration_count=800):
    if size % 10 or calibration_count % 10:
        raise ValueError("Pool sizes must be multiples of ten")
    rng = np.random.default_rng(seed)
    result = {k: [] for k in ("tuned", "reference", "calibration")}
    n, c = size // 10, calibration_count // 10
    for label in range(10):
        indices = rng.permutation(np.flatnonzero(np.asarray(labels) == label))
        if 2 * n + c > len(indices):
            raise ValueError(f"Insufficient held-out examples of digit {label}")
        result["tuned"].extend(indices[:n].tolist())
        result["reference"].extend(indices[n:2*n].tolist())
        result["calibration"].extend(indices[2*n:2*n+c].tolist())
    return result


def balanced_audit(pool, limit=200):
    each = min(len(pool), limit) // 10
    return pool.reshape(10, -1, pool.shape[-1])[:, :each].reshape(-1, pool.shape[-1])


def membership_auc(member_losses, reference_losses):
    m, n = len(member_losses), len(reference_losses)
    ranks = rankdata(-np.r_[member_losses, reference_losses], method="average")
    return float((ranks[:m].sum() - m * (m + 1) / 2) / (m * n))


@torch.no_grad()
def nearest(queries, pool, batch=128):
    distances, indices = [], []
    for part in queries.split(batch):
        d, i = torch.cdist(part, pool).min(dim=1)
        distances.append(d)
        indices.append(i)
    return torch.cat(distances).numpy(), torch.cat(indices).numpy()


@torch.no_grad()
def per_image_loss(model, pool, device, seed, repeats, batch):
    generator = torch.Generator(device=device).manual_seed(seed)
    total = torch.zeros(len(pool), device=device)
    for _ in range(repeats):
        for start in range(0, len(pool), batch):
            target = pool[start:start+batch].to(device)
            noise = torch.randn(target.shape, generator=generator, device=device)
            times = torch.rand(len(target), 1, generator=generator, device=device)
            prediction = model((1-times)*noise + times*target, times)
            total[start:start+len(target)] += (prediction - (target-noise)).square().mean(dim=1)
    return (total / repeats).cpu().numpy()


def copy_metrics(samples, tuned, reference, calibration, alphas=(0.01, 0.05)):
    calibration_distance, _ = nearest(calibration, tuned)
    thresholds = {alpha: float(np.quantile(calibration_distance, alpha)) for alpha in alphas}
    metrics, audit = {}, {}
    for name, visible in (("raw", samples), ("visible", samples.clamp(-1, 1))):
        train_d, train_i = nearest(visible, tuned)
        reference_d, _ = nearest(visible, reference)
        preference = train_d < reference_d
        metrics[f"{name}_train_preference"] = float(preference.mean())
        metrics[f"{name}_neighbor_advantage"] = float(
            np.mean((reference_d-train_d) / np.maximum(reference_d+train_d, 1e-12)))
        for alpha, threshold in thresholds.items():
            match = preference & (train_d < threshold)
            suffix = str(round(alpha*100))
            metrics[f"{name}_copy_rate_{suffix}pct"] = float(match.mean())
            metrics[f"{name}_copy_coverage_{suffix}pct"] = float(
                len(np.unique(train_i[match])) / len(tuned))
            metrics[f"copy_distance_threshold_{suffix}pct"] = threshold
        audit[f"{name}_train_distances"] = train_d
        audit[f"{name}_reference_distances"] = reference_d
        audit[f"{name}_train_nearest_indices"] = train_i
    metrics["generated_saturation_fraction"] = float((samples.abs() >= 4.999).float().mean())
    return metrics, audit


def evaluate(model, pools, generation_noise, probe_noise, directions, ftss, args, seed):
    model.eval()
    samples = ftss.generate(model, generation_noise, args.ode_steps, args.eval_batch)
    curve, minimum = ftss.ftss(model, probe_noise, directions, args.ode_steps,
                               args.time_points, args.epsilon_scale)
    tuned, reference, calibration = pools
    metrics, audit = copy_metrics(samples, tuned, reference, calibration)
    losses = [per_image_loss(model, p, generation_noise.device, seed + 8000,
                             args.loss_repeats, args.eval_batch)
              for p in (balanced_audit(tuned), balanced_audit(reference), calibration)]
    member, nonmember, calibration_loss = losses
    metrics.update(g_min=minimum, ftss_curve=curve,
                   member_loss=float(member.mean()), reference_loss=float(nonmember.mean()),
                   loss_gap=float(nonmember.mean()-member.mean()),
                   membership_auc=membership_auc(member, nonmember),
                   loss_audit_members=len(member), loss_audit_nonmembers=len(nonmember))
    for alpha in (0.01, 0.05):
        threshold = float(np.quantile(calibration_loss, alpha))
        metrics[f"membership_tpr_{round(alpha*100)}pct"] = float((member < threshold).mean())
        metrics[f"membership_fpr_{round(alpha*100)}pct"] = float((nonmember < threshold).mean())
    audit.update(generated_samples=samples.numpy(), member_losses=member,
                 reference_losses=nonmember, calibration_losses=calibration_loss)
    return metrics, audit


def parser():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, default=Path("table_runs/unmarked_memorization"))
    p.add_argument("--ftss-repo", type=Path, default=Path("../ftss-memorization"))
    p.add_argument("--models", nargs="+", choices=list(CHECKPOINTS), default=["mlp", "unet"])
    p.add_argument("--sizes", nargs="+", type=int, default=[10, 50, 1000, 4000])
    p.add_argument("--steps", nargs="+", type=int, default=[0, 300, 1000, 3000])
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--n-generated", type=int, default=256)
    p.add_argument("--n-probes", type=int, default=15)
    p.add_argument("--time-points", type=int, default=20)
    p.add_argument("--ode-steps", type=int, default=200)
    p.add_argument("--epsilon-scale", type=float, default=0.1)
    p.add_argument("--loss-repeats", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if (min(args.sizes) < 10 or max(args.sizes) > 4000 or any(n % 10 for n in args.sizes)
            or min(args.steps) < 0 or max(args.steps) < 1 or args.runs < 1 or args.lr <= 0
            or min(args.batch_size, args.eval_batch, args.n_generated, args.n_probes,
                   args.loss_repeats) < 1 or args.ode_steps < args.time_points
            or args.time_points < 2 or args.epsilon_scale <= 0):
        p.error("Invalid study parameters")
    args.steps = sorted(set(args.steps) | {0})
    args.sizes = sorted(set(args.sizes))
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    ftss = load_ftss(args.ftss_repo.resolve())
    dataset = MNIST("data", train=False, download=False)
    images = dataset.data.flatten(1).float() / 127.5 - 1
    labels = dataset.targets.numpy()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"}
    config.update(checkpoints={m: {"path": str(CHECKPOINTS[m]), "sha256": digest(CHECKPOINTS[m])}
                               for m in args.models},
                  ftss_source_sha256=digest(Path(ftss.__file__)),
                  model_source_sha256={"mlp": digest(Path("repair_mlp_watermarks.py")),
                                       "unet": digest(Path("flow_watermark_unet_lora.py"))},
                  unwatermarked_only=True, dataset="MNIST test split, excluded from base training",
                  calibration_images=800, pools="Disjoint, equal tuned/reference sizes, balanced digits",
                  metrics={"raw_copy": "Original FTSS repo copying rule at 1% and 5% calibration quantiles",
                           "visible_copy": "Same rule after clipping generated pixels to [-1,1]",
                           "neighbor_preference": "Generated sample closer to tuned than equal-size reference pool",
                           "membership": "AUROC using negative MC flow loss, plus calibrated TPR/FPR",
                           "loss_gap": "Reference mean flow loss minus tuned mean flow loss"},
                  ftss_protocol="Upstream finite differences with velocity [-10,10], states [-5,5]")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "study.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = args.output / "config.json"
        if config_path.exists():
            if not args.resume or json.loads(config_path.read_text()) != config:
                raise ValueError("Existing study requires --resume with exactly the same protocol")
            report = json.loads((args.output / "results.json").read_text())
        else:
            write_json(config_path, config)
            report = {"state": "running", "measurements": [], "elapsed_seconds": 0}
        begin = time.monotonic()
        previous_elapsed = report["elapsed_seconds"]
        report.update(state="running")
        report.pop("error", None)
        try:
            for name in args.models:
                for run in range(args.runs):
                    seed = args.seed + 1000 * run
                    evaluation_rng = torch.Generator(device=device).manual_seed(seed)
                    generation_noise = torch.randn(args.n_generated, 784, device=device, generator=evaluation_rng)
                    probe_noise = torch.randn(args.n_probes, 784, device=device, generator=evaluation_rng)
                    directions = torch.randn(args.n_probes, 784, device=device, generator=evaluation_rng)
                    for size in args.sizes:
                        prefix = f"{name}_n{size}_run{run}"
                        split = balanced_split(labels, size, seed)
                        write_json(args.output / f"{prefix}_indices.json", split)
                        pools = [images[split[k]] for k in ("tuned", "reference", "calibration")]
                        pool = pools[0].to(device)
                        rows = [r for r in report["measurements"] if
                                (r["model"], r["run"], r["size"]) == (name, run, size)]
                        if any(r["step"] == max(args.steps) for r in rows):
                            continue
                        torch.manual_seed(seed)
                        model = load_model(name, device)
                        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
                        training_rng = torch.Generator(device=device).manual_seed(seed + 100)
                        resume_path = args.output / f"{prefix}_latest.pt"
                        start_step = 0
                        if resume_path.exists():
                            state = torch.load(resume_path, map_location=device, weights_only=True)
                            model.load_state_dict(state["model"])
                            optimizer.load_state_dict(state["optimizer"])
                            training_rng.set_state(state["generator"].cpu())
                            start_step = state["step"]

                        def measure(step):
                            metrics, audit = evaluate(model, pools, generation_noise, probe_noise,
                                                      directions, ftss, args, seed)
                            baseline = next((r for r in rows if r["step"] == 0), metrics)
                            ratio = metrics["g_min"] / baseline["g_min"]
                            profile_ratio = np.maximum(metrics["ftss_curve"], 1e-12) / np.maximum(
                                baseline["ftss_curve"], 1e-12)
                            row = {"model": name, "run": run, "size": size, "step": step,
                                   **metrics, "ftss_ratio": ratio,
                                   "contraction": -math.log(max(ratio, 1e-12)),
                                   "profile_contraction": float(-np.log(profile_ratio).mean())}
                            np.savez_compressed(args.output / f"{prefix}_step{step}_audit.npz", **audit)
                            snapshot = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                        "generator": training_rng.get_state(), "step": step}
                            temporary = resume_path.with_suffix(".tmp")
                            torch.save(snapshot, temporary)
                            temporary.replace(resume_path)
                            rows.append(row)
                            report["measurements"].append(row)
                            report["elapsed_seconds"] = previous_elapsed + time.monotonic() - begin
                            write_json(args.output / "results.json", report)
                            print(f"{prefix} step={step} FTSS/base={ratio:.4f} "
                                  f"rawcopy={metrics['raw_copy_rate_1pct']:.3f} "
                                  f"visiblecopy={metrics['visible_copy_rate_1pct']:.3f} "
                                  f"NN={metrics['visible_train_preference']:.3f} "
                                  f"AUC={metrics['membership_auc']:.3f} "
                                  f"gap={metrics['loss_gap']:.4f}", flush=True)

                        if not rows or not any(r["step"] == start_step for r in rows):
                            measure(start_step)
                        for step in range(start_step + 1, max(args.steps) + 1):
                            model.train()
                            indices = torch.randint(size, (min(args.batch_size, size),),
                                                    device=device, generator=training_rng)
                            target = pool[indices]
                            noise = torch.randn(target.shape, generator=training_rng, device=device)
                            times = torch.rand(len(target), 1, generator=training_rng, device=device)
                            prediction = model((1-times)*noise + times*target, times)
                            loss = (prediction - (target-noise)).square().mean()
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                            optimizer.step()
                            if step in args.steps:
                                measure(step)
                        del model, optimizer, pool
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
            report["state"] = "complete"
        except Exception as exc:
            report.update(state="failed", error=str(exc))
            raise
        finally:
            report["elapsed_seconds"] = previous_elapsed + time.monotonic() - begin
            write_json(args.output / "results.json", report)


if __name__ == "__main__":
    main()
