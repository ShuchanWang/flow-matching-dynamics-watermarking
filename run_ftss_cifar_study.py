"""FTSS and behavioral outcomes during clean pretrained CIFAR-10 fine-tuning."""

import argparse
import copy
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

import watermark_hf_flow_unet as flow
from evaluate_generative_generalization import prdc, semantic_diversity
from ftss_batched import estimate_ftss_batched
from run_ftss_finetuning import digest, write_json
from run_ftss_outcome_study import correlation, make_split
from run_matched_ftss_mlp import numeric
from run_unmarked_memorization import membership_auc


ROOT = Path(__file__).resolve().parent
OUTCOMES = ("training_preference", "training_anchor_advantage", "loss_gap", "membership_auc",
            "coverage", "precision", "style_diversity_ratio", "pixel_training_preference")


class FixedClassFlow(torch.nn.Module):
    def __init__(self, model, shape, labels, batch=128):
        super().__init__()
        self.model, self.shape, self.labels, self.batch = model, shape, labels, batch

    def forward(self, x, t):
        if len(x) % len(self.labels):
            raise ValueError("FTSS branch batch does not preserve trajectory class ordering")
        labels = self.labels.repeat(len(x) // len(self.labels))
        outputs = []
        for start in range(0, len(x), self.batch):
            sl = slice(start, start+self.batch)
            velocity = flow.model_velocity(self.model, x[sl].reshape(-1, *self.shape),
                                           t[sl].flatten(), labels[sl])
            outputs.append(velocity.flatten(1))
        return torch.cat(outputs)


def conditional_neighbors(generated, labels, bank, bank_labels):
    distances = np.empty(len(generated))
    indices = np.empty(len(generated), dtype=int)
    for label in np.unique(labels):
        query = np.flatnonzero(labels == label)
        anchors = np.flatnonzero(bank_labels == label)
        if not len(anchors):
            raise ValueError("Missing reference class")
        a = torch.as_tensor(generated[query], dtype=torch.float32)
        b = torch.as_tensor(bank[anchors], dtype=torch.float32)
        d, i = torch.cdist(a, b).min(1)
        distances[query] = d.numpy()
        indices[query] = anchors[i.numpy()]
    return distances, indices


@torch.no_grad()
def generate(model, device, count, steps, batch, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    labels = torch.arange(count, device=device) % 10
    noise = torch.randn(count, 3, 32, 32, generator=generator, device=device)
    outputs = []
    model.eval()
    for start in range(0, count, batch):
        x = noise[start:start+batch].clone()
        y = labels[start:start+batch]
        for step in range(steps):
            t = torch.full((len(x),), step/steps, device=device)
            x = (x + flow.model_velocity(model, x, t, y)/steps).clamp(-5, 5)
        outputs.append(x.clamp(-1, 1).cpu())
    return torch.cat(outputs), labels.cpu().numpy()


@torch.no_grad()
def flow_losses(model, images, labels, split, device, seed, repeats=8, batch=128):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    n = len(split["tuned"])
    losses = [torch.zeros(n, device=device), torch.zeros(n, device=device)]
    for _ in range(repeats):
        noise = torch.randn(n, 3, 32, 32, generator=generator, device=device)
        times = torch.rand(n, device=device, generator=generator)
        for k, part in enumerate(("tuned", "reference")):
            index = torch.tensor(split[part], device=device)
            for start in range(0, n, batch):
                sl = slice(start, start+batch)
                target, y = images[index[sl]], labels[index[sl]]
                t = times[sl]
                xt = (1-t[:, None, None, None])*noise[sl] + t[:, None, None, None]*target
                prediction = flow.model_velocity(model, xt, t, y)
                losses[k][sl] += (prediction-(target-noise[sl])).square().flatten(1).mean(1)/repeats
    values = [v.cpu().numpy() for v in losses]
    return dict(training_flow_loss=float(values[0].mean()), reference_flow_loss=float(values[1].mean()),
                loss_gap=float(values[1].mean()-values[0].mean()), membership_auc=membership_auc(*values)), values


def summarize(rows):
    base = {(r["seed"], r["size"]): r["metrics"] for r in rows if r["step"] == 0}
    changes = []
    for row in rows:
        if not row["step"]:
            continue
        before = base[row["seed"], row["size"]]
        m = row["metrics"]
        r = {k: row[k] for k in ("seed", "size", "step")}
        r.update(ftss_ratio=m["min"]/before["min"], contraction=float(-np.log(m["min"]/before["min"])))
        r.update({k+"_change": m[k]-before[k] for k in OUTCOMES})
        changes.append(r)
    associations = {}
    for outcome in OUTCOMES:
        def corr(selected):
            return correlation([r["contraction"] for r in selected], [r[outcome+"_change"] for r in selected])
        associations[outcome] = dict(pooled=corr(changes),
            by_step={str(s): corr([r for r in changes if r["step"] == s]) for s in sorted({r["step"] for r in changes})},
            by_size={str(s): corr([r for r in changes if r["size"] == s]) for s in sorted({r["size"] for r in changes})})
    return dict(changes=changes, associations=associations)


class Measurement:
    def __init__(self, args, device, images, labels):
        self.args, self.device, self.images, self.labels = args, device, images, labels
        cache = args.output / "real_features.pt"
        if cache.exists():
            self.real = torch.load(cache, map_location="cpu", weights_only=True)
        else:
            print("Extracting held-out CIFAR-10 Inception features", flush=True)
            self.real = flow.inception_features(images.cpu(), device, args.feature_batch)
            torch.save(self.real, cache)
        self.bank = self.real.numpy()

    def controls(self, seed):
        path = self.args.output / f"quality_controls_seed{seed}.json"
        if path.exists():
            return
        labels = self.labels.cpu().numpy()
        master = make_split(labels, 1000, 71000+seed)
        excluded = np.concatenate([master[k] for k in ("tuned", "reference", "calibration")])
        rng = np.random.default_rng(111000+seed)
        indices = []
        for label in range(10):
            available = np.setdiff1d(np.flatnonzero(labels == label), excluded)
            if len(available) < self.args.samples // 10:
                raise ValueError("Insufficient disjoint real images for quality controls")
            indices.extend(rng.permutation(available)[:self.args.samples // 10].tolist())
        reference = self.bank[master["calibration"]]
        other = self.bank[indices]
        result = prdc(reference, other, 3)
        result["fid"] = flow.compute_fid_from_features(torch.from_numpy(reference), torch.from_numpy(other))
        result.update(reference_indices=master["calibration"], comparison_indices=indices,
                      purpose="Disjoint real-versus-real finite-sample quality control, not model selection")
        write_json(path, result)

    @torch.no_grad()
    def __call__(self, model, split, seed, name):
        args, device = self.args, self.device
        started = time.monotonic()
        model.eval()
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            probe_labels = torch.arange(args.probes, device=device) % 10
            wrapper = FixedClassFlow(model, (3, 32, 32), probe_labels, args.eval_batch)
            m = numeric(estimate_ftss_batched(wrapper, device, args.probes, args.ode_steps, 3072))
            samples, sample_labels = generate(model, device, args.samples, args.sample_steps, args.eval_batch, seed+1)
            fake = flow.inception_features(samples, device, args.feature_batch).numpy()
            labels = self.labels.cpu().numpy()
            train_d, train_i = conditional_neighbors(fake, sample_labels, self.bank[split["tuned"]], labels[split["tuned"]])
            ref_d, _ = conditional_neighbors(fake, sample_labels, self.bank[split["reference"]], labels[split["reference"]])
            m.update(training_preference=float((train_d < ref_d).mean()),
                     training_anchor_advantage=float(np.log((np.square(ref_d).mean()+1e-24)/(np.square(train_d).mean()+1e-24))))
            pixels = samples.flatten(1).numpy()
            tr_pixel, _ = conditional_neighbors(pixels, sample_labels, self.images[split["tuned"]].cpu().flatten(1).numpy(), labels[split["tuned"]])
            ref_pixel, _ = conditional_neighbors(pixels, sample_labels, self.images[split["reference"]].cpu().flatten(1).numpy(), labels[split["reference"]])
            m["pixel_training_preference"] = float((tr_pixel < ref_pixel).mean())
            # RMSE is measured in visible [0,1] RGB pixels; both proximity rules
            # additionally require a nearer tuned than equal-size reference image.
            rmse = tr_pixel / (2*np.sqrt(3072))
            m["pixel_copy_rmse_005"] = float(((rmse <= .05) & (tr_pixel < ref_pixel)).mean())
            m["pixel_copy_rmse_010"] = float(((rmse <= .10) & (tr_pixel < ref_pixel)).mean())
            reference = self.bank[split["calibration"]]
            m.update(prdc(reference, fake, 3))
            style = semantic_diversity(fake, sample_labels, reference, labels[split["calibration"]])
            # Classes here are requested conditions, not inferred semantic labels.
            m["style_diversity_ratio"] = style["style_diversity_ratio"]
            loss, arrays = flow_losses(model, self.images, self.labels, split, device, seed+2)
            m.update(loss)
            m["fid"] = flow.compute_fid_from_features(torch.from_numpy(reference), torch.from_numpy(fake))
            m["measurement_seconds"] = time.monotonic()-started
            np.savez_compressed(args.output / f"{name}_audit.npz", samples=samples.numpy(), labels=sample_labels,
                                features=fake, training_distances=train_d, reference_distances=ref_d,
                                training_anchor_indices=np.asarray(split["tuned"])[train_i],
                                member_losses=arrays[0], reference_losses=arrays[1], pixel_training_rmse=rmse)
            save_image(samples[:100].add(1).div(2), args.output / f"{name}_samples.png", nrow=10)
            print(f"MEASURED {name}: FTSS={m['min']:.5f} preference={m['training_preference']:.3f} "
                  f"coverage={m['coverage']:.3f} FID={m['fid']:.2f} elapsed={m['measurement_seconds']:.0f}s", flush=True)
            return m


def save_report(args, report):
    write_json(args.output / "results.json", report)
    summary = summarize(report["rows"])
    write_json(args.output / "summary.json", summary)
    lines = ["# Pretrained CIFAR-10 RF-UNet: FTSS During Clean Fine-Tuning", "", f"State: {report['state']}", "",
             "One public clean class-conditional checkpoint, no watermark or adapters. Fine-tuning and evaluation "
             "pools use CIFAR-10 test images, disjoint from the checkpoint's standard training split. "
             "The source checkpoint's complete training provenance is not independently audited. "
             "Nested class-balanced tuning pools and equal-size untouched reference pools are disjoint from a fixed "
             "1000-image quality reference per subset seed. All evaluated weights are EMA (decay 0.999).", "",
             f"Sizes {args.sizes}; subset seeds {args.seeds}; checkpoints {args.check_steps}. Full-parameter AdamW "
             f"fine-tuning, LR {args.lr}, cosine decay over {args.steps} steps, batch {args.batch_size}. "
             f"FTSS uses {args.probes} class-balanced trajectories, 20 injection times, {args.ode_steps} Euler steps, "
             "relative perturbation 0.1, and the original state/velocity clamps. Each trajectory and all its "
             f"perturbations retain the same class condition. Generation uses {args.samples} balanced class conditions, "
             f"{args.sample_steps} Euler steps and the checkpoint's original sampling clamps. "
             "FTSS and generation use paired random inputs within each seed. Inception features provide quality, "
             "conditional within-class diversity and nearest-neighbor outcomes. Disjoint real-versus-real "
             "quality controls use matched reference/sample counts and are saved per seed. Pixel proximity is a secondary check. "
             "Preference is attraction, not proof of copying. Flow-loss gaps and membership AUC average eight noise/time draws.", "",
             "| Images | Steps | Runs | FTSS / Before | Preference Change | Loss Gap Change | Coverage Change | FID / Before |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for size, step in sorted({(r["size"], r["step"]) for r in summary["changes"]}):
        selected = [r for r in summary["changes"] if (r["size"], r["step"]) == (size, step)]
        cells = []
        for key in ("ftss_ratio", "training_preference_change", "loss_gap_change", "coverage_change"):
            v = [r[key] for r in selected]
            cells.append(f"{np.mean(v):.4f} +/- {np.std(v):.4f}")
        ratios = []
        for r in selected:
            before = next(x for x in report["rows"] if x["seed"] == r["seed"] and x["size"] == size and x["step"] == 0)
            after = next(x for x in report["rows"] if x["seed"] == r["seed"] and x["size"] == size and x["step"] == step)
            ratios.append(after["metrics"]["fid"]/before["metrics"]["fid"])
        cells.append(f"{np.mean(ratios):.3f} +/- {np.std(ratios):.3f}")
        lines.append(f"| {size} | {step} | {len(selected)} | " + " | ".join(cells) + " |")
    lines += ["", "Descriptive associations only: repeated checkpoints share one pretrained base. "
              "Positive contraction means -log(FTSS / before). Outcome changes subtract the paired baseline. "
              "By-size and by-step associations are retained in summary.json; pooled correlations can reflect "
              "the shared effect of data size. No independent-observation p-values are reported.", ""]
    lines += [f"- {k}: Spearman {v['pooled']}" for k, v in summary["associations"].items()]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")


def plot_results(output, rows):
    from matplotlib import pyplot as plt
    changes = summarize(rows)["changes"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    colors = {50: "#267b9d", 200: "#c34e40", 1000: "#38825c"}
    for ax, key, title in zip(axes, ("training_preference", "loss_gap", "style_diversity_ratio"),
                              ("Training preference change", "Flow-loss gap change", "Within-class diversity change")):
        for size in sorted({r["size"] for r in changes}):
            selected = [r for r in changes if r["size"] == size]
            ax.scatter([r["contraction"] for r in selected], [r[key+"_change"] for r in selected],
                       color=colors.get(size, "gray"), alpha=.75, label=f"{size} images")
        ax.axhline(0, lw=.7, color="gray")
        ax.axvline(0, lw=.7, color="gray")
        ax.set_xlabel("Contraction: -log(FTSS / before)")
        ax.set_ylabel(title)
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "outcome_associations.png", dpi=180)
    fig.savefig(output / "outcome_associations.pdf")
    plt.close(fig)


def run(args):
    from huggingface_hub import hf_hub_download
    device = torch.device(args.device)
    torch.set_num_threads(4)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    cfg = flow.DEFAULT_CONFIGS["cifar10"]
    checkpoint = Path(hf_hub_download("WayBob/FlowMatching-Unet-Celeb-64x64", cfg.filename, local_files_only=True))
    config = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "sources": {str(p): digest(p) for p in [checkpoint, Path(__file__), ROOT / "ftss_batched.py",
                ROOT / "watermark_hf_flow_unet.py", ROOT / "run_ftss_outcome_study.py"]},
              "clean_full_parameter_finetuning": True, "feature_space": "Original Inception IMAGENET1K_V1, 2048-D pool features"}
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Resume requires identical sources and protocol")
    write_json(config_path, config)
    path = args.output / "results.json"
    report = json.loads(path.read_text()) if path.exists() else dict(state="running", rows=[])
    dataset = CIFAR10(str(ROOT / "data"), train=False, download=False)
    images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).to(device).float().div(127.5).sub(1)
    labels = torch.tensor(dataset.targets, device=device)
    try:
        report.update(state="running")
        measurement = Measurement(args, device, images, labels)
        for seed in args.seeds:
            measurement.controls(seed)
            split_seed = 71000 + seed
            for size in args.sizes:
                prefix = f"seed{seed}_n{size}"
                rows = [r for r in report["rows"] if (r["seed"], r["size"]) == (seed, size)]
                if {r["step"] for r in rows}.issuperset(args.check_steps):
                    continue
                report["current"] = dict(seed=seed, size=size)
                save_report(args, report)
                split = make_split(np.asarray(dataset.targets), size, split_seed)
                write_json(args.output / f"{prefix}_indices.json", split)
                torch.manual_seed(81000+seed)
                model = flow.build_model(cfg, device)
                flow.load_state_dict_flexible(model, str(checkpoint), device)
                model.requires_grad_(True)
                ema = copy.deepcopy(model).eval().requires_grad_(False)
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
                pool = torch.tensor(split["tuned"], device=device)
                latest = args.output / f"{prefix}_latest.pt"
                step = 0
                if latest.exists():
                    saved = torch.load(latest, map_location="cpu", weights_only=True)
                    for key, obj in (("model", model), ("ema", ema), ("optimizer", optimizer), ("scheduler", scheduler)):
                        obj.load_state_dict(saved[key])
                    step = saved["step"]
                    torch.set_rng_state(saved["cpu_rng"])
                    if device.type == "cuda":
                        torch.cuda.set_rng_state(saved["cuda_rng"], device)
                    del saved

                def measure(current):
                    if any(r["step"] == current for r in rows):
                        return
                    metrics = measurement(ema, split, 91000+seed, f"{prefix}_step{current}")
                    row = dict(seed=seed, size=size, step=current, metrics=metrics)
                    report["rows"].append(row)
                    rows.append(row)
                    save_report(args, report)

                measure(step)
                started = time.monotonic()
                for current in range(step+1, args.steps+1):
                    model.train()
                    index = pool[torch.randint(len(pool), (args.batch_size,), device=device)]
                    target = images[index]
                    noise = torch.randn_like(target)
                    t = torch.rand(len(target), device=device)
                    xt = (1-t[:, None, None, None])*noise + t[:, None, None, None]*target
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.mse_loss(flow.model_velocity(model, xt, t, labels[index]), target-noise)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Nonfinite loss {prefix} step {current}")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    scheduler.step()
                    with torch.no_grad():
                        torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), .001)
                        for a, b in zip(ema.buffers(), model.buffers()):
                            a.copy_(b)
                    if current % 100 == 0:
                        print(f"TRAIN {prefix} {current}/{args.steps} loss={float(loss):.5f} elapsed={time.monotonic()-started:.0f}s", flush=True)
                    if current in args.check_steps:
                        saved = {key: obj.state_dict() for key, obj in (("model", model), ("ema", ema),
                                 ("optimizer", optimizer), ("scheduler", scheduler))}
                        saved.update(step=current, cpu_rng=torch.get_rng_state())
                        if device.type == "cuda":
                            saved["cuda_rng"] = torch.cuda.get_rng_state(device)
                        temp = latest.with_suffix(".tmp")
                        torch.save(saved, temp)
                        temp.replace(latest)
                        del saved
                        measure(current)
                del model, ema, optimizer
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        plot_results(args.output, report["rows"])
        report.update(state="complete")
        report.pop("current", None)
    except Exception as exc:
        report.update(state="failed", error=str(exc))
        raise
    finally:
        save_report(args, report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "table_runs/ftss_cifar_study")
    p.add_argument("--sizes", type=int, nargs="+", default=[50, 200, 1000])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--check-steps", type=int, nargs="+", default=[0, 300, 1000, 3000])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch", type=int, default=128)
    p.add_argument("--feature-batch", type=int, default=32)
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--sample-steps", type=int, default=100)
    p.add_argument("--probes", type=int, default=20)
    p.add_argument("--ode-steps", type=int, default=200)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if (min(args.steps, args.sample_steps, args.ode_steps, args.batch_size, args.eval_batch, args.feature_batch) < 1
            or args.probes < 10 or args.probes % 10 or args.samples < 10 or args.samples % 10
            or args.lr <= 0 or any(n < 10 or n > 1000 or n % 10 for n in args.sizes)
            or any(s < 0 or s > args.steps for s in args.check_steps)):
        p.error("Invalid sizes, steps or class-balanced probe/sample counts")
    args.check_steps = sorted(set(args.check_steps) | {0, args.steps})
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "study.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
