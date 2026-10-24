"""Paired clean MLP fine-tuning: FTSS versus independent generative outcomes."""

import argparse
import copy
import fcntl
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics.pairwise import euclidean_distances
import torch
from torch.nn import functional as F
from torchvision.datasets import MNIST

from evaluate_generative_generalization import (DigitEvaluator, features, prdc,
                                                semantic_diversity, training_preference)
from ftss_batched import estimate_ftss_batched
from run_ftss_finetuning import digest, load_ftss, write_json
from run_matched_ftss_mlp import ROOT, numeric, original_definitions
from run_unmarked_memorization import balanced_split, membership_auc


def make_split(labels, size, seed):
    # A shared held-out bank and nested tuning pools avoid changing the reference
    # distribution along with the number of fine-tuning examples.
    master = balanced_split(labels, 1000, seed, calibration_count=1000)
    tuned = np.asarray(master["tuned"]).reshape(10, 100)[:, :size // 10].reshape(-1).tolist()
    reference = np.asarray(master["reference"]).reshape(10, 100)[:, :size // 10].reshape(-1).tolist()
    return dict(tuned=tuned, reference=reference, calibration=master["calibration"])


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def summarize(rows):
    baseline = {(r["base"], r["seed"], r["size"]): r for r in rows if r["step"] == 0}
    changed = []
    for r in rows:
        if not r["step"]:
            continue
        before = baseline[r["base"], r["seed"], r["size"]]["metrics"]
        m = r["metrics"]
        row = {k: r[k] for k in ("base", "seed", "size", "step")}
        row["contraction"] = float(-np.log(m["min"] / before["min"]))
        row["ftss_ratio"] = m["min"] / before["min"]
        for key in ("training_preference", "training_anchor_advantage", "loss_gap", "membership_auc",
                    "coverage", "precision", "style_diversity_ratio", "effective_digit_modes"):
            row[key + "_change"] = m[key] - before[key]
        changed.append(row)
    outcomes = [k for k in changed[0] if k.endswith("_change")] if changed else []
    associations = {}
    for key in outcomes:
        associations[key] = {
            "pooled_spearman": correlation([r["contraction"] for r in changed], [r[key] for r in changed]),
            "by_step": {str(step): correlation([r["contraction"] for r in changed if r["step"] == step],
                                               [r[key] for r in changed if r["step"] == step])
                        for step in sorted({r["step"] for r in changed})},
            "by_base": {str(b): correlation([r["contraction"] for r in changed if r["base"] == b],
                                            [r[key] for r in changed if r["base"] == b])
                        for b in sorted({r["base"] for r in changed})}}
    return {"changes": changed, "associations": associations,
            "interpretation": "Descriptive associations; repeated checkpoints and shared bases are dependent. "
                              "Subset seeds are repeated fine-tuning realizations, not independent pretrained models."}


@torch.no_grad()
def loss_metrics(model, images, split, device, seed, repeats=8):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    n = len(split["tuned"])
    member = images[split["tuned"]].to(device)
    reference = images[split["reference"]].to(device)
    losses = [torch.zeros(n, device=device), torch.zeros(n, device=device)]
    for _ in range(repeats):
        noise = torch.randn(member.shape, device=device, generator=generator)
        t = torch.rand(n, 1, device=device, generator=generator)
        for j, target in enumerate((member, reference)):
            for start in range(0, n, 256):
                sl = slice(start, start + 256)
                pred = model((1-t[sl])*noise[sl] + t[sl]*target[sl], t[sl])
                losses[j][sl] += (pred - (target[sl]-noise[sl])).square().mean(1) / repeats
    values = [v.cpu().numpy() for v in losses]
    return {"training_flow_loss": float(values[0].mean()), "reference_flow_loss": float(values[1].mean()),
            "loss_gap": float(values[1].mean()-values[0].mean()),
            "membership_auc": membership_auc(*values)}, values


class Evaluator:
    def __init__(self, args, device):
        self.args, self.device = args, device
        dataset = MNIST(str(ROOT / "data"), train=False, download=False)
        self.images = dataset.data.flatten(1).float() / 127.5 - 1
        self.labels = dataset.targets.numpy()
        model = DigitEvaluator().to(device)
        model.load_state_dict(torch.load(args.classifier, map_location="cpu", weights_only=True)["model"])
        self.real = features(model.eval(), dataset.data[:, None].float() / 255, device)
        self.classifier = model.eval().requires_grad_(False)
        self.helpers = load_ftss(args.repo)

    @torch.no_grad()
    def __call__(self, model, split, seed, name):
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            model.eval()
            started = time.monotonic()
            torch.manual_seed(seed)
            m = numeric(estimate_ftss_batched(model, self.device, self.args.probes, self.args.ode_steps))
            torch.manual_seed(seed + 1)
            noise = torch.randn(self.args.samples, 784, device=self.device)
            samples = self.helpers.generate(model, noise, self.args.ode_steps, 256)
            generated = features(self.classifier, samples.clamp(-1, 1).reshape(-1, 1, 28, 28).add(1).div(2), self.device)
            reference = self.real["spatial"][split["calibration"]]
            m.update(prdc(reference, generated["spatial"], 3))
            m.update(semantic_diversity(generated["spatial"], generated["probabilities"].argmax(1),
                                        reference, self.labels[split["calibration"]]))
            m.update(training_preference(generated["spatial"], self.real["spatial"][split["tuned"]],
                                         self.real["spatial"][split["reference"]]))
            distances = [euclidean_distances(generated["spatial"], self.real["spatial"][split[k]], squared=True).min(1)
                         for k in ("tuned", "reference")]
            # This interval measures Monte Carlo sample uncertainty conditional on
            # this model, not uncertainty across pretrained models or subsets.
            hits = (distances[0] < distances[1]).astype(float)
            rng = np.random.default_rng(seed + 2)
            means = hits[rng.integers(len(hits), size=(1000, len(hits)))].mean(1)
            m["training_preference_mc_interval"] = np.quantile(means, [.025, .975]).tolist()
            loss, arrays = loss_metrics(model, self.images, split, self.device, seed + 3)
            m.update(loss)
            m["measurement_seconds"] = time.monotonic()-started
            np.savez_compressed(self.args.output / f"{name}_audit.npz", samples=samples.numpy(),
                                **generated, training_distances_squared=distances[0],
                                reference_distances_squared=distances[1], member_losses=arrays[0],
                                reference_losses=arrays[1])
            print(f"MEASURED {name} FTSS={m['min']:.5f} preference={m['training_preference']:.3f} "
                  f"gap={m['loss_gap']:.4f} coverage={m['coverage']:.3f}", flush=True)
            return m


def save_report(args, report):
    write_json(args.output / "results.json", report)
    summary = summarize(report["rows"])
    write_json(args.output / "summary.json", summary)
    lines = ["# Clean Fine-Tuning: FTSS and Generative Outcomes", "", f"State: {report['state']}", "",
             f"{len(args.bases)} saved matched-recipe clean base models; {len(args.seeds)} subset seeds per base. "
             f"Class-balanced, nested {args.sizes}-image tuning subsets use MNIST test data excluded from base training. "
             "Equal-size untouched reference pools control nearest-neighbor set size. A fixed disjoint 1000-image "
             "bank supplies PRDC and diversity references. All outcomes use EMA weights, paired generation noise "
             f"and FTSS probes within each run. Training uses the matched {args.steps}-step OneCycle recipe.", "",
             f"FTSS retains the upstream estimator definition: {args.probes} trajectories, 20 injection times, "
             f"{args.ode_steps} Euler steps and relative perturbation 0.1. Generated outcomes use {args.samples} samples; flow losses "
             "average eight noise/time draws per real example. Precision and coverage assess generation quality; "
             "training preference and loss gap assess training-set attraction and overfitting; diversity assesses "
             "concentration. These are distinct outcomes, not interchangeable definitions of memorization.", "",
             "| Images | Steps | Runs | FTSS / Before | Preference Change | Loss Gap Change | Coverage Change | Style Diversity Change |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for size, step in sorted({(r["size"], r["step"]) for r in summary["changes"]}):
        selected = [r for r in summary["changes"] if (r["size"], r["step"]) == (size, step)]
        fields = []
        for key in ("ftss_ratio", "training_preference_change", "loss_gap_change", "coverage_change", "style_diversity_ratio_change"):
            v = [r[key] for r in selected]
            fields.append(f"{np.mean(v):.4f} +/- {np.std(v):.4f}")
        lines.append(f"| {size} | {step} | {len(selected)} | " + " | ".join(fields) + " |")
    lines += ["", "Means +/- population standard deviations across completed fine-tuning realizations.", "",
              "## Associations", "", summary["interpretation"], "",
              "Contraction = -log(FTSS / baseline); positive values mean lower FTSS. "
              "Changes subtract the same run's step-zero outcome. Spearman associations are also stratified "
              "by checkpoint step and pretrained base in summary.json. No independence-based p-values are used.", ""]
    for key, value in summary["associations"].items():
        lines.append(f"- {key}: Spearman {value['pooled_spearman']}")
    (args.output / "report.md").write_text("\n".join(lines) + "\n")


def plot_results(output, rows):
    from matplotlib import pyplot as plt
    changes = summarize(rows)["changes"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    colors = {50: "#267b9d", 200: "#c34e40", 1000: "#38825c"}
    metrics = [("training_preference_change", "Training preference change"),
               ("loss_gap_change", "Flow-loss gap change"),
               ("style_diversity_ratio_change", "Style diversity change")]
    for ax, (key, label) in zip(axes, metrics):
        for size in sorted({r["size"] for r in changes}):
            selected = [r for r in changes if r["size"] == size]
            ax.scatter([r["contraction"] for r in selected], [r[key] for r in selected],
                       color=colors.get(size, "gray"), alpha=.75, label=f"{size} images")
        ax.axhline(0, color="gray", lw=.7)
        ax.axvline(0, color="gray", lw=.7)
        ax.set_xlabel("Contraction: -log(FTSS / before)")
        ax.set_ylabel(label)
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "outcome_associations.png", dpi=180)
    fig.savefig(output / "outcome_associations.pdf")
    plt.close(fig)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    checkpoints = [args.source / "reproduction/checkpoints" / f"mlp_mnist_n12000_run{b}.pt" for b in args.bases]
    config = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "sources": {str(p): digest(p) for p in [Path(__file__), ROOT / "ftss_batched.py",
                          ROOT / "run_matched_ftss_mlp.py", ROOT / "evaluate_generative_generalization.py",
                          args.repo / "train_memorization_mlp.py", args.repo / "finetune_memorization_ftss.py",
                          args.classifier, *checkpoints]},
              "plan": "Saved clean bases x subset seeds x nested sizes; paired EMA evaluations",
              "prespecified_outcomes": ["training_preference", "loss_gap", "membership_auc", "coverage",
                                         "precision", "style_diversity_ratio", "effective_digit_modes"]}
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Resume requires an identical protocol and unchanged source files")
    write_json(config_path, config)
    path = args.output / "results.json"
    report = json.loads(path.read_text()) if path.exists() else {"state": "running", "rows": []}
    evaluator = Evaluator(args, device)
    namespace = original_definitions(args.repo, device, None)
    try:
        report["state"] = "running"
        for base_id, checkpoint in zip(args.bases, checkpoints):
            for seed in args.seeds:
                for size in args.sizes:
                    prefix = f"base{base_id}_seed{seed}_n{size}"
                    rows = [r for r in report["rows"] if (r["base"], r["seed"], r["size"]) == (base_id, seed, size)]
                    if {r["step"] for r in rows}.issuperset([0, *args.check_steps]):
                        continue
                    report["current"] = dict(base=base_id, seed=seed, size=size)
                    save_report(args, report)
                    split = make_split(evaluator.labels, size, 41000 + seed)
                    write_json(args.output / f"{prefix}_indices.json", split)
                    model = namespace["MLPFlow"]().to(device)
                    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
                    ema = copy.deepcopy(model).eval().requires_grad_(False)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4,
                                                                   total_steps=args.steps, pct_start=.1)
                    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
                    torch.manual_seed(51000 + seed + base_id*1000)
                    pool = evaluator.images[split["tuned"]].to(device)
                    latest = args.output / f"{prefix}_latest.pt"
                    step = 0
                    if latest.exists():
                        saved = torch.load(latest, map_location="cpu", weights_only=True)
                        for name, obj in (("model", model), ("ema", ema), ("optimizer", optimizer),
                                          ("scheduler", scheduler), ("scaler", scaler)):
                            obj.load_state_dict(saved[name])
                        torch.set_rng_state(saved["cpu_rng"])
                        if device.type == "cuda":
                            torch.cuda.set_rng_state(saved["cuda_rng"], device)
                        step = saved["step"]

                    def measure(current):
                        if any(r["step"] == current for r in rows):
                            return
                        m = evaluator(ema, split, 61000 + base_id*1000 + seed, f"{prefix}_step{current}")
                        row = dict(base=base_id, seed=seed, size=size, step=current, metrics=m)
                        rows.append(row)
                        report["rows"].append(row)
                        save_report(args, report)

                    measure(step)
                    for current in range(step+1, args.steps+1):
                        model.train()
                        index = torch.randperm(len(pool), device=device)[:min(64, len(pool))]
                        target = pool[index]
                        noise = torch.randn_like(target)
                        t = torch.rand(len(target), 1, device=device)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device.type, enabled=device.type == "cuda"):
                            loss = F.mse_loss(model((1-t)*noise+t*target, t), target-noise)
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Nonfinite loss {prefix} step {current}")
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        namespace["update_ema"](ema, model, .999)
                        if current % 500 == 0:
                            print(f"TRAIN {prefix} {current}/{args.steps} loss={float(loss):.5f}", flush=True)
                        if current in args.check_steps:
                            saved = {name: obj.state_dict() for name, obj in (("model", model), ("ema", ema),
                                     ("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler))}
                            saved.update(step=current, cpu_rng=torch.get_rng_state())
                            if device.type == "cuda":
                                saved["cuda_rng"] = torch.cuda.get_rng_state(device)
                            temp = latest.with_suffix(".tmp")
                            torch.save(saved, temp)
                            temp.replace(latest)
                            measure(current)
                    del model, ema, optimizer, pool
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
        plot_results(args.output, report["rows"])
        report["state"] = "complete"
        report.pop("current", None)
    except Exception as exc:
        report.update(state="failed", error=str(exc))
        raise
    finally:
        save_report(args, report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", type=Path, default=ROOT / "table_runs/ftss_mlp_matched")
    p.add_argument("--output", type=Path, default=ROOT / "table_runs/ftss_outcome_study")
    p.add_argument("--repo", type=Path, default=ROOT.parent / "ftss-memorization")
    p.add_argument("--classifier", type=Path, default=ROOT / "table_runs/generative_generalization/feature_evaluator.pt")
    p.add_argument("--sizes", nargs="+", type=int, default=[50, 200, 1000])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--bases", nargs="+", type=int, default=[0, 1])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--check-steps", nargs="+", type=int, default=[1000, 3000])
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--probes", type=int, default=15)
    p.add_argument("--ode-steps", type=int, default=200)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if (args.steps < 20 or min(args.samples, args.probes, args.ode_steps) < 4
            or any(n < 10 or n > 1000 or n % 10 for n in args.sizes)
            or any(s < 0 or s > args.steps for s in args.check_steps)):
        p.error("Invalid sizes, step counts or evaluation sample counts")
    args.check_steps = sorted(set(args.check_steps) | {args.steps})
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "study.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
