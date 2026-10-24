"""Independent feature-space quality, coverage, diversity, and training preference."""

import argparse
import contextlib
import copy
import fcntl
import importlib.metadata
import io
import json
import os
from pathlib import Path
import time

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST

from analyze_memorization_collapse import association
from run_ftss_finetuning import digest, write_json


class DigitEvaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.embedding = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU())
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        hidden = self.conv((x - 0.1307) / 0.3081)
        embedding = self.embedding(hidden)
        spatial = F.adaptive_avg_pool2d(hidden, (4, 4)).flatten(1)
        return self.classifier(embedding), F.normalize(spatial, dim=1), F.normalize(embedding, dim=1)


@torch.no_grad()
def accuracy(model, x, y, batch=256):
    model.eval()
    correct = 0
    for start in range(0, len(x), batch):
        logits = model(x[start:start + batch])[0]
        correct += int((logits.argmax(1) == y[start:start + batch]).sum())
    return correct / len(x)


def evaluator(output, device, epochs):
    path = output / "feature_evaluator.pt"
    protocol = {"seed": 20261020, "epochs": epochs, "fit_examples": 50000,
                "validation_examples": 10000, "architecture": "DigitEvaluator_conv32_conv64_fc128_v1",
                "selection": "Best validation accuracy; MNIST test excluded from fitting and selection"}
    model = DigitEvaluator().to(device)
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["protocol"] != protocol:
            raise ValueError("Existing evaluator uses a different training protocol")
        model.load_state_dict(saved["model"])
        return model.eval().requires_grad_(False), saved["metadata"]
    torch.manual_seed(protocol["seed"])
    model = DigitEvaluator().to(device)
    dataset = MNIST("data", train=True, download=False)
    x = dataset.data[:, None].float().to(device) / 255
    y = dataset.targets.to(device)
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(protocol["seed"]))
    fit, validation = order[:50000].to(device), order[50000:].to(device)
    generator = torch.Generator(device=device).manual_seed(protocol["seed"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_accuracy, best_state, history = -1., None, []
    for epoch in range(epochs):
        model.train()
        permutation = fit[torch.randperm(len(fit), device=device, generator=generator)]
        for index in permutation.split(256):
            loss = F.cross_entropy(model(x[index])[0], y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        score = accuracy(model, x[validation], y[validation])
        history.append({"epoch": epoch + 1, "validation_accuracy": score})
        if score > best_accuracy:
            best_accuracy = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"Classifier epoch {epoch + 1}/{epochs}: validation accuracy {score:.4f}", flush=True)
    metadata = {"history": history, "best_validation_accuracy": best_accuracy,
                "split_sha256": __import__("hashlib").sha256(order.numpy().tobytes()).hexdigest(),
                "training_source_sha256": digest(Path(__file__))}
    torch.save({"model": best_state, "protocol": protocol, "metadata": metadata}, path)
    model.load_state_dict(best_state)
    return model.eval().requires_grad_(False), metadata


@torch.no_grad()
def features(model, images, device):
    spatial, embedding, probabilities = [], [], []
    for batch in images.split(256):
        logits, s, e = model(batch.to(device))
        spatial.append(s.cpu().numpy())
        embedding.append(e.cpu().numpy())
        probabilities.append(logits.softmax(1).cpu().numpy())
    return {"spatial": np.concatenate(spatial), "embedding": np.concatenate(embedding),
            "probabilities": np.concatenate(probabilities)}


def balanced_part(indices, start, count):
    if count % 10 or len(indices) % 10:
        raise ValueError("Balanced pool sizes must be multiples of ten")
    groups = np.asarray(indices).reshape(10, -1)
    if start + count // 10 > groups.shape[1]:
        raise ValueError("Not enough balanced held-out examples")
    return groups[:, start:start + count // 10].reshape(-1)


def semantic_diversity(generated, predictions, real, labels):
    ratios, counts = [], np.bincount(predictions, minlength=10)
    for digit in range(10):
        group = generated[predictions == digit]
        reference = real[labels == digit]
        if len(group) < 2 or len(reference) < 2:
            continue
        variance = np.square(group - group.mean(0)).sum() / (len(group) - 1)
        reference_variance = np.square(reference - reference.mean(0)).sum() / (len(reference) - 1)
        ratios.append(float(variance / max(reference_variance, 1e-24)))
    p = counts[counts > 0] / len(predictions)
    return {"style_diversity_ratio": float(np.mean(ratios)) if ratios else None,
            "digits_with_diversity_estimate": len(ratios), "digit_counts": counts.tolist(),
            "effective_digit_modes": float(np.exp(-np.sum(p * np.log(p)))),
            "represented_digits": int((counts > 0).sum())}


def training_preference(generated, tuned, reference):
    train_d = euclidean_distances(generated, tuned, squared=True).min(1)
    ref_d = euclidean_distances(generated, reference, squared=True).min(1)
    return {"training_preference": float((train_d < ref_d).mean()),
            "training_distance_squared": float(train_d.mean()),
            "reference_distance_squared": float(ref_d.mean()),
            "training_anchor_advantage": float(np.log((ref_d.mean() + 1e-24) / (train_d.mean() + 1e-24)))}


def prdc(real, fake, k):
    from prdc import compute_prdc
    with contextlib.redirect_stdout(io.StringIO()):
        values = compute_prdc(real_features=real, fake_features=fake, nearest_k=k)
    return {key: float(value) for key, value in values.items()}


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    source = json.loads((args.study / "results.json").read_text())
    if source["state"] != "complete":
        raise ValueError("The source study must be complete")
    model, metadata = evaluator(args.output, device, args.epochs)
    dataset = MNIST("data", train=False, download=False)
    x = dataset.data[:, None].float() / 255
    labels = dataset.targets.numpy()
    real_bank = features(model, x, device)
    metadata["mnist_test_accuracy"] = float((real_bank["probabilities"].argmax(1) == labels).mean())
    print("Independent classifier test accuracy:", metadata["mnist_test_accuracy"], flush=True)
    np.savez_compressed(args.output / "real_features.npz", **real_bank, labels=labels)
    protocol = {"source_results_sha256": digest(args.study / "results.json"),
                "classifier_sha256": digest(args.output / "feature_evaluator.pt"),
                "evaluation_source_sha256": digest(Path(__file__)), "classifier": metadata,
                "primary_features": "L2-normalized 4x4 pooled convolutional features (1024 dimensions)",
                "secondary_features": "L2-normalized penultimate classifier features (128 dimensions)",
                "prdc_package": importlib.metadata.version("prdc"), "nearest_k": args.k,
                "prdc_sample_count": args.samples,
                "heldout_reference": "Balanced calibration examples, disjoint from fine-tuning and equal-size neighbor-reference pools",
                "generation": "Existing saved outputs clipped to display range; no new generative training",
                "style_diversity": "Macro within-predicted-digit feature variance / matching real within-digit variance; missing classes reported",
                "status": "Exploratory external-outcome test; features and metric protocol fixed before computing outcomes"}
    write_json(args.output / "protocol.json", protocol)
    report = {"state": "running", "protocol": protocol, "measurements": [], "controls": {}}
    begin = time.monotonic()
    for original in source["measurements"]:
        row = {key: original[key] for key in ("model", "run", "size", "step", "contraction", "ftss_ratio")}
        prefix = "{model}_n{size}_run{run}".format(**row)
        split = json.loads((args.study / f"{prefix}_indices.json").read_text())
        reference_indices = balanced_part(split["calibration"], 0, args.samples)
        control_indices = balanced_part(split["calibration"], args.samples // 10, args.samples)
        with np.load(args.study / f"{prefix}_step{row['step']}_audit.npz") as audit:
            samples = audit["generated_samples"]
            if len(samples) < args.samples:
                raise ValueError("Requested more samples than the stored audit")
            images = torch.from_numpy(samples.clip(-1, 1)).reshape(-1, 1, 28, 28).add(1).div(2)
        generated = features(model, images, device)
        predictions = generated["probabilities"].argmax(1)
        row["mean_classifier_confidence"] = float(generated["probabilities"].max(1).mean())
        control_key = f"n{row['size']}_run{row['run']}"
        for space in ("spatial", "embedding"):
            real = real_bank[space][reference_indices]
            fake = generated[space]
            values = prdc(real, fake[:args.samples], args.k)
            values.update(semantic_diversity(fake, predictions, real_bank[space][split["calibration"]],
                                             labels[split["calibration"]]))
            values.update(training_preference(fake, real_bank[space][split["tuned"]],
                                               real_bank[space][split["reference"]]))
            row.update({f"{space}_{key}": value for key, value in values.items()})
            if control_key not in report["controls"]:
                report["controls"][control_key] = {}
            if space not in report["controls"][control_key]:
                report["controls"][control_key][space] = prdc(real, real_bank[space][control_indices], args.k)
        np.savez_compressed(args.output / f"{prefix}_step{row['step']}_features.npz", **generated)
        report["measurements"].append(row)
        report["elapsed_seconds"] = time.monotonic() - begin
        write_json(args.output / "results.json", report)
        print(f"{prefix} step={row['step']}: precision={row['spatial_precision']:.3f} "
              f"coverage={row['spatial_coverage']:.3f} recall={row['spatial_recall']:.3f} "
              f"style={row['spatial_style_diversity_ratio']:.3f} "
              f"train-pref={row['spatial_training_preference']:.3f}", flush=True)
    report["state"] = "complete"
    report["elapsed_seconds"] = time.monotonic() - begin
    write_json(args.output / "results.json", report)
    summarize(args.output, report)


def summarize(directory, report):
    rows = copy.deepcopy(report["measurements"])
    baselines = {(r["model"], r["size"], r["run"]): r for r in rows if r["step"] == 0}
    outcomes = []
    for space in ("spatial", "embedding"):
        for metric, sign in (("coverage", -1), ("recall", -1), ("precision", 1),
                             ("style_diversity_ratio", -1), ("effective_digit_modes", -1),
                             ("training_preference", 1), ("training_anchor_advantage", 1)):
            key, outcome = f"{space}_{metric}", f"{space}_{metric}_oriented_change"
            outcomes.append(outcome)
            for row in rows:
                before = baselines[row["model"], row["size"], row["run"]][key]
                after = row[key]
                row[outcome] = sign * (after - before) if before is not None and after is not None else None
    associations = {model: {key: association([r for r in rows if r["model"] == model], key)
                            for key in outcomes} for model in sorted({r["model"] for r in rows})}
    analysis = {"associations": associations, "measurements": rows, "protocol": report["protocol"],
                "interpretation": "Positive correlations mean contraction tracks coverage/diversity loss, increased precision, or increased training preference; these outcomes are not interchangeable"}
    write_json(directory / "analysis.json", analysis)
    write_report(directory, analysis, report["controls"])
    plot(directory, rows)


def write_report(directory, analysis, controls):
    rows = analysis["measurements"]
    lines = ["# Generative Generalization in Independent Features", "",
             "An independent MNIST CNN was fitted on 50,000 original training images and selected on "
             "10,000 other original training images. Fine-tuning/evaluation images are in MNIST test; "
             "none were used to train or select this evaluator. Its test accuracy is "
             f"{100 * analysis['protocol']['classifier']['mnist_test_accuracy']:.2f}%.", "",
             "Primary features are normalized spatial convolutional features; penultimate features provide "
             "a secondary sensitivity check. PRDC uses the authors' package, k=3, and 200 real/200 "
             "generated examples in the default protocol. Held-out examples are disjoint from fine-tuning. "
             "Within-digit style diversity is feature variance relative to real digits; digit counts and "
             "missing-class information are saved. Precision measures feature-manifold fidelity, not human "
             "quality judgments. Classifier confidence alone is not treated as quality.", "",
             "## Main Measurements", "",
             "Mean over three seeds. Each cell shows initial -> final. Coverage, recall, precision, and "
             "training preference are percentages; style diversity is a ratio to real-data diversity.", "",
             "| Model | Tuned images | Precision | Held-out coverage | Held-out recall | Within-digit diversity | Effective digits | Training preference |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in sorted({r["model"] for r in rows}):
        for size in sorted({r["size"] for r in rows}):
            group = [r for r in rows if r["model"] == model and r["size"] == size]
            cells = []
            for key, scale in (("precision", 100), ("coverage", 100), ("recall", 100),
                               ("style_diversity_ratio", 1), ("effective_digit_modes", 1),
                               ("training_preference", 100)):
                values = [np.mean([r[f"spatial_{key}"] for r in group if r["step"] == step])
                          for step in (0, max(r["step"] for r in group))]
                cells.append(f"{scale * values[0]:.3f} -> {scale * values[1]:.3f}")
            lines.append(f"| {model} | {size} | " + " | ".join(cells) + " |")
    lines += ["", "## Association With FTSS Contraction", "",
              "Baseline-relative changes are analyzed separately by architecture. Coverage, recall, style, "
              "and category outcomes are oriented as losses; precision and training attraction as gains. "
              "Adjusted correlations remove trajectory intercepts and common step effects. Three seed "
              "blocks support exploratory associations, not independent repetitions of every checkpoint.", "",
              "| Model | External outcome | Spearman | Adjusted Pearson |", "|---|---|---:|---:|"]
    for model, metrics in analysis["associations"].items():
        for key, values in metrics.items():
            cells = ["undefined" if values[k] is None else f"{values[k]:.3f}"
                     for k in ("spearman", "adjusted_pearson")]
            lines.append(f"| {model} | {key} | " + " | ".join(cells) + " |")
    lines += ["", "## Real-Versus-Real Controls", "",
              "Two disjoint balanced held-out pools with the same sample count establish finite-sample "
              "metric levels. These controls are not used to train features or tune metric parameters.", ""]
    for space in ("spatial", "embedding"):
        values = {metric: np.mean([c[space][metric] for c in controls.values()])
                  for metric in ("precision", "recall", "density", "coverage")}
        lines.append(f"- {space}: " + ", ".join(f"{k}={v:.3f}" for k, v in values.items()))
    lines += ["", "[Metric definitions and authors' implementation](https://github.com/clovaai/generative-evaluation-prdc). "
              "All measurements, feature tensors, evaluator weights, provenance, and per-seed associations "
              "are saved alongside this report.", ""]
    (directory / "report.md").write_text("\n".join(lines))


def plot(directory, rows):
    os.environ.setdefault("MPLCONFIGDIR", str((directory / ".matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    settings = [("coverage", "Held-out coverage (%)", 100), ("precision", "Precision (%)", 100),
                ("style_diversity_ratio", "Within-digit feature diversity / real", 1),
                ("effective_digit_modes", "Effective digit categories", 1),
                ("training_preference", "Training-neighbor preference (%)", 100)]
    models = sorted({r["model"] for r in rows})
    fig, axes = plt.subplots(len(settings), len(models), figsize=(5 * len(models), 12), squeeze=False)
    for col, model in enumerate(models):
        for i, (key, label, scale) in enumerate(settings):
            ax = axes[i, col]
            for size, color in zip(sorted({r["size"] for r in rows}), ["#176f94", "#bd493c", "#34805d", "#96732f"]):
                x, mean, std = [], [], []
                for step in sorted({r["step"] for r in rows}):
                    selected = [r[f"spatial_{key}"] for r in rows if
                                (r["model"], r["size"], r["step"]) == (model, size, step)]
                    x.append(step)
                    mean.append(scale * np.mean(selected))
                    std.append(scale * np.std(selected))
                ax.errorbar(x, mean, yerr=std, color=color, marker="o", capsize=2, label=f"{size} images")
            ax.set_xlabel("Fine-tuning steps")
            ax.set_ylabel(label)
            ax.grid(alpha=0.15)
            if scale == 100:
                ax.set_ylim(0, 100)
            if i == 0:
                ax.set_title(model.upper())
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "external_outcomes.png", dpi=160)
    fig.savefig(directory / "external_outcomes.pdf")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--study", type=Path, default=Path("table_runs/unmarked_memorization_multimodel"))
    p.add_argument("--output", type=Path, default=Path("table_runs/generative_generalization"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args()
    if args.epochs < 1 or args.samples < 10 or args.samples % 10 or not 1 <= args.k < args.samples:
        p.error("Invalid evaluator or metric parameters")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "evaluation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / "results.json").exists():
            p.error("Use a fresh output directory to preserve the existing evaluation")
        run(args)
