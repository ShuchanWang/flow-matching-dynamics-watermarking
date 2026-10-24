"""Analyze contraction associations within unwatermarked fine-tuning trajectories."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from run_ftss_finetuning import digest, write_json


METRICS = ["raw_copy_rate_1pct", "visible_copy_rate_1pct", "visible_copy_rate_5pct",
           "raw_copy_coverage_1pct", "visible_copy_coverage_1pct",
           "visible_replica_rmse_005", "visible_replica_rmse_010",
           "visible_train_preference", "visible_neighbor_advantage", "membership_auc", "loss_gap"]


def fixed_error_copy_rates(train_distances, reference_distances, dimension):
    # Distance coordinates are [-1,1]; the reported pixel RMSE uses [0,1].
    rmse = np.asarray(train_distances) / (2 * np.sqrt(dimension))
    preference = np.asarray(train_distances) < np.asarray(reference_distances)
    return {f"visible_replica_rmse_{suffix}": float(((rmse <= tolerance) & preference).mean())
            for suffix, tolerance in (("005", 0.05), ("010", 0.10))}


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def adjusted_correlation(rows, metric, rank=False, predictor="contraction"):
    """Remove trajectory intercepts and common checkpoint-step effects."""
    if not rows:
        return None
    x = np.array([r[predictor] for r in rows])
    y = np.array([r[metric + "_change"] for r in rows])
    if rank:
        x, y = rankdata(x), rankdata(y)
    arms = [(r["run"], r["size"]) for r in rows]
    arm_keys = sorted(set(arms))
    steps = sorted({r["step"] for r in rows})
    columns = [np.ones(len(rows))]
    columns += [np.array([a == key for a in arms], dtype=float) for key in arm_keys[1:]]
    columns += [np.array([r["step"] == step for r in rows], dtype=float) for step in steps[1:]]
    design = np.stack(columns, axis=1)
    x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return correlation(x, y)


def associations(records, bootstrap=500, predictor="contraction"):
    result = {}
    for model in sorted({r["model"] for r in records}):
        rows = [r for r in records if r["model"] == model and r["step"] > 0]
        seeds = sorted({r["run"] for r in rows})
        model_result = {}
        for metric in METRICS:
            x = [r[predictor] for r in rows]
            y = [r[metric + "_change"] for r in rows]
            entry = {"post_checkpoint_observations": len(rows),
                     "independent_seed_blocks": len(seeds),
                     "trajectories": len({(r["run"], r["size"]) for r in rows}),
                     "pearson_changes": correlation(x, y),
                     "spearman_changes": correlation(rankdata(x), rankdata(y)),
                     "arm_step_adjusted_pearson": adjusted_correlation(rows, metric, predictor=predictor),
                     "arm_step_adjusted_rank_correlation": adjusted_correlation(rows, metric, True, predictor),
                     "per_seed_spearman": {}, "per_step_spearman": {}}
            for seed in seeds:
                selected = [r for r in rows if r["run"] == seed]
                entry["per_seed_spearman"][str(seed)] = correlation(
                    rankdata([r[predictor] for r in selected]),
                    rankdata([r[metric + "_change"] for r in selected]))
            for step in sorted({r["step"] for r in rows}):
                selected = [r for r in rows if r["step"] == step]
                entry["per_step_spearman"][str(step)] = correlation(
                    rankdata([r[predictor] for r in selected]),
                    rankdata([r[metric + "_change"] for r in selected]))
            draws = []
            if len(seeds) >= 3:
                rng = np.random.default_rng(20261019)
                for _ in range(bootstrap):
                    sample = []
                    for block, seed in enumerate(rng.choice(seeds, len(seeds), replace=True)):
                        sample.extend(dict(r, run=block) for r in rows if r["run"] == seed)
                    value = adjusted_correlation(sample, metric, predictor=predictor)
                    if value is not None:
                        draws.append(value)
            entry["seed_bootstrap_adjusted_pearson_95_interval"] = (
                np.quantile(draws, [0.025, 0.975]).tolist() if draws else None)
            model_result[metric] = entry
        result[model] = model_result
    return result


def analyze(directory):
    report = json.loads((directory / "results.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    if report["state"] != "complete":
        raise ValueError("Wait for the study to complete before computing final associations")
    records = report["measurements"]
    keys = [(r["model"], r["size"], r["run"], r["step"]) for r in records]
    expected = {(model, size, run, step) for model in config["models"] for size in config["sizes"]
                for run in range(config["runs"]) for step in config["steps"]}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Missing, duplicate, or unexpected measurements")
    for row in records:
        name = f"{row['model']}_n{row['size']}_run{row['run']}_step{row['step']}_audit.npz"
        with np.load(directory / name) as audit:
            train_d = audit["visible_train_distances"]
            reference_d = audit["visible_reference_distances"]
            row.update(fixed_error_copy_rates(train_d, reference_d,
                                             audit["generated_samples"].shape[1]))
    baselines = {(r["model"], r["run"], r["size"]): r for r in records if r["step"] == 0}
    for row in records:
        base = baselines[(row["model"], row["run"], row["size"])]
        for metric in METRICS:
            row[metric + "_change"] = row[metric] - base[metric]
    grouped = []
    for model in config["models"]:
        for size in config["sizes"]:
            for step in config["steps"]:
                selected = [r for r in records if
                            (r["model"], r["size"], r["step"]) == (model, size, step)]
                if len(selected) != config["runs"]:
                    raise ValueError("Incomplete checkpoint/seed coverage")
                row = {"model": model, "size": size, "step": step, "seeds": len(selected)}
                for metric in METRICS + ["g_min", "ftss_ratio", "contraction", "profile_contraction"]:
                    values = [r[metric] for r in selected]
                    row[metric + "_mean"] = float(np.mean(values))
                    row[metric + "_std"] = float(np.std(values))
                grouped.append(row)
    result = {"config": config, "conditions": grouped,
              "checkpoint_integrity": {model: {
                  "expected_sha256": checkpoint["sha256"],
                  "current_sha256": digest(Path(checkpoint["path"])) if Path(checkpoint["path"]).exists() else None}
                  for model, checkpoint in config["checkpoints"].items()},
              "source_sha256": {"analysis": digest(Path(__file__)),
                                "runner": digest(Path(__file__).with_name("run_unmarked_memorization.py"))},
              "associations": associations(records),
              "secondary_profile_associations": associations(records, predictor="profile_contraction"),
              "analysis_protocol": {
                  "predictor": "-log(g_min/current pre-fine-tuning g_min); positive means increased contraction",
                  "secondary_predictor": "Negative mean log FTSS curve ratio, averaged over all measured times",
                  "outcomes": "Changes relative to the same trajectory at step zero",
                  "primary_replication": "Upstream raw-pixel rule at 1% calibration quantile",
                  "strict_replication": "Visible-image RMSE <= 0.05 or 0.10 in [0,1], nearer to tuned than reference",
                  "adjustment": "Trajectory intercepts and common checkpoint-step effects, fitted within architecture",
                  "resampling": "Seed blocks; repeated checkpoint measurements stay together",
                  "interpretation": "Exploratory seed-block associations without causal claims or pooled-observation p-values"}}
    write_json(directory / "analysis.json", result)
    with (directory / "conditions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(grouped[0]))
        writer.writeheader()
        writer.writerows(grouped)
    plot(directory, grouped, config)
    plot_associations(directory, records, config)
    write_report(directory, result)
    plot_neighbors(directory, records, config)
    return result


def write_report(directory, result):
    config = result["config"]
    lines = ["# Unwatermarked Fine-Tuning: Contraction and Memorization", "",
             "Ordinary full-parameter fine-tuning of existing clean MNIST MLP and UNet checkpoints. "
             "The tuned, untouched reference, and calibration pools are disjoint and class-balanced, "
             "drawn from the MNIST test split excluded from original model training. "
             "There is no watermark loss or watermark evaluation.", "",
             f"{config['runs']} seeds; subsets of {config['sizes']} images; "
             f"checkpoints at {config['steps']} steps; learning rate {config['lr']}. "
             f"Each checkpoint uses {config['n_generated']} generated samples and "
             f"{config['n_probes']} FTSS probes.", "",
             "## Operational Definitions", "",
             "- FTSS ratio: minimum terminal finite-difference RMS gain over probe injection times, "
             "divided by its pre-fine-tuning value. "
             "A ratio below one indicates increased contraction. Association predictor: minus log of this ratio.",
             "- Calibrated replication: generated sample is closer to the tuned than the equal-size reference "
             "pool and its tuned distance is below the 1% quantile of calibration-to-tuned distances. "
             "The upstream FTSS rule uses raw endpoints; visible-image replication clips endpoints to the display range.",
             "- Fixed-error replication: visible-image nearest-tuned pixel RMSE in [0,1] is at most 0.05 "
             "(or 0.10), and the sample is closer to tuned than reference images. These are supplementary "
             "pixel-tolerance definitions; inspect the neighbor figure to judge visual similarity.",
             "- Training attraction: fraction of samples nearer to tuned than reference images, plus "
             "normalized nearest-neighbor distance advantage. These measure preference, not exact replication.",
             "- Replication coverage: fraction of distinct tuned examples matched by at least one "
             "generated sample under the calibrated replication rule. This distinguishes copying several "
             "training examples from repeated output of one example.",
             "- Loss-based membership AUROC: negative flow-matching loss distinguishes tuned from reference "
             "images; 0.5 is chance. Loss gap: reference mean loss minus tuned mean loss. "
             "Both measure exposure or overfitting rather than image duplication.", "",
             "## Final Checkpoint", "",
             "Mean +/- seed standard deviation. Replication and preference are percentages.", "",
             "| Model | Tuned images | FTSS ratio | Raw calibrated replicas | Visible calibrated replicas | "
             "Visible RMSE <= 0.05 | Raw copy coverage | Train preference | Membership AUROC | Loss gap |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def cell(row, metric, scale=1):
        return f"{scale * row[metric + '_mean']:.3f} +/- {scale * row[metric + '_std']:.3f}"
    for row in result["conditions"]:
        if row["step"] != max(config["steps"]):
            continue
        metrics = [("ftss_ratio", 1), ("raw_copy_rate_1pct", 100),
                   ("visible_copy_rate_1pct", 100), ("visible_replica_rmse_005", 100),
                   ("raw_copy_coverage_1pct", 100),
                   ("visible_train_preference", 100), ("membership_auc", 1), ("loss_gap", 1)]
        lines.append(f"| {row['model']} | {row['size']} | " +
                     " | ".join(cell(row, metric, scale) for metric, scale in metrics) + " |")
    lines += ["", "## Association With Increased Contraction", "",
              "Outcomes are changes from the same trajectory's initial measurement. Positive correlations "
              "mean greater contraction accompanies greater replication or exposure. Adjusted correlations "
              "remove trajectory intercepts and shared checkpoint-step effects, separately within architecture. "
              "Repeated checkpoints stay together in seed-block bootstrap resampling. "
              f"There are {config['runs']} seed blocks; intervals are exploratory, and the checkpoints "
              "are not independent experimental repeats.", "",
              "| Model | Outcome change | Spearman | Adjusted Pearson | Seed-block 95% interval |",
              "|---|---|---:|---:|---|"]
    def value(number):
        return "undefined (constant or insufficient data)" if number is None else f"{number:.3f}"
    for model, metrics in result["associations"].items():
        for name, entry in metrics.items():
            interval = entry["seed_bootstrap_adjusted_pearson_95_interval"]
            text = "not estimated" if interval is None else f"[{interval[0]:.3f}, {interval[1]:.3f}]"
            lines.append(f"| {model} | {name} | {value(entry['spearman_changes'])} | "
                         f"{value(entry['arm_step_adjusted_pearson'])} | {text} |")
    lines += ["", "All checkpoint means are in `conditions.csv`; individual measurements and saved audits "
              "are in `results.json` and `*_audit.npz`. `analysis.json` contains per-seed and per-step "
              "associations. `trajectories.png` shows the full trajectories; `associations.png` plots "
              "baseline-relative outcomes against contraction. `neighbors.png` shows selected "
              "generated samples and their nearest tuned/reference images at the final checkpoint: "
              "selection is by minimum, median, and maximum tuned-image distance in the first seed's "
              f"{min(config['sizes'])}-image condition. `analysis.json` also reports a secondary sensitivity "
              "analysis using the full FTSS curve instead of its minimum; the primary definition remains "
              "the upstream minimum-gain statistic.", ""]
    (directory / "report.md").write_text("\n".join(lines))


def plot_associations(directory, rows, config):
    from matplotlib import pyplot as plt

    settings = [("raw_copy_rate_1pct_change", "Change in calibrated replication (pp)", 100),
                ("visible_replica_rmse_005_change", "Change in RMSE <= 0.05 replication (pp)", 100),
                ("membership_auc_change", "Change in membership AUROC", 1)]
    colors = ["#176f94", "#bd493c", "#34805d", "#96732f"]
    fig, axes = plt.subplots(len(settings), len(config["models"]),
                             figsize=(5 * len(config["models"]), 10), squeeze=False)
    for col, model in enumerate(config["models"]):
        for row_index, (metric, label, scale) in enumerate(settings):
            ax = axes[row_index, col]
            for size, color in zip(config["sizes"], colors):
                for run in range(config["runs"]):
                    selected = sorted([r for r in rows if r["model"] == model and
                                       r["size"] == size and r["run"] == run and r["step"] > 0],
                                      key=lambda r: r["step"])
                    ax.plot([r["contraction"] for r in selected],
                            [scale * r[metric] for r in selected], color=color, alpha=0.65,
                            marker="o", linewidth=0.8, markersize=4,
                            label=f"{size} images" if run == 0 else None)
            ax.axvline(0, color="gray", linewidth=0.6)
            ax.axhline(0, color="gray", linewidth=0.6)
            ax.set_xlabel("Increased contraction (-log FTSS ratio)")
            ax.set_ylabel(label)
            ax.grid(alpha=0.15)
            if row_index == 0:
                ax.set_title(model.upper())
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "associations.png", dpi=160)
    fig.savefig(directory / "associations.pdf")
    plt.close(fig)


def plot_neighbors(directory, rows, config):
    from matplotlib import pyplot as plt
    from torchvision.datasets import MNIST

    dataset = MNIST("data", train=False, download=False)
    images = dataset.data.numpy().reshape(-1, 784) / 127.5 - 1
    fig, axes = plt.subplots(3 * len(config["models"]), 3,
                             figsize=(8, 7 * len(config["models"])), squeeze=False)
    for model_index, model in enumerate(config["models"]):
        row = next(r for r in rows if r["model"] == model and r["run"] == 0
                   and r["size"] == min(config["sizes"]) and r["step"] == max(config["steps"]))
        prefix = f"{model}_n{row['size']}_run0"
        split = json.loads((directory / f"{prefix}_indices.json").read_text())
        with np.load(directory / f"{prefix}_step{row['step']}_audit.npz") as audit:
            samples = audit["generated_samples"].clip(-1, 1)
            order = np.argsort(audit["visible_train_distances"])
            selected = order[[0, len(order) // 2, -1]]
            for position, index in enumerate(selected):
                tuned = images[split["tuned"][audit["visible_train_nearest_indices"][index]]]
                reference = images[split["reference"]]
                nearest_reference = reference[np.argmin(((reference - samples[index]) ** 2).sum(1))]
                rmse = float(np.sqrt(((samples[index] - tuned) ** 2).mean()) / 2)
                titles = [f"{model.upper()} sample {int(index)}", f"Nearest tuned; RMSE {rmse:.3f}",
                          "Nearest reference"]
                for col, pixels in enumerate((samples[index], tuned, nearest_reference)):
                    ax = axes[3 * model_index + position, col]
                    ax.imshow(pixels.reshape(28, 28), cmap="gray", vmin=-1, vmax=1)
                    ax.set_title(titles[col], fontsize=10)
                    ax.axis("off")
    fig.tight_layout()
    fig.savefig(directory / "neighbors.png", dpi=160)
    fig.savefig(directory / "neighbors.pdf")
    plt.close(fig)


def plot(directory, rows, config):
    os.environ.setdefault("MPLCONFIGDIR", str((directory / ".matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    settings = [("contraction", "Increased contraction (-log ratio)", 1),
                ("raw_copy_rate_1pct", "Calibrated replicas (%)", 100),
                ("visible_replica_rmse_005", "Visible RMSE <= 0.05 (%)", 100),
                ("membership_auc", "Loss-based membership AUROC", 1),
                ("loss_gap", "Reference - training flow loss", 1)]
    colors = ["#176f94", "#bd493c", "#34805d", "#96732f"]
    fig, axes = plt.subplots(len(settings), len(config["models"]),
                             figsize=(5 * len(config["models"]), 12), squeeze=False)
    for col, model in enumerate(config["models"]):
        for i, (metric, label, scale) in enumerate(settings):
            ax = axes[i, col]
            for size, color in zip(config["sizes"], colors):
                selected = sorted([r for r in rows if r["model"] == model and r["size"] == size],
                                  key=lambda r: r["step"])
                ax.errorbar([r["step"] for r in selected],
                            [scale * r[metric + "_mean"] for r in selected],
                            yerr=[scale * r[metric + "_std"] for r in selected],
                            label=f"{size} images", color=color, marker="o", capsize=2)
            ax.set_ylabel(label)
            ax.set_xlabel("Fine-tuning steps")
            ax.grid(alpha=0.2)
            if scale == 100:
                ax.set_ylim(0, 100)
            elif metric == "membership_auc":
                ax.set_ylim(0, 1)
            if i == 0:
                ax.set_title(model.upper())
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "trajectories.png", dpi=160)
    fig.savefig(directory / "trajectories.pdf")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("directory", type=Path)
    output = analyze(p.parse_args().directory)
    for model, metrics in output["associations"].items():
        for name, values in metrics.items():
            print(model, name, "Spearman changes:", values["spearman_changes"],
                  "adjusted Pearson:", values["arm_step_adjusted_pearson"],
                  "seed-block interval:", values["seed_bootstrap_adjusted_pearson_95_interval"])
