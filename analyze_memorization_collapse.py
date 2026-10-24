"""Explore global and training-anchored collapse without changing saved FTSS results."""

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from analyze_unmarked_memorization import adjusted_correlation, correlation
from run_ftss_finetuning import digest, write_json


def dispersion(samples, assignments, pool_size):
    samples = np.asarray(samples, dtype=np.float64)
    assignments = np.asarray(assignments)
    if samples.ndim != 2 or len(samples) < 2 or assignments.shape != (len(samples),):
        raise ValueError("Expected at least two samples and one prototype assignment per sample")
    if np.any(assignments < 0) or np.any(assignments >= pool_size):
        raise ValueError("Assignment outside prototype pool")
    centered = samples - samples.mean(0)
    gram = centered @ centered.T / len(samples)
    trace = float(np.trace(gram))
    square_trace = float(np.square(gram).sum())
    participation_rank = trace * trace / square_trace if square_trace > 1e-24 else 0.
    counts = np.bincount(assignments, minlength=pool_size)
    nonempty = counts[counts > 0]
    probabilities = nonempty / len(samples)
    within_ss, pair_ss, ordered_pairs = 0., 0., 0
    for label in np.flatnonzero(counts):
        group = samples[assignments == label]
        ss = float(np.square(group - group.mean(0)).sum())
        within_ss += ss
        pair_ss += 2 * len(group) * ss
        ordered_pairs += len(group) * (len(group) - 1)
    return {
        "output_variance": trace,
        "participation_rank": participation_rank,
        "within_prototype_variance": within_ss / len(samples),
        "within_prototype_pair_mse": pair_ss / ordered_pairs if ordered_pairs else None,
        "within_prototype_ordered_pairs": ordered_pairs,
        "within_prototype_variance_fraction": within_ss / (len(samples) * trace) if trace > 1e-24 else None,
        "effective_occupied_prototypes": float(np.exp(-np.sum(probabilities * np.log(probabilities)))),
        "occupied_prototype_fraction": float(len(nonempty) / pool_size),
        "largest_prototype_mass": float(probabilities.max()),
    }


def association(rows, metric):
    selected = [r for r in rows if r["step"] > 0 and r[metric] is not None]
    x = [r["contraction"] for r in selected]
    y = [r[metric] for r in selected]
    # These outcomes are already baseline-relative collapse measures.
    adapted = [dict(r, **{metric + "_change": r[metric]}) for r in selected]
    return {"observations": len(selected),
            "spearman": correlation(rankdata(x), rankdata(y)),
            "adjusted_pearson": adjusted_correlation(adapted, metric),
            "per_seed_spearman": {
                str(seed): correlation(
                    rankdata([r["contraction"] for r in selected if r["run"] == seed]),
                    rankdata([r[metric] for r in selected if r["run"] == seed]))
                for seed in sorted({r["run"] for r in selected})}}


def analyze(directory):
    source = json.loads((directory / "results.json").read_text())
    if source["state"] != "complete":
        raise ValueError("The fine-tuning study must finish first")
    rows = copy.deepcopy(source["measurements"])
    for row in rows:
        name = "{model}_n{size}_run{run}_step{step}_audit.npz".format(**row)
        with np.load(directory / name) as audit:
            for space in ("raw", "visible"):
                samples = audit["generated_samples"]
                if space == "visible":
                    samples = samples.clip(-1, 1)
                values = dispersion(samples, audit[f"{space}_train_nearest_indices"], row["size"])
                values["training_anchor_mse"] = float(np.mean(audit[f"{space}_train_distances"] ** 2))
                values["reference_anchor_mse"] = float(np.mean(audit[f"{space}_reference_distances"] ** 2))
                values["training_reference_energy_ratio"] = (
                    values["training_anchor_mse"] / max(values["reference_anchor_mse"], 1e-24))
                row.update({f"{space}_{key}": value for key, value in values.items()})
    base = {(r["model"], r["size"], r["run"]): r for r in rows if r["step"] == 0}
    outcomes = []
    for space in ("raw", "visible"):
        for metric in ("output_variance", "participation_rank", "within_prototype_variance",
                       "within_prototype_pair_mse", "training_anchor_mse",
                       "training_reference_energy_ratio", "effective_occupied_prototypes"):
            key = f"{space}_{metric}"
            outcome = key + "_collapse"
            outcomes.append(outcome)
            for row in rows:
                before = base[row["model"], row["size"], row["run"]][key]
                after = row[key]
                row[outcome] = (float(-np.log(max(after, 1e-24) / before))
                                if before is not None and before > 1e-24 and after is not None else None)
    associations = {model: {outcome: association([r for r in rows if r["model"] == model], outcome)
                            for outcome in outcomes}
                    for model in sorted({r["model"] for r in rows})}
    result = {"state": "complete", "measurements": rows, "associations": associations,
              "source_results_sha256": digest(directory / "results.json"),
              "analysis_source_sha256": digest(Path(__file__)),
              "protocol": {
                  "status": "Exploratory follow-up; definitions were added after viewing replication results",
                  "spaces": "Raw integrated endpoints and visible endpoints clipped to [-1,1]",
                  "global": "Output covariance trace, participation rank, and effective prototype occupancy",
                  "local": "Output variance conditional on nearest tuned-example assignment",
                  "pair_control": "Within-prototype squared pair distance controls singleton clusters; pair counts retained",
                  "anchoring": "Squared distance to tuned examples, also normalized by untouched reference distance",
                  "association": "Positive minus-log baseline ratios indicate increased collapse; analyzed separately by architecture",
                  "scope": "Output diagnostics, not direct Jacobian rank or perturbation-basin measurements",
                  "selection": "All candidate associations reported; no maximum-correlation definition selected"}}
    write_json(directory / "collapse_analysis.json", result)
    write_report(directory, result)
    plot(directory, rows)
    return result


def write_report(directory, result):
    rows = result["measurements"]
    final_step = max(r["step"] for r in rows)
    lines = ["# Output Collapse Audit", "",
             "Exploratory follow-up using all saved generated samples, without retraining or changing FTSS. "
             "This audit distinguishes overall output diversity from spread around individual training examples.", "",
             "The covariance participation rank is trace(C)^2 / trace(C^2). It measures output-cloud "
             "dimension, not the rank of the flow Jacobian. Conditional variance groups samples by their "
             "nearest tuned example, then measures variance around each group's own mean. It therefore "
             "allows systematic deviations from training images and is not an exact-copy test. "
             "Prototype distance checks whether concentrated groups are actually anchored to training data. "
             "Pairwise conditional dispersion and pair counts are also saved to expose singleton-group effects.", "",
             "## Operational Construct", "",
             "Let Y be a generated output, a(Y) its nearest tuned-example index, T the tuned pool, and R "
             "the equal-size untouched pool. Define W = E[||Y - E[Y | a(Y)]||^2] and "
             "A = E[d(Y,T)^2] / E[d(Y,R)^2]. Training-anchored concentration means W decreases "
             "relative to its pre-fine-tuning baseline while A decreases toward zero. The first condition "
             "measures loss of variation around learned prototypes without requiring exact copies; "
             "the second establishes training specificity. Report the two quantities separately rather "
             "than choosing a correlation-optimized product or threshold. Effective prototype occupancy "
             "and global variance distinguish concentration around many examples from collapse to one output.", "",
             "## Final Outputs", "",
             "Ratios are final / initial, computed within each trajectory and averaged over three seeds. "
             "A value below one means reduced dispersion, rank, or distance. Effective prototypes is "
             "exp(entropy of nearest-training-example assignments). Both raw and displayed outputs are reported.", "",
             "| Model | Images | Space | Variance ratio | Rank ratio | Within-prototype variance ratio | "
             "Training distance-squared ratio | Train/reference distance-squared ratio | Effective prototypes |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for model in sorted({r["model"] for r in rows}):
        for size in sorted({r["size"] for r in rows}):
            selected = [r for r in rows if (r["model"], r["size"], r["step"]) == (model, size, final_step)]
            for space in ("raw", "visible"):
                metrics = ("output_variance", "participation_rank", "within_prototype_variance", "training_anchor_mse")
                values = [np.mean([np.exp(-r[f"{space}_{key}_collapse"]) for r in selected]) for key in metrics]
                values += [np.mean([r[f"{space}_training_reference_energy_ratio"] for r in selected]),
                           np.mean([r[f"{space}_effective_occupied_prototypes"] for r in selected])]
                lines.append(f"| {model} | {size} | {space} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += ["", "## Association With FTSS Contraction", "",
              "All candidate definitions are reported. Adjusted correlations remove trajectory intercepts "
              "and common step effects. Only three seed blocks are available; repeated checkpoints are "
              "not independent replications. These exploratory comparisons are not a validation of a "
              "definition selected by its correlation.", "",
              "| Model | Collapse outcome | Spearman | Adjusted Pearson |", "|---|---|---:|---:|"]
    for model, metrics in result["associations"].items():
        for name, values in metrics.items():
            cells = ["undefined" if values[key] is None else f"{values[key]:.3f}"
                     for key in ("spearman", "adjusted_pearson")]
            lines.append(f"| {model} | {name} | " + " | ".join(cells) + " |")
    lines += ["", "The intended training-anchored collapse construct combines reduced within-prototype "
              "spread with increased proximity to tuned rather than untouched examples. Global diversity "
              "loss and loss-based exposure are separate behavioral axes. Small FTSS estimates RMS "
              "local sensitivity; identifying Jacobian rank collapse additionally requires spectral information. "
              "In particular, RMS sensitivity depends on the sum of squared singular values, not their "
              "normalized spectrum: diag(1,1) and diag(sqrt(2),0) have the same RMS sensitivity but "
              "different rank. Output covariance rank is also distinct from Jacobian rank.", "",
              "## Context", "",
              "[Dynamical regimes of diffusion models](https://www.nature.com/articles/s41467-024-54281-3) "
              "discusses collapse onto training examples as a memorization mechanism. "
              "[Improved Precision and Recall](https://arxiv.org/abs/1904.06991) separates generated quality "
              "and coverage. The quantities W and A above are explicit exploratory definitions used in this "
              "audit, not metrics attributed to those papers.", ""]
    (directory / "collapse_report.md").write_text("\n".join(lines))


def plot(directory, rows):
    os.environ.setdefault("MPLCONFIGDIR", str((directory / ".matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    models = sorted({r["model"] for r in rows})
    settings = [("ftss_ratio", "FTSS gain / initial", False),
                ("raw_within_prototype_variance_collapse", "Within-prototype variance / initial", True),
                ("raw_output_variance_collapse", "Overall output variance / initial", True),
                ("raw_participation_rank_collapse", "Output participation rank / initial", True),
                ("raw_training_reference_energy_ratio", "Training / reference distance squared", False)]
    fig, axes = plt.subplots(len(settings), len(models), figsize=(5 * len(models), 13), squeeze=False)
    colors = ["#176f94", "#bd493c", "#34805d", "#96732f"]
    for col, model in enumerate(models):
        for i, (key, label, invert) in enumerate(settings):
            ax = axes[i, col]
            for size, color in zip(sorted({r["size"] for r in rows}), colors):
                x, mean, std = [], [], []
                for step in sorted({r["step"] for r in rows}):
                    selected = [r for r in rows if (r["model"], r["size"], r["step"]) == (model, size, step)]
                    values = [np.exp(-r[key]) if invert else r[key] for r in selected]
                    x.append(step)
                    mean.append(np.mean(values))
                    std.append(np.std(values))
                ax.errorbar(x, mean, yerr=std, marker="o", capsize=2, label=f"{size} images", color=color)
            ax.axhline(1., color="gray", linestyle=":", linewidth=0.8)
            ax.set_ylabel(label)
            ax.set_xlabel("Fine-tuning steps")
            ax.grid(alpha=0.15)
            if i == 0:
                ax.set_title(model.upper())
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "collapse_trajectories.png", dpi=160)
    fig.savefig(directory / "collapse_trajectories.pdf")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("directory", type=Path)
    result = analyze(p.parse_args().directory)
    for model, metrics in result["associations"].items():
        for metric, values in metrics.items():
            print(model, metric, values)
