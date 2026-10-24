"""Summarize a completed paired FTSS fine-tuning study without thresholding FTSS."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


FIELDS = ("g_min", "ftss_ratio_to_pretrained_base", "replication_rate",
          "reference_match_rate", "wm_recovery_pct", "tuned_flow_loss",
          "reference_flow_loss")


def summarize(directory):
    config = json.loads((directory / "config.json").read_text())
    report = json.loads((directory / "results.json").read_text())
    if report["state"] != "complete":
        raise ValueError("The study is not complete")
    records = report["measurements"]
    keys = sorted({(r["size"], r["arm"], r["step"]) for r in records})
    rows = []
    for size, arm, step in keys:
        selected = [r for r in records if (r["size"], r["arm"], r["step"]) == (size, arm, step)]
        if len(selected) != config["runs"]:
            raise ValueError("Incomplete seed coverage")
        row = {"size": size, "arm": arm, "step": step, "runs": len(selected)}
        for field in FIELDS:
            values = [r[field] for r in selected]
            row[field + "_mean"] = float(np.mean(values))
            row[field + "_std"] = float(np.std(values))
        row["generated_samples"] = len(selected) * config["n_generated"]
        row["replicated_samples"] = sum(round(r["replication_rate"] * config["n_generated"])
                                         for r in selected)
        rows.append(row)
    result = {"config": config, "per_condition": rows,
              "ftss_interpretation": "Lower ratios indicate spectral contraction relative to the saved clean model; no binary FTSS threshold is imposed.",
              "copy_interpretation": config["replication_definition"]}
    (directory / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (directory / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    os.environ.setdefault("MPLCONFIGDIR", str((directory / ".matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    sizes = sorted({r["size"] for r in rows})
    fig, axes = plt.subplots(3, len(sizes), figsize=(5 * len(sizes), 8), squeeze=False)
    settings = [("ftss_ratio_to_pretrained_base", "FTSS / pretrained baseline", 1),
                ("replication_rate", "Generated-image copying (%)", 100),
                ("wm_recovery_pct", "Watermark recovery (%)", 1)]
    for col, size in enumerate(sizes):
        for i, (field, label, scale) in enumerate(settings):
            ax = axes[i, col]
            for arm, name, color in [("clean_ft", "Clean fine-tuning", "#176f94"),
                                      ("watermark_ft", "Watermark embedding", "#bf493b")]:
                selected = sorted([r for r in rows if r["size"] == size and r["arm"] == arm],
                                  key=lambda r: r["step"])
                ax.errorbar([r["step"] for r in selected],
                            [scale * r[field + "_mean"] for r in selected],
                            yerr=[scale * r[field + "_std"] for r in selected],
                            label=name, color=color, marker="o", capsize=3)
            ax.set_ylabel(label)
            ax.set_xlabel("Fine-tuning steps")
            ax.grid(alpha=0.2)
            if i == 0:
                ax.axhline(1, color="gray", linestyle=":", linewidth=1)
                ax.set_title(f"{size} fine-tuning images")
            if i == 2:
                ax.set_ylim(-3, 103)
            if i == 1:
                upper = max(5, max(100 * r["replication_rate_mean"] for r in rows) * 1.2)
                ax.set_ylim(0, upper)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(directory / "comparison.png", dpi=180)
    fig.savefig(directory / "comparison.pdf")
    plt.close(fig)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    summary = summarize(parser.parse_args().directory)
    for row in summary["per_condition"]:
        if row["step"] == max(summary["config"]["steps"]) or row["arm"] == "saved_watermark":
            print(f"n={row['size']} {row['arm']} step={row['step']} "
                  f"FTSS/base={row['ftss_ratio_to_pretrained_base_mean']:.4f} "
                  f"copy={row['replicated_samples']}/{row['generated_samples']} "
                  f"WM={row['wm_recovery_pct_mean']:.1f}%")
