"""Separate ordinary recovery from clean-control contrast in sample-only students."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def contrast_summary(marked_trials, control_trials, bits):
    truth = np.asarray(bits, dtype=bool)
    marked = np.asarray([trial["scores"] for trial in marked_trials], dtype=float)
    control = np.asarray([trial["scores"] for trial in control_trials], dtype=float)
    if marked.ndim != 2 or control.shape != marked.shape or marked.shape[1] != len(truth):
        raise ValueError("Paired query trials and message dimensions must match")
    trial_difference = marked - control
    difference = trial_difference.mean(0)
    norm = float(np.linalg.norm(difference))
    correct = int(((difference > 0) == truth).sum()) if norm > 0 else None
    valid = np.linalg.norm(trial_difference, axis=1) > 0
    per_trial_accuracy = ((trial_difference > 0) == truth).mean(axis=1)
    return dict(message_bits=len(truth), query_trials=len(marked), score_change_l2=norm,
                nonzero_contrast_trials=int(valid.sum()),
                contrast_consensus_correct_bits=correct,
                contrast_consensus_accuracy_percent=None if correct is None else 100 * correct / len(truth),
                contrast_accuracy_percent=float(100 * per_trial_accuracy[valid].mean()) if valid.any() else None,
                reference_free_consensus_accuracy_percent=float(100 * ((marked.mean(0) > 0) == truth).mean()),
                clean_consensus_bit_agreement_percent=float(100 * ((control.mean(0) > 0) == truth).mean()),
                mean_signed_change=float((difference * (2 * truth.astype(float) - 1)).mean()))


def analyze(study, key_path):
    path = study / "results.json"
    results = json.loads(path.read_text())
    if results["state"] != "complete":
        raise ValueError("Wait for all predeclared students to finish before final analysis")
    manifest = json.loads((study / "config.json").read_text())
    if hashlib.sha256(key_path.read_bytes()).hexdigest() != manifest["key_sha256"]:
        raise ValueError("Wrong owner key")
    key = torch.load(key_path, map_location="cpu", weights_only=True)
    rows = []
    for marked in results["rows"]:
        if marked["role"] == "clean":
            continue
        control = next(r for r in results["rows"] if r["role"] == "clean" and r["seed"] == marked["seed"])
        row = dict(role=marked["role"], seed=marked["seed"],
                   reference_free_bit_accuracy_percent=marked["metrics"]["bit_accuracy_percent"],
                   clean_bit_agreement_percent=control["metrics"]["bit_accuracy_percent"],
                   paired_gain_pp=marked["metrics"]["bit_accuracy_percent"] - control["metrics"]["bit_accuracy_percent"])
        row.update(contrast_summary(marked["metrics"]["trials"], control["metrics"]["trials"], key["bits"].tolist()))
        rows.append(row)
    groups = []
    for role in sorted({r["role"] for r in rows}):
        selected = [r for r in rows if r["role"] == role]
        group = dict(role=role, student_seeds=len(selected))
        for name in ("reference_free_bit_accuracy_percent", "clean_bit_agreement_percent", "paired_gain_pp", "contrast_accuracy_percent",
                     "contrast_consensus_accuracy_percent", "reference_free_consensus_accuracy_percent"):
            values = [r[name] for r in selected if r[name] is not None]
            group[name + "_mean"] = float(np.mean(values)) if values else None
            group[name + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
        groups.append(group)
    analysis = dict(state="complete", rows=rows, groups=groups,
                    source_results_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    key_sha256=manifest["key_sha256"],
                    queries_per_model_per_trial=manifest["arguments"]["queries"],
                    method="For each matched query trial subtract clean-student scores before sign decoding, then average bit accuracy. Separately report consensus after averaging all query scores for each model.",
                    scope="Diagnostic only: requires clean-generated samples, a clean-trained student and matched generation/training/query randomness. Not reference-free recovery. One key; query trials are repeated measurements of the same 32 message bits.")
    output = study / "analysis.json"
    output.write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "table_runs/endpoint_loop_distillation")
    parser.add_argument("--key", type=Path, default=ROOT / "table_runs/baseline_first/tune/32/0/key.pt")
    args = parser.parse_args()
    print(json.dumps(analyze(args.study, args.key)["groups"], indent=2))
