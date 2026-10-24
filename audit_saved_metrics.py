"""Derive saved exact-message and optional score-gated verification metrics."""

import argparse
import csv
import json
import math
from pathlib import Path


def percentage(value):
    if value is None:
        return None
    value = float(value)
    if not -1e-5 <= value <= 100 + 1e-5:
        raise ValueError(f"Invalid percentage: {value}")
    return min(100.0, max(0.0, value))


def empirical_auc(positive, negative):
    if not positive or not negative:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in positive for n in negative) / (len(positive) * len(negative))


def metric_row(source, config, metrics, phase="final", condition=None):
    recovery = percentage(metrics.get("raw_exact_recovery", metrics.get("exact_recovery",
                              metrics.get("wm_acc", metrics.get("detection_accuracy_wm")))))
    if recovery is None:
        return None
    message = metrics.get("message", config.get("wm_message", config.get("message", "")))
    bits = metrics.get("bits", config.get("wm_bits", len(message)))
    bit_accuracy = percentage(metrics.get("bit_accuracy"))
    ber = 100 - bit_accuracy if bit_accuracy is not None else (0.0 if recovery == 100 else None)
    failure = 100 - recovery
    return {
        "source": str(source), "phase": phase, "condition": condition,
        "model": metrics.get("model", config.get("architecture", config.get("model_family"))),
        "dataset": metrics.get("dataset", config.get("dataset")), "message": message,
        "bits": bits, "codebook_mode": metrics.get("codebook_mode", config.get("codebook_mode")),
        "queries": metrics.get("n_queries", metrics.get("N", config.get("n_queries", config.get("queries")))),
        "trials": metrics.get("n_detect_trials", config.get("n_detect_trials", config.get("trials"))),
        "exact_recovery_pct": recovery, "recovery_failure_pct": failure,
        "ber_pct": ber, "ber_basis": "saved_bit_accuracy" if bit_accuracy is not None else
            ("all_messages_exact" if ber == 0 else "unavailable"),
        "ber_lower_bound_pct": failure / bits if bits else None,
        "ber_upper_bound_pct": failure,
        "clean_target_match_pct": percentage(metrics.get("clean_target_match", metrics.get("clean_fp",
                                                     metrics.get("detection_accuracy_clean")))),
        "wrong_key_target_match_pct": percentage(metrics.get("wrong_key_acc")),
        "ownership_fpr_pct": None, "ownership_fnr_pct": None,
        "ownership_tpr_pct": None, "ownership_threshold": None,
        "ownership_caveat": "No saved ownership decision rule; raw matches are not ownership FPR",
        "roc_auc": None, "roc_score": None, "roc_exploratory_only": None,
        "tpr_at_specified_fpr": None,
    }


def diagnostic_rows(path, payload):
    if "wm" not in payload or not isinstance(payload["wm"], dict):
        return []
    config = dict(payload, architecture="sd35", dataset="sd35_latents", wm_message=payload.get("message", ""))
    row = metric_row(path, config, payload["wm"])
    if row is None:
        return []
    row["clean_target_match_pct"] = percentage(payload.get("clean", {}).get("exact_recovery"))
    positive = payload["wm"].get("minimum_signed_coordinate_per_trial", [])
    negative = payload.get("clean", {}).get("minimum_signed_coordinate_per_trial", [])
    if positive and negative:
        row.update(roc_auc=empirical_auc(positive, negative), roc_score="minimum_signed_bit_coordinate",
                   roc_exploratory_only=True)
    return [row]


def mlp_repair_rows(path, payload):
    rows = []
    for item in payload.get("per_message", []):
        config = dict(payload.get("config", {}), model_family="classic_mlp", dataset="mnist",
                      wm_message=item["message"], codebook_mode="hypercube" if path.parent.name.startswith("hypercube") else "orthogonal")
        for phase in ("before", "after"):
            if phase not in item:
                continue
            row = metric_row(path, config, item[phase], phase)
            if row is None:
                continue
            clean = item.get("clean", {})
            row["clean_target_match_pct"] = percentage(clean.get("raw_exact_recovery"))
            positive = percentage(item[phase].get("ownership_acceptance"))
            negative = percentage(clean.get("ownership_acceptance"))
            if positive is not None and negative is not None:
                row.update(ownership_fpr_pct=negative, ownership_fnr_pct=100 - positive,
                           ownership_tpr_pct=positive, ownership_threshold=item[phase].get("threshold"),
                           ownership_caveat=payload.get("gate_caveat", "Empirical calibrated gate"))
            rows.append(row)
    return rows


def extract(path, payload):
    if not isinstance(payload, dict):
        return []
    if "calibration_scores" in payload and "protocol" in payload and "metrics" in payload:
        metrics = payload["metrics"]
        config = {"architecture": payload["architecture"], "dataset": "fma",
                  "wm_message": payload["message"], "codebook_mode": payload["codebook_mode"],
                  "n_queries": payload["protocol"]["queries"], "n_detect_trials": payload["protocol"]["test_trials"]}
        row = metric_row(path, config, {"wm_acc": 100 - metrics["recovery_failure_pct"],
                                       "bit_accuracy": 100 - metrics["ber_pct"]}, phase="ownership_audit")
        row.update(ownership_fpr_pct=metrics["ownership_fpr_pct"],
                   ownership_fnr_pct=metrics["ownership_fnr_pct"],
                   ownership_tpr_pct=metrics["ownership_tpr_pct"],
                   ownership_threshold=metrics["threshold"], ownership_caveat=payload["caveat"],
                   roc_auc=metrics["score_auc_vs_clean"], roc_score="target_message_score",
                   roc_exploratory_only=False)
        return [row]
    if path.name == "repair_results.json":
        return mlp_repair_rows(path, payload)
    if path.name.startswith("recovery_step_"):
        records = payload.get("trials", [])
        if not records:
            return []
        # Truth is in the message directory, not necessarily the decoded prediction.
        message = path.parent.name
        config = {"wm_message": message, "architecture": "stable", "dataset": "fma_stable_screening",
                  "queries": payload["queries"], "trials": len(records)}
        metrics = {"wm_acc": 100 * sum(r["wm"]["exact"] for r in records) / len(records),
                   "bit_accuracy": 100 * sum(r["wm"]["bit_accuracy"] for r in records) / len(records),
                   "clean_fp": 100 * sum(r["clean"]["exact"] for r in records) / len(records)}
        row = metric_row(path, config, metrics, phase="screening", condition=f"step_{payload['step']}")
        mode = payload.get("codebook_mode") or (
            "hypercube" if len(message) > 20 or any(part.startswith("hypercube") for part in path.parts)
            else "orthogonal")
        row["codebook_mode"] = mode
        def score(values):
            if mode == "hypercube":
                return sum(v * (2 * int(b) - 1) for v, b in zip(values, message)) / math.sqrt(len(message))
            return values[sum(int(b) << i for i, b in enumerate(message))]
        if all("scores" in r[label] for r in records for label in ("wm", "clean")):
            row.update(roc_auc=empirical_auc([score(r["wm"]["scores"]) for r in records],
                                            [score(r["clean"]["scores"]) for r in records]),
                       roc_score="target_message_score", roc_exploratory_only=True)
        return [row]
    diagnostic = diagnostic_rows(path, payload)
    if diagnostic:
        return diagnostic
    rows = []
    config = payload.get("config", {})
    if path.name in ("results.json", "recovery_results.json"):
        if path.name == "recovery_results.json" and (path.parent / "results.json").exists():
            return []
        if payload.get("per_message"):
            entries = payload["per_message"]
        else:
            entries = [payload.get("metrics", {})]
        for metrics in entries:
            row = metric_row(path, config, metrics)
            if row:
                rows.append(row)
        for index, metrics in enumerate(payload.get("sweeps", [])):
            row = metric_row(path, config, metrics, phase="sweep", condition=f"{config.get('sweep')}:{metrics.get('value', index)}")
            if row:
                rows.append(row)
    elif path.name == "sweep_results.json":
        for index, metrics in enumerate(payload.get("results", [])):
            row = metric_row(path, config, metrics, phase="sweep", condition=str(metrics.get("value", index)))
            if row:
                rows.append(row)
    elif "wm_acc" in payload and "condition" in payload:
        row = metric_row(path, payload, payload, phase="robustness", condition=payload["condition"])
        if row:
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=Path("table_runs"))
    parser.add_argument("--output", type=Path, default=Path("table_runs/metrics_audit"))
    args = parser.parse_args()
    rows, unreadable = [], []
    for path in sorted(args.root.rglob("*.json")):
        if args.output == path.parent or args.output in path.parents:
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            unreadable.append(str(path))
            continue
        rows.extend(extract(path, payload))
    for row in rows:
        row.update(exact_match_fpr_pct=row["clean_target_match_pct"] if row["message"] else None,
                   exact_match_fnr_pct=row["recovery_failure_pct"] if row["message"] else None,
                   exact_match_decision="Accept iff decoded message equals predeclared target")
        bits = row["bits"]
        row.update(random_claim_baseline_pct=100 * 2.0**(-bits) if isinstance(bits, int) and bits > 0 else None,
                   random_claim_baseline_basis="Analytical uniform independent full-codebook claim; ungated decoder returns one word")
    args.output.mkdir(parents=True, exist_ok=True)
    caveats = ["All rates and BER are percentages; blank/null means unavailable.",
               "For the fixed exact-message decision, clean target matches are empirical FPR and marked recovery failures are FNR.",
               "The separately saved ownership_fpr/fnr fields apply an additional calibrated score gate; distinguish that decision from exact_match_fpr/fnr.",
               "Zero BER follows logically from 100% exact recovery; partial recovery alone gives BER bounds only.",
               "MLP ownership rates use the saved empirical clean-calibrated gate on one clean model, not a guaranteed population FPR.",
               "Do not pool screening, robustness, original, and repaired runs or different query budgets.",
               "Exploratory empirical ROC-AUC is computed only where both per-trial score distributions are saved; score definitions differ across runs.",
               "TPR at a specified FPR is not estimated from score means/std or a post-hoc threshold. Screening AUC has only two trials per class."]
    report = {"rows": rows, "caveats": caveats, "unreadable_during_scan": unreadable}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    if rows:
        with (args.output / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "ber_available": sum(r["ber_pct"] is not None for r in rows),
                      "ownership_rates_available": sum(r["ownership_fpr_pct"] is not None for r in rows),
                      "exploratory_auc_available": sum(r["roc_auc"] is not None for r in rows),
                      "output": str(args.output), "unreadable": unreadable}, indent=2))


if __name__ == "__main__":
    main()
