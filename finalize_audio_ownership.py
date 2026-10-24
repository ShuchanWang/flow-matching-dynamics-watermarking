"""Publish only a complete, validated audio ownership audit into the paper."""

import argparse
import fcntl
import json
from pathlib import Path
import statistics
import subprocess
import sys

from run_audio_ownership import ROOT, PROTOCOL, summarize, write_json


def audio_groups(rows):
    groups = []
    for architecture, name in (("mlp", "Audio MLP"), ("transformer", "Audio Transformer"), ("stable", "Stable Audio")):
        for mode, code in (("orthogonal", "O5"), ("hypercube", "H32")):
            group = [r for r in rows if r["architecture"] == architecture and r["codebook_mode"] == mode]
            if len(group) != 5 or len({r["message"] for r in group}) != 5:
                raise ValueError(f"Incomplete group: {architecture}/{mode}")
            groups.append((name, code, group))
    return groups


def cell(values, precision=1):
    return f"${statistics.mean(values):.{precision}f}{{\\pm}}{statistics.pstdev(values):.{precision}f}$"


def render(rows):
    groups = audio_groups(rows)
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{Ownership verification for settings with an evaluated acceptance rule.",
             r"FPR, FNR, and BER are percentages, mean $\pm$ population standard deviation across five messages.",
             r"Protocol details are given in the implementation appendix.}",
             r"\label{tab:error-metrics}\label{tab:audio-ownership}",
             r"\begin{tabular}{lllccc}\toprule",
             r"Generator & Data & Code & FPR & FNR & BER \\",
             r"\midrule"]
    for scheme, code in (("orthogonal5", "O5"), ("hypercube5", "H5"), ("hypercube32", "H32")):
        source = Path("table_runs/repairs/mlp") / scheme / "repair_results.json"
        messages = json.loads(source.read_text())["per_message"]
        if len(messages) != 5 or len({m["message"] for m in messages}) != 5:
            raise ValueError(f"Incomplete MLP setting: {scheme}")
        cells = [cell([m["clean"]["ownership_acceptance"] for m in messages]),
                 cell([100 - m["after"]["ownership_acceptance"] for m in messages]),
                 cell([100 - m["after"]["bit_accuracy"] for m in messages])]
        lines.append(" & ".join(["Image MLP", "MNIST", code, *cells]) + r" \\")
    for name, code, group in groups:
        cells = [cell([r["metrics"][metric] for r in group])
                 for metric in ("ownership_fpr_pct", "ownership_fnr_pct", "ber_pct")]
        lines.append(" & ".join([name, "FMA", code, *cells]) + r" \\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(lines) + "\n"


def render_details(rows):
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{Audio verification diagnostics using the same frozen thresholds as",
             r"Table~\ref{tab:error-metrics}. Wrong-key acceptance is in percent;",
             r"AUC ranks marked and clean target scores. Mean $\pm$ population standard deviation across five messages.}",
             r"\label{tab:audio-score-diagnostics}",
             r"\begin{tabular}{llcc}\toprule",
             r"Generator & Code & Wrong-key acceptance & Score AUC \\",
             r"\midrule"]
    for name, code, group in audio_groups(rows):
        cells = [cell([r["metrics"]["wrong_key_acceptance_pct"] for r in group]),
                 cell([r["metrics"]["score_auc_vs_clean"] for r in group], 3)]
        lines.append(" & ".join([name, code, *cells]) + r" \\")
    lines += [r"\bottomrule\end{tabular}\end{table}",
              r"AUC evaluates score ranking, while FPR/FNR use the frozen message-and-score decision.",
              r"One fixed wrong key is evaluated per message; these trials do not establish a population rate over keys."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--wait", action="store_true", help="Wait for the running evaluator's lock")
    args = parser.parse_args()
    if args.wait:
        with (ROOT / "study.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
    state = json.loads((ROOT / "summary.json").read_text())
    if state["state"] != "complete" or set(state["architectures"]) != {"mlp", "transformer", "stable"}:
        raise RuntimeError("All three audio architectures must complete before paper publication")
    rows = []
    for architecture in ("mlp", "transformer", "stable"):
        for path in sorted((ROOT / architecture).glob("*_*.json")):
            if path.name.endswith(".progress.json") or path.name == "frozen_thresholds.json":
                continue
            row = json.loads(path.read_text())
            if row["protocol"] != PROTOCOL or row["metrics"] != summarize(row["calibration_scores"], row["trials"]):
                raise ValueError(f"Metrics/protocol mismatch: {path}")
            rows.append(row)
    if len(rows) != 30:
        raise ValueError("Expected exactly 30 final evaluations")
    content = render(rows)
    target = Path("paper/audio_ownership_results.tex")
    temporary = target.with_suffix(".tmp")
    temporary.write_text(content)
    temporary.replace(target)
    details = Path("paper/audio_ownership_details.tex")
    temporary = details.with_suffix(".tmp")
    temporary.write_text(render_details(rows))
    temporary.replace(details)
    subprocess.run([sys.executable, "audit_saved_metrics.py"], check=True)
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd="paper", check=True)
    write_json(ROOT / "paper_status.json", {"state": "complete", "rows": len(rows), "table": str(target), "pdf": "paper/main.pdf"})


if __name__ == "__main__":
    main()
