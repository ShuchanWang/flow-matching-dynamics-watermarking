"""Publish completed cosine-removal measurements into the existing paper table."""
from repo_layout import ROOT, source_path

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


LABEL = "\\label{tab:removal-transfer}"


def table_span(text):
    label = text.index(LABEL)
    return text.rfind("\\begin{table}", 0, label), text.index("\\end{table}", label) + len("\\end{table}")


def render_rows(results):
    if results.get("state") != "complete":
        raise ValueError("Cannot publish an incomplete experiment")
    rows = {}
    for scheme in ("O5", "H32"):
        selected = [r for r in results["rows"] if r["scheme"] == scheme]
        if len(selected) != 5 or len({r["message"] for r in selected}) != 5:
            raise ValueError(f"Need all five distinct {scheme} messages")
        values = []
        for key, digits in (("recovery_percent", 1), ("bit_accuracy_percent", 2), ("fid_ratio_teacher", 3)):
            data = np.asarray([r["metrics"][key] for r in selected], dtype=float)
            if not np.isfinite(data).all():
                raise ValueError("Nonfinite results")
            values.append(f"${np.mean(data):.{digits}f}{{\\pm}}{np.std(data):.{digits}f}$")
        rows[scheme] = "& Cosine-carrier subtraction & " + " & ".join(values) + r" \\" + "\n"
    return rows


def publish(root, paper):
    results = json.loads((root / "results.json").read_text())
    rows = render_rows(results)
    plan = json.loads((root / "paper_update_plan.json").read_text())
    text = paper.read_text()
    start, end = table_span(text)
    table = text[start:end]
    expected = hashlib.sha256(table.encode()).hexdigest()
    if expected != plan["table_sha256"]:
        raise ValueError("The removal table changed during the run; preserve edits and integrate manually")
    if "Cosine-carrier subtraction" in table:
        raise ValueError("Cosine rows already present")
    original_lines = table.splitlines(keepends=True)
    updated, scheme = [], None
    for line in original_lines:
        if line.startswith("O5 &"):
            scheme = "O5"
        elif line.startswith("H32 &"):
            scheme = "H32"
        updated.append(line)
        if line.startswith("& Sine-carrier subtraction"):
            updated.append(rows.pop(scheme))
    if rows:
        raise ValueError("Could not locate both sine-carrier rows")
    text = text[:start] + "".join(updated) + text[end:]
    if all(row["metrics"]["recovery_percent"] == 100 for row in results["rows"]):
        anchor = "encodings, whereas strengths one and 1.5 give near-chance bit accuracy."
        addition = ("\nIn contrast, mismatched cosine subtraction preserves exact recovery\n"
                    "for both encodings. Sine and cosine are orthogonal under uniform-time\n"
                    "sampling, so the subtracted cosine component has zero population\n"
                    "correlation with the owner's sine carrier.")
        if text.count(anchor) != 1:
            raise ValueError("Removal discussion changed; preserve edits and integrate manually")
        text = text.replace(anchor, anchor + addition)
    temporary = paper.with_suffix(".carrier.tmp")
    temporary.write_text(text)
    temporary.replace(paper)
    (root / "paper_update_done.json").write_text(json.dumps(dict(
        paper=str(paper), sha256=hashlib.sha256(text.encode()).hexdigest()), indent=2) + "\n")


def build(paper, root):
    directory = Path("/tmp/carrier_mismatch_final_paper")
    directory.mkdir(exist_ok=True)
    with (root / "paper_build.log").open("w") as log:
        latex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                 f"-output-directory={directory}", paper.name]
        subprocess.run(latex, cwd=paper.parent, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(["biber", "--input-directory", str(directory), "--output-directory",
                        str(directory), paper.stem], cwd=paper.parent,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
        for _ in range(2):
            subprocess.run(latex, cwd=paper.parent, stdout=log, stderr=subprocess.STDOUT, check=True)
    shutil.copyfile(directory / "main.pdf", paper.with_suffix(".pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "table_runs/carrier_mismatch")
    parser.add_argument("--paper", type=Path, default=ROOT / "paper/main.tex")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run-study", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        text = args.paper.read_text()
        start, end = table_span(text)
        path = args.root / "paper_update_plan.json"
        if path.exists():
            raise FileExistsError("Existing publication plan is preserved")
        path.write_text(json.dumps(dict(table_sha256=hashlib.sha256(text[start:end].encode()).hexdigest()),
                                   indent=2) + "\n")
    else:
        if args.run_study:
            subprocess.run([sys.executable, "-u",
                            str(source_path('run_carrier_mismatch.py')), "--output", str(args.root)],
                           cwd=ROOT, check=True)
        publish(args.root, args.paper)
        build(args.paper, args.root)
        print("Cosine-removal results inserted; paper/main.pdf rebuilt", flush=True)
