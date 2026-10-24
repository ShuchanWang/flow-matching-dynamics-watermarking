"""Refresh the guarded inline payload results after all study measurements finish."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from artifact_io import write_json


BEGIN = "% BEGIN GENERATED PAYLOAD RESULTS"
END = "% END GENERATED PAYLOAD RESULTS"


def payload_block(text):
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError("Payload block markers are missing or duplicated")
    first = text.index(BEGIN) + len(BEGIN)
    last = text.index(END, first)
    return first, last, text[first:last]


def block_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def render(summary):
    rows = summary["groups"]
    main = [next(r for r in rows if r["bits"] == bits and r["queries"] == 4096) for bits in (64, 96, 128)]
    if any(r["messages"] != 5 for r in main):
        raise ValueError("The main table requires five messages at each payload")
    lines = ["", "", "Table~\\ref{tab:payload-main} varies the hypercube payload on CIFAR-10",
             "RF-UNet at fixed projection dimension $k=128$, nominal signal strength",
             "$\\epsilon=1.5$, and query count $N=4096$. Bit accuracy measures",
             "individual decisions, while exact recovery requires every bit to be",
             "correct. FID ratios assess quality preservation at each payload.", "",
             "\\begin{table}[t]", "\\centering",
             "\\caption{Payload scaling for the keyed hypercube code on CIFAR-10",
             "RF-UNet at $k=128$ and $N=4096$. Mean $\\pm$ population standard",
             "deviation across five messages, with 40 detection trials per message.",
             "FID ratios use 500 generated samples and CIFAR-10 test images per run.}",
             "\\label{tab:payload-main}", "\\small", "\\begin{tabular}{cccc}", "\\toprule",
             "Bits & Exact recovery ($\\uparrow$) & Bit accuracy ($\\uparrow$) & FID ratio ($\\downarrow$) \\\\",
             "\\midrule"]
    for row in main:
        exact = f"${row['exact_recovery_percent_mean']:.1f}{{\\pm}}{row['exact_recovery_percent_std']:.1f}$\\%"
        bits = f"${row['bit_accuracy_percent_mean']:.2f}{{\\pm}}{row['bit_accuracy_percent_std']:.2f}$\\%"
        fid = f"${row['fid_ratio_mean']:.3f}{{\\pm}}{row['fid_ratio_std']:.3f}$"
        lines.append(f"{row['bits']} & {exact} & {bits} & {fid} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", "",
              "Figure~\\ref{fig:payload-queries} varies query count for the same",
              "watermarked models. The signed margin of bit $j$ is",
              "$(2m_j-1)\\langle\\hat{s},e_j\\rangle$; a negative margin gives",
              "an incorrect bit. Averaging reduces query variation, while the",
              "learned bit margins determine whether that averaging yields",
              "complete recovery.", ""]
    largest = max(r["queries"] for r in rows)
    long_row = next(r for r in rows if r["bits"] == 128 and r["queries"] == largest)
    if long_row["exact_recovery_percent_mean"] < 50 and long_row["bits_with_nonpositive_trial_mean_mean"] > 0:
        lines += ["At the longest payload, increasing query count does not restore",
                  "reliable recovery for all messages: some bits retain negative",
                  "empirical signed means. The observed payload transition thus",
                  "reflects both finite-query variation and the learned bit margins.", ""]
    lines += ["\\begin{figure}[t]", "\\centering",
              "\\includegraphics[width=\\textwidth]{figures/payload_query_scaling.pdf}",
              "\\caption{Query scaling for 64-, 96-, and 128-bit hypercube watermarks",
              "at fixed $k=128$ and $\\epsilon=1.5$. Curves show exact recovery,",
              "bit accuracy, and the minimum signed bit margin, averaged over five",
              "messages; error bars indicate population standard deviation. Each",
              "query count uses 40 trials per message, with shared query prefixes",
              "within each trial.}", "\\label{fig:payload-queries}", "\\end{figure}", "", ""]
    return "\n".join(lines)


def finalize(directory, paper, expected):
    summary = json.loads((directory / "summary.json").read_text())
    if summary["state"] != "complete":
        raise ValueError("Incomplete payload study")
    replacement = render(summary)
    (directory / "paper_payload_block.tex").write_text(replacement)
    text = paper.read_text()
    first, last, current = payload_block(text)
    if block_hash(current) != expected and current != replacement:
        write_json(directory / "paper_update.json", {"state": "needs_merge",
                   "reason": "Payload section changed during the experiment; user edits preserved",
                   "replacement": str(directory / "paper_payload_block.tex")})
        return
    destination = paper.parent / "figures" / "payload_query_scaling.pdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(directory / "query_scaling.pdf", destination)
    paper.write_text(text[:first] + replacement + text[last:])
    build = directory / "latex_build"
    build.mkdir(exist_ok=True)
    commands = [["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={build}", paper.name],
                ["biber", f"--input-directory={build}", f"--output-directory={build}", paper.stem]]
    commands += [commands[0], commands[0]]
    with (directory / "latex_build.log").open("w") as log:
        for command in commands:
            subprocess.run(command, cwd=paper.parent, stdout=log, stderr=subprocess.STDOUT, check=True)
    shutil.copyfile(build / "main.pdf", paper.with_suffix(".pdf"))
    write_json(directory / "paper_update.json", {"state": "complete", "paper": str(paper),
               "pdf": str(paper.with_suffix(".pdf")), "figure": str(destination)})


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--paper", type=Path, required=True)
    p.add_argument("--expected-block-sha256", required=True)
    args = p.parse_args()
    finalize(args.directory.resolve(), args.paper.resolve(), args.expected_block_sha256)
