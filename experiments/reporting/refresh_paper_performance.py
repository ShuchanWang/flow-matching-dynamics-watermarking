"""Build one performance table from final checkpoints and measured quality."""

import argparse
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def hf_group(root):
    paths = sorted(Path(root).glob("*/results.json"))
    if len(paths) != 5:
        raise ValueError(f"Expected five messages: {root}")
    return [dict(read(p)["metrics"], source=str(p)) for p in paths]


def sd_group(scheme):
    if scheme == "orthogonal5":
        paths = sorted(Path("table_runs").glob("sd35_msg_*/sweep_results.json"))
    else:
        paths = sorted(Path("table_runs/main_architectures/sd35") .joinpath(scheme).glob("*/sweep_results.json"))
    if len(paths) != 5:
        raise ValueError("Expected five SD3.5 messages")
    rows = []
    for path in paths:
        data = read(path)
        result = data["results"][-1]
        quality_path = Path("table_runs/consistent_image_quality/sd35") / scheme / path.parent.name / "quality.json"
        quality = read(quality_path)["quality"] if quality_path.exists() else None
        if quality is not None and quality["sampling_protocol"] != "diffusers_sd3":
            raise ValueError("Expected standard SD3 sampling")
        rows.append({"wm_acc": result["detection_accuracy_wm"],
                     "clean_fp": result["detection_accuracy_clean"],
                     "fid_ratio": quality["fid_ratio"] if quality else None, "source": str(path)})
    return rows


def groups():
    output = []
    fresh = read("table_runs/fresh_mlp_fpr/results.json")
    if not fresh.get("complete"):
        raise ValueError("Fresh MLP clean-model evaluation must complete before publication")
    randomized_mlp = {r["scheme"]: r["random_key_random_claim"] for r in fresh["summary"]}
    for scheme, code in (("orthogonal5", "O5"), ("hypercube5", "H5"), ("hypercube32", "H32")):
        source = Path("table_runs/repairs/mlp") / scheme / "repair_results.json"
        quality = read(Path("table_runs/mlp_inception_quality") / f"{scheme}.json")
        if not quality.get("complete") or len(quality["per_message"]) != 5:
            raise ValueError("MLP Inception quality evaluation must complete before publication")
        ratios = {r["message"]: r["fid_ratio"] for r in quality["per_message"]}
        rows = [{"wm_acc": r["after"]["raw_exact_recovery"],
                 "clean_fp": r["clean"]["raw_exact_recovery"],
                 "randomized_fpr": randomized_mlp[scheme],
                 "fid_ratio": ratios[r["message"]], "source": str(source)} for r in read(source)["per_message"]]
        output.append(("Image MLP", "MNIST", code, rows))
    for dataset, name in (("mnist", "MNIST"), ("cifar10", "CIFAR-10"), ("celeba64", "CelebA-64")):
        for scheme, code in (("orthogonal5", "O5"), ("hypercube5", "H5"), ("hypercube32", "H32")):
            if scheme == "orthogonal5":
                root = Path("table_runs/hf_flow_unet") / dataset
            elif scheme == "hypercube32" and dataset == "cifar10":
                root = Path("table_runs/flow_unet_payload_hypercube/cifar10")
            else:
                root = Path("table_runs/main_architectures/unet") / scheme / dataset
            output.append(("RF-UNet", name, code, hf_group(root)))
    for scheme, code in (("orthogonal5", "O5"), ("hypercube5", "H5")):
        output.append(("SD3.5 DiT", "Image latents", code, sd_group(scheme)))
    diagnostics = read("table_runs/repairs/sd35/hypercube32/suite_results.json")["per_message"]
    pilot = read("table_runs/sd35_diagnostics/hypercube32_compensated_eps3_01100011.json")
    if pilot["message"] not in {r["message"] for r in diagnostics}:
        diagnostics.append(pilot)
    quality = read("table_runs/repairs/sd35/hypercube32/standard_quality/quality_results.json")
    ratios = {r["message"]: r["quality"]["fid_ratio"] for r in quality["per_message"]}
    output.append(("SD3.5 DiT", "Image latents", "H32", [
        {"wm_acc": r["wm"]["exact_recovery"], "clean_fp": r["clean"]["exact_recovery"],
         "fid_ratio": ratios[r["message"]], "source": r.get("result", r.get("source"))} for r in diagnostics]))
    for architecture, name in (("mlp", "Audio MLP"), ("transformer", "Audio Transformer")):
        for scheme, code in (("orthogonal5", "O5"), ("hypercube32", "H32")):
            rows = hf_group(Path("table_runs/main_architectures/audio") / architecture / scheme / "fma")
            for row in rows:
                row["fid_ratio"] = row["fd_clap_ratio"]
            output.append((name, "FMA", code, rows))
    stable = hf_group("table_runs/stable_audio_validation/orthogonal5/fma_stable_validation")
    for row in stable:
        row["fid_ratio"] = row["fd_clap_ratio"]
    output.append(("Stable Audio", "FMA", "O5", stable))
    paths = list(Path("table_runs/stable_audio_margin_pilot/hypercube32").glob("*/*/results.json"))
    paths += list(Path("table_runs/stable_audio_margin_suite/hypercube32").glob("*/*/results.json"))
    rows = [dict(read(path)["metrics"], source=str(path)) for path in paths]
    for row in rows:
        row["fid_ratio"] = row["fd_clap_ratio"]
    output.append(("Stable Audio", "FMA", "H32", rows))
    for name, dataset, code, rows in output:
        cohort = "_".join((name, dataset, code)).replace(" ", "_").replace(".", "")
        path = Path("table_runs/paper_fpr") / f"{cohort}.json"
        if path.exists():
            report = read(path)
            if not report.get("complete") or (report["generator"], report["dataset"], report["code"]) != (name, dataset, code):
                raise ValueError("Invalid cohort FPR report")
            for row in rows:
                row["randomized_fpr"] = report["metric"]
    return output


def cell(values, precision=1):
    return f"${statistics.mean(values):.{precision}f}{{\\pm}}{statistics.pstdev(values):.{precision}f}$"


def render(data, require_complete=False):
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{Recovery, clean-model FPR, and quality preservation across",
             r"image and audio generators at $k=32$."]
    pending = any(r["fid_ratio"] is None for _, _, _, rows in data for r in rows)
    missing_fpr = any(not any(r.get("randomized_fpr") is not None for r in rows) for _, _, _, rows in data)
    if missing_fpr and require_complete:
        raise ValueError("All FPR measurements must complete before final publication")
    if missing_fpr:
        lines.append(r"A dash denotes an unevaluated FPR.")
    if pending and require_complete:
        raise ValueError("Standard-sampler SD3.5 O5/H5 quality must complete before final publication")
    if pending:
        lines.append(r"A dash denotes the pending standard-sampler SD3.5 quality evaluations.")
    lines += [r"}", r"\label{tab:main}\label{tab:hypercube-main}\label{tab:audio-main}",
              r"\label{tab:main-quality}\label{tab:quality}\label{tab:encoding-quality}",
              r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\textwidth}{!}{\begin{tabular}{lllccc}\toprule",
              r"Generator & Data & Code & Recovery (\%) & FPR (\%) & Quality ratio \\", r"\midrule"]
    previous = None
    for name, dataset, code, rows in data:
        if len(rows) != 5:
            raise ValueError(f"Expected five messages: {name}/{dataset}/{code}")
        recovery = [r["wm_acc"] for r in rows]
        negative = [r["clean_fp"] for r in rows]
        if any(not 0 <= value <= 100 for value in recovery + negative):
            raise ValueError("Invalid recovery/FPR rates")
        quality = [r["fid_ratio"] for r in rows]
        if any(v is not None and (not math.isfinite(v) or v < 0) for v in quality):
            raise ValueError("Invalid quality ratios")
        quality_cell = cell(quality, 3) if all(v is not None for v in quality) else r"\textemdash"
        randomized = [r.get("randomized_fpr") for r in rows]
        fpr_cell = r"\textemdash"
        if any(r is not None for r in randomized):
            if randomized[0] is None or any(r != randomized[0] for r in randomized):
                raise ValueError("Expected one shared randomized FPR study for the cohort")
            metric = randomized[0]
            if metric["trials"] < 1 or not 0 <= metric["matches"] <= metric["trials"]:
                raise ValueError("Invalid randomized FPR counts")
            observed = 100 * metric["matches"] / metric["trials"]
            if not math.isclose(metric["fpr_pct"], observed, abs_tol=1e-12):
                raise ValueError("Randomized FPR disagrees with match counts")
            fpr_cell = f"{observed:.3f}" if metric["matches"] else "0 (observed)"
        if previous is not None and previous != (name, dataset):
            lines.append(r"\addlinespace[2pt]")
        lines.append(" & ".join([name, dataset, code, cell(recovery),
                                 fpr_cell, quality_cell]) + r" \\")
        previous = (name, dataset)
    lines += [r"\bottomrule\end{tabular}}\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    data = groups()
    content = render(data, args.require_complete)
    path = Path("paper/performance_results.tex")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(content)
    temporary.replace(path)
    print(f"Published {len(data)} final model/codebook settings")


if __name__ == "__main__":
    main()
