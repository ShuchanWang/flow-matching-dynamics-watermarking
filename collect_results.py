"""Collect table experiment outputs and report mean/std across messages.

Reads either a manifest created by run_table_experiments.py or recursively
searches an output root. It understands:

  - watermark_sd35.py outputs: sweep_results.json
  - watermark_hf_flow_unet.py outputs: results.json
  - flow_watermark_mnist_mlp.py / flow_watermark_unet_lora.py outputs:
    results.json with per-message rows

The main purpose is to avoid reporting a single lucky watermark message.
Use the aggregate CSV for paper tables and keep the per-run JSON as audit
trail.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


METRICS = [
    "detection_accuracy_wm",
    "detection_accuracy_clean",
    "separation_sigma",
    "score_margin",
    "wm_score_mean",
    "wm_score_std",
    "wm_margin_mean",
    "wm_margin_std",
    "nearest_code_cosine",
    "clean_score_mean",
    "clean_score_std",
    "paired_endpoint_distance_mean",
    "fid_ratio",
    "fid_real_clean",
    "fid_real_wm",
    "wm_acc",
    "clean_fp",
    "sep_sigma",
]


def resolved_output_dir(cmd: list[str]) -> Path | None:
    if "--output_dir" not in cmd:
        return None
    out_dir = Path(cmd[cmd.index("--output_dir") + 1])
    if "--dataset" in cmd and "--wm_message" in cmd:
        return out_dir / cmd[cmd.index("--dataset") + 1] / cmd[cmd.index("--wm_message") + 1]
    return out_dir


def read_manifest_outputs(manifest: Path) -> list[tuple[str, str, Path]]:
    with open(manifest) as f:
        jobs = json.load(f)
    outputs = []
    for job in jobs:
        cmd = job.get("command", [])
        out_dir = resolved_output_dir(cmd)
        if out_dir is None:
            continue
        outputs.append((job.get("name", out_dir.name), job.get("table", ""), out_dir))
    return outputs


def discover_outputs(root: Path) -> list[tuple[str, str, Path]]:
    dirs = set()
    for path in root.rglob("sweep_results.json"):
        dirs.add(path.parent)
    for path in root.rglob("results.json"):
        dirs.add(path.parent)
    return [(path.name, "", path) for path in sorted(dirs)]


def final_sd35_rows(path: Path) -> list[dict]:
    result_path = path / "sweep_results.json"
    if not result_path.exists():
        return []
    with open(result_path) as f:
        payload = json.load(f)
    config = payload.get("config", {})
    rows = []
    for row in payload.get("results", []):
        merged = dict(row)
        merged.setdefault("message", config.get("wm_message"))
        merged.setdefault("bits", len(str(config.get("wm_message", ""))))
        merged.setdefault("model_family", "sd35")
        merged.setdefault("output_dir", str(path))
        rows.append(merged)
    return rows


def hf_rows(path: Path) -> list[dict]:
    result_paths = [path / "results.json"] if (path / "results.json").exists() else sorted(path.rglob("results.json"))
    rows = []

    def enrich(row: dict, config: dict, result_path: Path) -> dict:
        row.setdefault("message", config.get("wm_message"))
        row.setdefault("bits", len(str(config.get("wm_message", ""))))
        row.setdefault("model_family", "hf_flow_unet")
        row.setdefault("dataset", config.get("dataset"))
        row.setdefault("steps", config.get("steps"))
        row.setdefault("post_ft_steps", config.get("post_ft_steps"))
        row.setdefault("n_detect_queries", row.get("value", config.get("n_queries")))
        row.setdefault("output_dir", str(result_path.parent))
        return row

    for result_path in result_paths:
        with open(result_path) as f:
            payload = json.load(f)
        config = payload.get("config", {})
        if config.get("model_family") in {"classic_mlp", "classic_unet"}:
            continue
        metrics = payload.get("metrics", {})
        if not metrics:
            continue
        rows.append(enrich(dict(metrics), config, result_path))
        for sweep_row in payload.get("sweeps", []):
            rows.append(enrich(dict(sweep_row), config, result_path))
    return rows


def classic_rows(path: Path) -> list[dict]:
    result_path = path / "results.json"
    if not result_path.exists():
        return []
    with open(result_path) as f:
        payload = json.load(f)
    config = payload.get("config", {})
    family = config.get("model_family")
    if family not in {"classic_mlp", "classic_unet"}:
        return []

    rows = []
    for item in payload.get("per_message", []):
        row = dict(item)
        row.setdefault("message", item.get("message"))
        row.setdefault("bits", config.get("wm_bits", len(str(item.get("message", "")))))
        row.setdefault("model_family", family)
        row.setdefault("dataset", config.get("dataset"))
        row.setdefault("steps", config.get("steps", config.get("target_steps")))
        row.setdefault("lora_steps", config.get("lora_steps"))
        row.setdefault("lora_rank", config.get("lora_rank"))
        row.setdefault("n_detect_queries", config.get("n_queries"))
        row.setdefault("output_dir", str(path))
        rows.append(row)
    return rows


def infer_group(job_name: str, row: dict) -> tuple[str, str]:
    if row.get("model_family") == "classic_mlp":
        return "classic_mlp_main", str(row.get("dataset", "mnist"))
    if row.get("model_family") == "classic_unet":
        return "classic_unet_main", str(row.get("dataset", "unknown"))
    if job_name.startswith("sd35_main"):
        return "sd35_main", "5bit"
    if job_name.startswith("sd35_cross"):
        return "sd35_cross", "matrix"
    if job_name.startswith("sd35_query"):
        m = re.search(r"sd35_query_(\d+)", job_name)
        return "sd35_query", m.group(1) if m else str(row.get("n_detect_queries"))
    if job_name.startswith("sd35_epsilon"):
        m = re.search(r"sd35_epsilon_([0-9.]+)", job_name)
        return "sd35_epsilon", m.group(1) if m else "unknown"
    if job_name.startswith("sd35_rank"):
        m = re.search(r"sd35_rank_(\d+)", job_name)
        return "sd35_rank", m.group(1) if m else "unknown"
    if job_name.startswith("sd35_steps"):
        return "sd35_steps", str(row.get("steps"))
    if job_name.startswith("sd35_payload"):
        m = re.search(r"sd35_payload_(\d+)bit", job_name)
        return "sd35_payload", f"{m.group(1)}bit" if m else str(row.get("bits"))
    if job_name.startswith("flow_unet_payload"):
        m = re.search(r"flow_unet_payload_capacity_k(\d+)_([^_]+)_(\d+)bit", job_name)
        if m:
            return f"flow_unet_payload_capacity_{m.group(2)}_k{m.group(1)}", f"{m.group(3)}bit"
        m = re.search(r"flow_unet_payload_capacity_([^_]+)_(\d+)bit", job_name)
        if m:
            return f"flow_unet_payload_capacity_{m.group(1)}", f"{m.group(2)}bit"
        m = re.search(r"flow_unet_payload_stress_([^_]+)_N(\d+)_(\d+)bit", job_name)
        if m:
            return (
                f"flow_unet_payload_stress_{m.group(1)}_{m.group(3)}bit",
                f"N={row.get('n_detect_queries', m.group(2))}",
            )
        m = re.search(r"flow_unet_payload_stress_([^_]+)_(\d+)bit", job_name)
        if m:
            return f"flow_unet_payload_stress_{m.group(1)}", f"{m.group(2)}bit"
        m = re.search(r"flow_unet_payload_([^_]+)_(\d+)bit", job_name)
        if m:
            return f"flow_unet_payload_{m.group(1)}", f"{m.group(2)}bit"
        return "flow_unet_payload", str(row.get("bits"))
    if job_name.startswith("flow_unet_epsilon"):
        m = re.search(r"flow_unet_epsilon_([0-9.]+)", job_name)
        return "flow_unet_epsilon", m.group(1) if m else "unknown"
    if job_name.startswith("flow_unet_rank"):
        m = re.search(r"flow_unet_rank_(\d+)", job_name)
        return "flow_unet_rank", m.group(1) if m else "unknown"
    if job_name.startswith("flow_unet_finetune"):
        m = re.search(r"flow_unet_finetune_(\d+)", job_name)
        return "flow_unet_finetune", m.group(1) if m else str(row.get("post_ft_steps"))
    if job_name.startswith("flow_unet_cifar_query"):
        return "flow_unet_query", str(row.get("n_detect_queries"))
    if job_name.startswith("flow_unet"):
        return "flow_unet_main", str(row.get("dataset", "unknown"))
    if "query" in str(row.get("output_dir", "")):
        return "query", str(row.get("n_detect_queries"))
    return "misc", str(row.get("steps", "final"))


def collect(outputs: list[tuple[str, str, Path]]) -> list[dict]:
    rows = []
    seen = set()
    for job_name, table, out_dir in outputs:
        for row in final_sd35_rows(out_dir) + classic_rows(out_dir) + hf_rows(out_dir):
            group, setting = infer_group(job_name, row)
            row["job_name"] = job_name
            row["table"] = table
            row["group"] = group
            row["setting"] = setting
            identity = (
                row.get("output_dir"),
                group,
                setting,
                str(row.get("message")),
                row.get("steps"),
                row.get("post_ft_steps"),
                row.get("n_detect_queries"),
                row.get("wm_eps"),
                row.get("lora_rank"),
                row.get("bits"),
            )
            if identity in seen:
                continue
            seen.add(identity)
            rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row.get("group"), row.get("setting"))].append(row)

    agg_rows = []
    for (group, setting), bucket in sorted(buckets.items()):
        out = {
            "group": group,
            "setting": setting,
            "n_messages": len({str(r.get("message")) for r in bucket}),
            "n_runs": len(bucket),
            "messages": ",".join(sorted({str(r.get("message")) for r in bucket})),
        }
        for metric in METRICS:
            vals = [r.get(metric) for r in bucket if isinstance(r.get(metric), (int, float))]
            if vals:
                out[f"{metric}_mean"] = mean(vals)
                out[f"{metric}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
                out[f"{metric}_min"] = min(vals)
                out[f"{metric}_max"] = max(vals)
        agg_rows.append(out)
    return agg_rows


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--root", default="table_runs")
    parser.add_argument("--out_dir", default="table_runs/summary")
    return parser.parse_args()


def main():
    args = parse_args()
    outputs = read_manifest_outputs(Path(args.manifest)) if args.manifest else discover_outputs(Path(args.root))
    rows = collect(outputs)
    agg_rows = aggregate(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "per_run.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(out_dir / "aggregate.json", "w") as f:
        json.dump(agg_rows, f, indent=2)
    write_csv(out_dir / "per_run.csv", rows)
    write_csv(out_dir / "aggregate.csv", agg_rows)

    print(f"Collected {len(rows)} runs into {len(agg_rows)} aggregate rows.")
    print(f"Summary directory: {out_dir}")
    for row in agg_rows:
        wm = row.get("detection_accuracy_wm_mean", row.get("wm_acc_mean"))
        sep = row.get("separation_sigma_mean", row.get("sep_sigma_mean"))
        wm_s = row.get("detection_accuracy_wm_std", row.get("wm_acc_std"))
        sep_s = row.get("separation_sigma_std", row.get("sep_sigma_std"))
        print(
            f"{row['group']}[{row['setting']}]: n={row['n_runs']} "
            f"msgs={row['n_messages']} WM={wm}±{wm_s} Sep={sep}±{sep_s}"
        )


if __name__ == "__main__":
    main()
