"""Validate batched FTSS against saved literal paper evaluations on real models."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from ftss_batched import estimate_ftss_batched
from run_ftss_finetuning import digest, write_json
from run_matched_ftss_mlp import original_definitions


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--study", type=Path, default=Path("table_runs/ftss_mlp_matched"))
    p.add_argument("--repo", type=Path, default=Path("../ftss-memorization"))
    args = p.parse_args()
    report = json.loads((args.study / "results.json").read_text())
    torch.set_num_threads(4)
    device = torch.device("cuda")
    source = original_definitions(args.repo, device, None)
    results = []
    for row in report["scratch"]:
        model = source["MLPFlow"]().to(device)
        model.load_state_dict(torch.load(row["checkpoint"], map_location="cpu", weights_only=True)["model"])
        torch.manual_seed(90000 + row["run"])
        torch.cuda.synchronize()
        start = time.monotonic()
        result = estimate_ftss_batched(model, device)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        reference = np.asarray(row["metrics"]["mean_curve"])
        difference = np.abs(result["mean_curve"] - reference)
        passed = bool(np.allclose(result["mean_curve"], reference, rtol=5e-4, atol=5e-5))
        record = {"size": row["size"], "run": row["run"], "passed": passed,
                  "max_absolute_curve_error": float(difference.max()),
                  "max_relative_curve_error": float((difference / np.maximum(reference, 1e-12)).max()),
                  "minimum_absolute_error": float(abs(result["min"] - row["metrics"]["min"])),
                  "batched_ftss_seconds": elapsed, "checkpoint_sha256": digest(Path(row["checkpoint"]))}
        results.append(record)
        print(record, flush=True)
        del model
    result = {"all_passed": len(results) >= 2 and all(r["passed"] for r in results),
              "candidate_source_sha256": digest(Path(__file__).with_name("ftss_batched.py")),
              "reference_results_sha256": digest(args.study / "results.json"),
              "reference": "Saved literal paper estimator, same seeds, probes, times, and Euler steps",
              "tolerance": {"rtol": 5e-4, "atol": 5e-5}, "models": results}
    write_json(args.study / "batched_validation.json", result)
    if not result["all_passed"]:
        raise RuntimeError("Batched estimator failed real-model equivalence checks")


if __name__ == "__main__":
    main()
