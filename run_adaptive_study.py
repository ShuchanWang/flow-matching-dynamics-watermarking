"""Run unfinished CIFAR-10 studies sequentially with explicit stopping rules."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


MESSAGES = ("00000", "00111", "01010", "10101", "11001")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--stop-accuracy", type=float, default=100.0,
                        help="100 stops at imperfect recovery; 3.125 stops at chance-level pooled recovery")
    cli = parser.parse_args()
    root = Path("table_runs/adaptive_study")
    root.mkdir(parents=True, exist_ok=True)
    jobs = []

    def save():
        (root / "status.json").write_text(json.dumps(jobs, indent=2) + "\n")

    def run(name, script, *args):
        log = root / f"{name}.log"
        job = {"name": name, "state": "running", "log": str(log)}
        jobs.append(job)
        save()
        print(f"Starting {name}: {log}", flush=True)
        with log.open("a") as stream:
            result = subprocess.run([sys.executable, "-u", script, *map(str, args)],
                                    stdout=stream, stderr=subprocess.STDOUT)
        job.update(state="complete" if result.returncode == 0 else "failed",
                   exit_code=result.returncode)
        save()
        print(f"{name}: {job['state']}", flush=True)
        return result.returncode == 0

    # Give the short embedding replay an uncontended GPU for timing.
    fine_output = Path("table_runs/embedding_budget/cifar10_fine")
    if not (fine_output / "summary.json").exists():
        if run("embedding_1_to_10", "evaluate_embedding_budget.py", "--refine",
               "--refine-limit", 10, "--refine-save-every", 1, "--output", fine_output):
            summary = json.loads((fine_output / "summary.json").read_text())
            if summary["earliest_tested_success"] is None:
                run("embedding_11_to_100", "evaluate_embedding_budget.py", "--refine",
                    "--refine-limit", 100, "--refine-save-every", 1,
                    "--output", fine_output)

    stopped = False
    for lr, output, steps in (
        ("1e-5", Path("table_runs/robustness_cifar10"), [100, 500, 1000, 1500, 2000, 3000, 5000]),
        ("1e-4", Path("table_runs/robustness_cifar10_lr1e4"), [100, 250, 500, 1000, 1500, 2000, 3000, 5000]),
    ):
        if not run(f"baseline_lr{lr}", "run_flow_unet_robustness.py", "--messages", "all",
                   "--only", "baseline", "--ft_lr", lr, "--output_dir", output):
            break
        for step in steps:
            if not run(f"ft_lr{lr}_{step}", "run_flow_unet_robustness.py", "--messages", "all",
                       "--only", "ft", "--ft_steps", step, "--ft_lr", lr, "--output_dir", output):
                break
            rows = [json.loads((output / message / f"clean_ft_{step}.json").read_text())
                    for message in MESSAGES]
            mean_accuracy = sum(row["wm_acc"] for row in rows) / len(rows)
            failed = (mean_accuracy < 100.0 if cli.stop_accuracy == 100.0
                      else mean_accuracy <= cli.stop_accuracy)
            outcome = {"lr": lr, "step": step, "mean_recovery": mean_accuracy,
                       "message_recovery": {row["message"]: row["wm_acc"] for row in rows},
                       "stop_accuracy": cli.stop_accuracy, "stop_reached": failed}
            (root / "finetune_search.json").write_text(json.dumps(outcome, indent=2) + "\n")
            if failed:
                print(f"Stopping clean fine-tuning: {outcome}", flush=True)
                stopped = True
                break
        if stopped:
            break
    if not stopped:
        print("Stopping rule not reached within 5,000 updates per learning rate; no 10,000-update run queued.", flush=True)

    run("remaining_pruning_quantization", "run_flow_unet_robustness.py", "--messages", "all",
        "--only", "prune", "quant", "--output_dir", "table_runs/robustness_cifar10")
    print("Queue finished; check status.json for any failed jobs.", flush=True)


if __name__ == "__main__":
    main()
