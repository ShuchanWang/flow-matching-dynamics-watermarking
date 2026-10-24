"""Complete the four remaining messages with the successful pilot's training stages."""

import fcntl
import json
import os
from pathlib import Path
import subprocess

from run_main_architectures import MESSAGES_32
from run_stable_audio_margin_pilot import command as repair_command
from run_table_experiments import set_arg


ROOT = Path("table_runs/stable_audio_margin_suite")


def commands(message):
    existing = Path("table_runs/stable_audio_more_steps/hypercube32/fma_stable_more_steps") / message
    preparation = ROOT / "preparation" / "fma_stable_prep" / message
    jobs = []
    if (existing / "adapter_step_300.pt").exists():
        source = existing
    else:
        source = preparation
        if not (preparation / "adapter_step_300.pt").exists():
            original = Path("table_runs/stable_audio_validation/hypercube32/fma_stable_validation") / message
            cmd = repair_command(message, original)
            set_arg(cmd, "--continue_adapter", original / "adapter_step_100.pt")
            set_arg(cmd, "--dataset", "fma_stable_prep")
            set_arg(cmd, "--output_dir", ROOT / "preparation")
            set_arg(cmd, "--steps", 200)
            set_arg(cmd, "--wm_bit_weight", 0)
            cmd.remove("--compensate_clean_bias")
            cmd.extend(["--training_only"])
            jobs.append(("preparation", cmd))
    cmd = repair_command(message, source)
    set_arg(cmd, "--dataset", "fma_stable_margin_suite")
    set_arg(cmd, "--output_dir", ROOT / "hypercube32")
    jobs.append(("repair_validation", cmd))
    return jobs


def matching_result(result):
    config, metrics = result["config"], result["metrics"]
    expected = {"embedding_updates": 400, "initial_embedding_updates": 300,
                "steps": 100, "wm_bit_weight": 1.0, "compensate_clean_bias": True,
                "clean_bias_split": "train", "clean_bias_queries": 4096,
                "n_queries": 4096, "n_detect_trials": 20, "evaluation_seed_offset": 100000,
                "n_fid_samples": 32, "n_sample_steps": 100, "wm_eps": 1.5}
    return metrics.get("quality_complete") is True and all(config.get(k) == v for k, v in expected.items())


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "study.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pilot_path = Path("table_runs/stable_audio_margin_pilot/summary.json")
        pilot = json.loads(pilot_path.read_text())
        if pilot["state"] != "complete" or not matching_result(pilot):
            raise RuntimeError("Expected a completed pilot with the same repair/evaluation protocol")
        status = {"state": "running", "pid": os.getpid(), "total_updates": 400,
                  "quality_preliminary": True, "per_message": [
                      {"message": MESSAGES_32[0], "source": str(pilot_path), "metrics": pilot["metrics"]}]}
        summary = ROOT / "summary.json"
        try:
            for message in MESSAGES_32[1:]:
                result_path = ROOT / "hypercube32" / "fma_stable_margin_suite" / message / "results.json"
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                    if not matching_result(result):
                        raise RuntimeError(f"Existing result has a different protocol: {result_path}")
                    print(f"Skipping completed: {message}", flush=True)
                else:
                    for stage, cmd in commands(message):
                        status.update(message=message, stage=stage)
                        summary.write_text(json.dumps(status, indent=2) + "\n")
                        print(f"{message}: {stage}", flush=True)
                        with (ROOT / f"{message}_{stage}.log").open("a") as stream:
                            subprocess.run(cmd, check=True, stdout=stream, stderr=subprocess.STDOUT)
                    result = json.loads(result_path.read_text())
                    if not matching_result(result):
                        raise RuntimeError(f"Incomplete or mismatched result: {result_path}")
                status["per_message"].append({"message": message, "source": str(result_path), "metrics": result["metrics"]})
            status["state"] = "complete"
        except Exception as exc:
            status.update(state="failed", error=str(exc))
            raise
        finally:
            summary.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
