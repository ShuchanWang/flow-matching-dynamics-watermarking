"""Validate the saved early checkpoint and the remaining Stable Audio messages."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from run_main_architectures import SCHEMES


def commands():
    first = "10101"
    for scheme in ("orthogonal5", "hypercube32"):
        mode, messages = SCHEMES[scheme]
        ordered = [first] + [m for m in messages if m != first] if scheme == "orthogonal5" else messages
        for message in ordered:
            command = [sys.executable, "-u", "watermark_audio_flow.py",
                       "--latents", "table_runs/cache/fma_stable_audio3_pilot.pt",
                       "--dataset", "fma_stable_validation", "--architecture", "stable",
                       "--wm_message", message, "--codebook_mode", mode,
                       "--steps", "100", "--batch_size", "1", "--eval_batch_size", "4",
                       "--n_queries", "4096", "--n_detect_trials", "20",
                       "--evaluation_seed_offset", "60000",
                       "--n_fid_samples", "32", "--n_sample_steps", "100",
                       "--output_dir", f"table_runs/stable_audio_validation/{scheme}"]
            if scheme == "orthogonal5" and message == first:
                command.extend(["--evaluate_adapter",
                                "table_runs/stable_audio_pilot/orthogonal5/fma_stable_pilot/10101/adapter_step_100.pt"])
            else:
                command.extend(["--checkpoint_every", "100"])
            yield f"{scheme}_{message}", command


def main():
    root = Path("table_runs/stable_audio_validation")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "study.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {"state": "running", "pid": os.getpid(), "quality_preliminary": True,
                  "per_message": []}
        summary = root / "summary.json"
        try:
            for name, command in commands():
                status["stage"] = name
                summary.write_text(json.dumps(status, indent=2) + "\n")
                directory = Path(command[command.index("--output_dir") + 1]) / "fma_stable_validation" / command[command.index("--wm_message") + 1]
                result_path = directory / "results.json"
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                    expected = {"n_queries": 4096, "n_detect_trials": 20, "evaluation_seed_offset": 60000,
                                "n_fid_samples": 32, "n_sample_steps": 100, "embedding_updates": 100}
                    if not result["metrics"].get("quality_complete") or any(result["config"].get(k) != v for k, v in expected.items()):
                        raise RuntimeError(f"Existing result has a different protocol: {result_path}")
                    print(f"Skipping completed: {name}", flush=True)
                else:
                    print(f"Starting: {name}", flush=True)
                    with (root / f"{name}.log").open("a") as stream:
                        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
                    result = json.loads(result_path.read_text())
                status["per_message"].append({"name": name, "metrics": result["metrics"]})
            status["state"] = "complete"
        except Exception as exc:
            status.update(state="failed", error=str(exc))
            raise
        finally:
            summary.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
