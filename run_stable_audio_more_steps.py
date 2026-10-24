"""Steps-only continuation of the four partially recovered Stable Audio marks."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from run_main_architectures import MESSAGES_32


def commands():
    root = Path("table_runs/stable_audio_validation/hypercube32/fma_stable_validation")
    for message in MESSAGES_32:
        source = root / message
        result = json.loads((source / "results.json").read_text())
        if result["metrics"]["wm_acc"] == 100:
            continue
        yield message, [sys.executable, "-u", "watermark_audio_flow.py",
                        "--latents", "table_runs/cache/fma_stable_audio3_pilot.pt",
                        "--dataset", "fma_stable_more_steps", "--architecture", "stable",
                        "--wm_message", message, "--codebook_mode", "hypercube",
                        "--continue_adapter", str(source / "adapter_step_100.pt"),
                        "--steps", "200", "--checkpoint_every", "50",
                        "--batch_size", "1", "--eval_batch_size", "4",
                        "--n_queries", "4096", "--n_detect_trials", "20",
                        "--evaluation_seed_offset", "80000", "--n_fid_samples", "32",
                        "--n_sample_steps", "100", "--output_dir", "table_runs/stable_audio_more_steps/hypercube32"]


def main():
    root = Path("table_runs/stable_audio_more_steps")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "study.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {"state": "running", "pid": os.getpid(), "additional_updates": 200,
                  "total_updates": 300, "recipe_unchanged": True, "optimizer_restarted": True,
                  "quality_preliminary": True, "per_message": []}
        summary = root / "summary.json"
        try:
            for message, command in commands():
                status["stage"] = message
                summary.write_text(json.dumps(status, indent=2) + "\n")
                directory = root / "hypercube32" / "fma_stable_more_steps" / message
                if (directory / "results.json").exists():
                    raise RuntimeError(f"Refusing to overwrite completed continuation: {directory}")
                print(f"Continuing: {message}, 100 + 200 updates", flush=True)
                with (root / f"{message}.log").open("a") as stream:
                    subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
                result = json.loads((directory / "results.json").read_text())
                status["per_message"].append({"message": message, "metrics": result["metrics"]})
            status["state"] = "complete"
        except Exception as exc:
            status.update(state="failed", error=str(exc))
            raise
        finally:
            summary.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
