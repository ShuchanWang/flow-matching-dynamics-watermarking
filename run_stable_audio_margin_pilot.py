"""Single-message training-only clean-bias and raw-bit-margin repair pilot."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from run_main_architectures import MESSAGES_32


def command(message=MESSAGES_32[0], source=None):
    if source is None:
        source = Path("table_runs/stable_audio_more_steps/hypercube32/fma_stable_more_steps") / message
    return [sys.executable, "-u", "watermark_audio_flow.py",
            "--latents", "table_runs/cache/fma_stable_audio3_pilot.pt",
            "--dataset", "fma_stable_margin_pilot", "--architecture", "stable",
            "--wm_message", message, "--codebook_mode", "hypercube",
            "--continue_adapter", str(source / "adapter_step_300.pt"),
            "--steps", "100", "--checkpoint_every", "50",
            "--batch_size", "1", "--eval_batch_size", "4",
            "--wm_bit_weight", "1", "--compensate_clean_bias", "--clean_bias_queries", "4096",
            "--n_queries", "4096", "--n_detect_trials", "20",
            "--evaluation_seed_offset", "100000", "--n_fid_samples", "32",
            "--n_sample_steps", "100", "--output_dir", "table_runs/stable_audio_margin_pilot/hypercube32"]


def main():
    root = Path("table_runs/stable_audio_margin_pilot")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "study.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {"state": "running", "pid": os.getpid(), "pilot_only": True,
                  "source_updates": 300, "repair_updates": 100, "total_updates": 400,
                  "rank_unchanged": True, "nominal_strength_unchanged": True,
                  "decoder_unchanged": True, "calibration_split": "train",
                  "objective_changed": True, "optimizer_restarted": True,
                  "quality_preliminary": True}
        summary = root / "summary.json"
        cmd = command()
        result_path = root / "hypercube32" / "fma_stable_margin_pilot" / MESSAGES_32[0] / "results.json"
        if result_path.exists():
            raise RuntimeError("Refusing to overwrite a completed margin pilot")
        try:
            summary.write_text(json.dumps(status, indent=2) + "\n")
            with (root / "pilot.log").open("a") as stream:
                subprocess.run(cmd, check=True, stdout=stream, stderr=subprocess.STDOUT)
            result = json.loads(result_path.read_text())
            status.update(state="complete", metrics=result["metrics"], config=result["config"])
        except Exception as exc:
            status.update(state="failed", error=str(exc))
            raise
        finally:
            summary.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
