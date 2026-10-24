"""Small real-FMA execution test for the Stable Audio rectified-flow base."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from run_main_architectures import MESSAGES_32


def commands():
    pool = "table_runs/cache/fma_stable_audio3_smoke.pt"
    jobs = [("encode", [sys.executable, "-u", "prepare_audio_latents.py", "--dataset", "fma",
             "--root", "data/audio/fma_small", "--metadata", "data/audio/fma_metadata/tracks.csv",
             "--codec", "stable", "--stable-model", "stabilityai/stable-audio-3-small-music-base",
             "--seconds", "6", "--max-train-clips", "2", "--max-test-clips", "2", "--output", pool])]
    for scheme, mode, message in (("orthogonal5", "orthogonal", "10101"),
                                  ("hypercube32", "hypercube", MESSAGES_32[0])):
        jobs.append((scheme, [sys.executable, "-u", "watermark_audio_flow.py", "--latents", pool,
                    "--dataset", "fma_stable_smoke", "--architecture", "stable", "--wm_message", message,
                    "--codebook_mode", mode, "--steps", "5", "--batch_size", "1", "--eval_batch_size", "2",
                    "--n_queries", "128", "--n_detect_trials", "2", "--n_fid_samples", "2",
                    "--n_sample_steps", "8", "--output_dir", f"table_runs/stable_audio_smoke/{scheme}"]))
    return jobs


def main():
    import soundfile as sf

    root = Path("table_runs/stable_audio_smoke")
    root.mkdir(parents=True, exist_ok=True)
    status = {"state": "running", "pid": os.getpid(), "smoke_only": True,
              "warning": "Tiny execution test, not an embedding-budget or quality study"}
    summary = root / "summary.json"
    status["per_scheme"] = []
    try:
        for name, command in commands():
            status["stage"] = name
            summary.write_text(json.dumps(status, indent=2) + "\n")
            print(f"Stage: {name}", flush=True)
            with (root / f"{name}.log").open("w") as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
            if name == "encode":
                continue
            directory = Path(command[command.index("--output_dir") + 1]) / "fma_stable_smoke" / command[command.index("--wm_message") + 1]
            result = json.loads((directory / "results.json").read_text())
            metrics = result["metrics"]
            if not metrics["quality_complete"] or not all(np.isfinite(metrics[k])
                    for k in ("fd_clap_clean", "fd_clap_wm", "fd_clap_ratio")):
                raise RuntimeError("Incomplete or nonfinite audio quality evaluation")
            audio_checks = []
            for path in sorted(directory.glob("audio_*/*.wav")):
                waveform, sample_rate = sf.read(path)
                rms = float(np.sqrt(np.mean(waveform ** 2)))
                if not np.isfinite(waveform).all() or rms < 1e-8:
                    raise RuntimeError(f"Nonfinite or silent generated audio: {path}")
                audio_checks.append({"path": str(path), "seconds": len(waveform) / sample_rate,
                                     "rms": rms, "peak": float(np.max(np.abs(waveform)))})
            if len(audio_checks) != 4:
                raise RuntimeError("Expected two clean and two marked generated recordings")
            status["per_scheme"].append({"scheme": name, "metrics": result["metrics"],
                                         "audio_checks": audio_checks})
        status["state"] = "complete"
    except Exception as exc:
        status.update(state="failed", error=str(exc))
        raise
    finally:
        summary.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
