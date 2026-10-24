"""Sequential FMA codec smoke tests, latent preparation, and main audio grid."""
from repo_layout import ROOT, source_path

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import torch

from run_main_architectures import MESSAGES_32


def preparation_command(output, train_count, test_count):
    return [sys.executable, "-u", str(source_path('prepare_audio_latents.py')), "--dataset", "fma",
            "--root", "data/audio/fma_small", "--metadata", "data/audio/fma_metadata/tracks.csv",
            "--max-train-clips", str(train_count), "--max-test-clips", str(test_count),
            "--output", str(output)]


def cache_complete(path, train_count, test_count):
    if not path.exists():
        return False
    data = torch.load(path, map_location="cpu", weights_only=False)
    config = data.get("config", {})
    return (config.get("codec") == "music2latent" and config.get("dataset") == "fma"
            and config.get("seconds") == 6.0
            and config.get("root") == "data/audio/fma_small"
            and config.get("metadata") == "data/audio/fma_metadata/tracks.csv"
            and len(data["train"]) == train_count and len(data["test"]) == test_count
            and len(data["test_audio"]) == test_count
            and not set(data["sources"]["train"]) & set(data["sources"]["test"])
            and all(torch.isfinite(data[s]).all() for s in ("train", "test", "test_audio")))


def smoke_commands(pool):
    jobs = []
    for architecture in ("mlp", "transformer"):
        for scheme, mode, message in (("orthogonal5", "orthogonal", "10101"),
                                      ("hypercube32", "hypercube", MESSAGES_32[0])):
            jobs.append([sys.executable, "-u", str(source_path('watermark_audio_flow.py')), "--latents", str(pool),
                         "--dataset", "fma_smoke", "--architecture", architecture,
                         "--wm_message", message, "--codebook_mode", mode,
                         "--base_steps", "5", "--steps", "5", "--n_queries", "128",
                         "--n_detect_trials", "2", "--n_fid_samples", "2",
                         "--n_sample_steps", "2", "--batch_size", "2", "--eval_batch_size", "2",
                         "--output_dir", f"table_runs/audio_smoke/{architecture}/{scheme}",
                         "--base_dir", "table_runs/audio_smoke/bases"])
    return jobs


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    root = Path("table_runs")
    root.mkdir(exist_ok=True)
    status_path = root / "audio_study_status.json"
    status = {"pid": os.getpid(), "state": "running", "started": datetime.now(timezone.utc).isoformat()}

    def run(command, stage):
        status.update(stage=stage, command=command)
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        print(f"\nStage: {stage}\n{' '.join(command)}", flush=True)
        subprocess.run(command, check=True)

    with (root / "audio_study.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("An FMA audio study is already running")
        try:
            smoke_pool = root / "cache/fma_music2latent_smoke.pt"
            if not cache_complete(smoke_pool, 2, 2):
                run(preparation_command(smoke_pool, 2, 2), "smoke_latents")
            for index, command in enumerate(smoke_commands(smoke_pool)):
                run(command, f"smoke_{index + 1}_of_4")
            if not args.smoke_only:
                pool = root / "cache/fma_music2latent.pt"
                if not cache_complete(pool, 2000, 500):
                    run(preparation_command(pool, 2000, 500), "prepare_2000_train_500_test")
                run([sys.executable, "-u", str(source_path('run_main_architectures.py')), "--families", "audio",
                     "--audio-pools", f"fma={pool}", "--schemes", "orthogonal5", "hypercube32",
                     "--manifest", "table_runs/manifest_main_audio.json", "--run", "--skip_existing"],
                    "main_20_jobs")
                run([sys.executable, str(source_path('collect_results.py')), "--manifest", "table_runs/manifest_main_audio.json",
                     "--out_dir", "table_runs/summary_main_audio"], "aggregate")
        except Exception as exc:
            status.update(state="failed", error=str(exc))
            raise
        else:
            status.update(state="complete")
        finally:
            status["updated"] = datetime.now(timezone.utc).isoformat()
            status_path.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
