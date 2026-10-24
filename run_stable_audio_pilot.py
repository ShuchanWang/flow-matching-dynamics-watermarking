"""Bounded Stable Audio screening study, separate from the full paper protocol."""

import fcntl
import json
import os
from pathlib import Path
import subprocess

from run_stable_audio_smoke import commands as smoke_commands


def commands():
    jobs = smoke_commands()
    replacements = {
        "--max-train-clips": "128", "--max-test-clips": "32",
        "--steps": "500", "--n_queries": "1024", "--n_detect_trials": "5",
        "--n_fid_samples": "32", "--n_sample_steps": "50",
    }
    for name, original in jobs:
        command = [arg.replace("fma_stable_audio3_smoke.pt", "fma_stable_audio3_pilot.pt")
                   .replace("stable_audio_smoke/", "stable_audio_pilot/")
                   .replace("fma_stable_smoke", "fma_stable_pilot") for arg in original]
        for option, value in replacements.items():
            if option in command:
                command[command.index(option) + 1] = value
        if name != "encode":
            command.extend(["--checkpoint_every", "100", "--checkpoint_queries", "512",
                            "--checkpoint_trials", "2"])
        yield name, command


def main():
    root = Path("table_runs/stable_audio_pilot")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "study.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {"state": "running", "pid": os.getpid(), "screening_only": True,
                  "per_scheme": []}
        summary = root / "summary.json"
        try:
            for name, command in commands():
                status["stage"] = name
                summary.write_text(json.dumps(status, indent=2) + "\n")
                print(f"Stage: {name}", flush=True)
                if name == "encode":
                    pool = Path(command[command.index("--output") + 1])
                    if pool.exists():
                        print(f"Using existing pilot pool: {pool}", flush=True)
                        continue
                else:
                    directory = Path(command[command.index("--output_dir") + 1]) / "fma_stable_pilot" / command[command.index("--wm_message") + 1]
                    result_path = directory / "results.json"
                    if result_path.exists():
                        raise RuntimeError(f"Refusing to overwrite completed pilot: {result_path}")
                with (root / f"{name}.log").open("w") as stream:
                    subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
                if name != "encode":
                    result = json.loads(result_path.read_text())
                    status["per_scheme"].append({"scheme": name, "metrics": result["metrics"]})
            status["state"] = "complete"
        except Exception as exc:
            status.update(state="failed", error=str(exc))
            raise
        finally:
            summary.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
