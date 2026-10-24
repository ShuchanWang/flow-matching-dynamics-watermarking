"""Run frozen development tuning, then independent-key confirmation, sequentially."""
from repo_layout import ROOT, source_path

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

import run_watermark_security as security


def main(output):
    root = ROOT
    output.mkdir(parents=True, exist_ok=True)
    status = dict(state="starting", protocol_sha256=security.digest(root / "baseline_protocol.md"),
                  runner_sha256=security.digest(source_path('run_baseline_first.py')))
    with (output / "sequence.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for stage in ("tune", "confirm"):
                status.update(state="running", stage=stage)
                security.write_json(output / "sequence.json", status)
                subprocess.run([sys.executable, "-u", str(source_path('run_baseline_first.py')),
                                "--output", str(output), "--stage", stage], cwd=root, check=True)
                results = json.loads((output / stage / "results.json").read_text())
                if results["state"] != "complete":
                    raise RuntimeError(f"Stage {stage} did not complete")
            status.update(state="ready_for_attack_design", stage=None)
            print("Tuning and five-key confirmation complete. Comparative attacks are the next stage.", flush=True)
        except Exception as exc:
            status.update(state="failed", error=repr(exc))
            raise
        finally:
            security.write_json(output / "sequence.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "table_runs/baseline_first")
    args = parser.parse_args()
    main(args.output.resolve())
