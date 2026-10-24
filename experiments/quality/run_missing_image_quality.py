"""Finish consistent image-quality measurements and refresh the shared paper table."""
from repo_layout import ROOT, source_path

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path("table_runs/consistent_image_quality")


def save(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def wait_for_study(root, require_complete=True):
    with (root / "study.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
    report = json.loads((root / "summary.json").read_text())
    if require_complete and report["state"] != "complete":
        raise RuntimeError(f"Dependency did not complete: {root}")


def refresh(final=False):
    command = [sys.executable, str(source_path('refresh_paper_performance.py'))]
    if final:
        command.append("--require-complete")
    subprocess.run(command, check=True)
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd="paper", check=True)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "study.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {"state": "running", "pid": os.getpid(), "training": False,
                  "mlp_samples": 500, "sd35_samples": 30, "per_message": []}
        summary = ROOT / "summary.json"
        try:
            status["stage"] = "waiting_for_mlp_fid"
            save(summary, status)
            wait_for_study(Path("table_runs/mlp_inception_quality"))
            refresh()
            status["stage"] = "waiting_for_audio_gpu_release"
            save(summary, status)
            # Audio is only a GPU scheduling dependency, not a quality reference.
            wait_for_study(Path("table_runs/audio_ownership"), require_complete=False)
            sources = {
                "orthogonal5": sorted(Path("table_runs").glob("sd35_msg_*/sweep_results.json")),
                "hypercube5": sorted(Path("table_runs/main_architectures/sd35/hypercube5").glob("*/sweep_results.json"))}
            for scheme, paths in sources.items():
                if len(paths) != 5:
                    raise ValueError(f"Expected five saved SD3.5 checkpoints: {scheme}")
                for source in paths:
                    out = ROOT / "sd35" / scheme / source.parent.name / "quality.json"
                    config = json.loads(source.read_text())["config"]
                    command = [sys.executable, "-u", str(source_path('diagnose_sd35_hypercube.py')),
                               "--run-dir", str(source.parent), "--output", str(out),
                               "--quality-only", "--quality-samples", "30",
                               "--quality-protocol", "standard", "--quality-batch-size", "8",
                               "--loading-gpu-budget-gib", "12"]
                    existing = json.loads(out.read_text()) if out.exists() else None
                    if existing is not None:
                        quality = existing["quality"]
                        if (existing["source"] != str(source.parent) or not existing["quality_only"]
                                or quality["sampling_protocol"] != "diffusers_sd3" or quality["n_samples"] != 30
                                or quality["n_sampling_steps"] != config["n_eval_steps"]):
                            raise ValueError(f"Existing quality result has a different protocol: {out}")
                    else:
                        out.parent.mkdir(parents=True, exist_ok=True)
                        status.update(stage="sd35_standard_quality", scheme=scheme, message=config["wm_message"])
                        save(summary, status)
                        print(f"Quality-only evaluation: {scheme}/{config['wm_message']}", flush=True)
                        with (out.parent / "run.log").open("a") as stream:
                            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
                        existing = json.loads(out.read_text())
                    status["per_message"].append({"scheme": scheme, "message": config["wm_message"],
                                                   "fid_ratio": existing["quality"]["fid_ratio"], "source": str(out)})
                    save(summary, status)
                    refresh()
            refresh(final=True)
            status["state"] = "complete"
        except Exception as exc:
            status.update(state="failed", error=str(exc))
            raise
        finally:
            save(summary, status)


if __name__ == "__main__":
    main()
