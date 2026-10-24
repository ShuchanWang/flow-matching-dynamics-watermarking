"""Apply the same original-key SD3.5 repair schedule to each selected 32-bit message."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from run_main_architectures import MESSAGES_32


def commands(source, output, calibration):
    common = [sys.executable, "-u", "diagnose_sd35_hypercube.py", "--run-dir", str(source),
              "--queries", "256", "--batch-size", "32", "--repair-proj-weight", "10",
              "--repair-bit-weight", "1", "--repair-batch-size", "2", "--repair-grad-accum", "2"]
    warmup = common + ["--output", str(output / "warmup.json"), "--repair-steps", "50",
                       "--trials", "1", "--seed", "20261010"]
    compensated = common + ["--output", str(output / "compensated.json"),
                            "--adapter-path", str(output / "warmup"), "--repair-steps", "100",
                            "--repair-clean-calibration", str(calibration), "--trials", "1", "--seed", "20261012"]
    final = common + ["--output", str(output / "final.json"), "--adapter-path", str(output / "compensated"),
                      "--repair-steps", "50", "--repair-epsilon", "3",
                      "--repair-clean-calibration", str(calibration), "--trials", "20",
                      "--seed", "20261014", "--quality-samples", "30"]
    return [warmup, compensated, final]


def stage_complete(command):
    path = Path(command[command.index("--output") + 1])
    try:
        result = json.loads(path.read_text())
        if Path(result["source"]).resolve() != Path(command[command.index("--run-dir") + 1]).resolve():
            return False
        expected_adapter = (Path(command[command.index("--adapter-path") + 1]) if "--adapter-path" in command
                            else Path(command[command.index("--run-dir") + 1]) / "final")
        if Path(result["input_adapter"]).resolve() != expected_adapter.resolve():
            return False
        if result["repair"].get("repair_lr") != 5e-5:
            return False
        for flag in ("--queries", "--trials", "--seed"):
            if result[flag[2:]] != int(command[command.index(flag) + 1]):
                return False
        for flag in ("--repair-steps", "--repair-proj-weight", "--repair-bit-weight", "--repair-batch-size", "--repair-grad-accum"):
            if result["repair"][flag[2:].replace("-", "_")] != float(command[command.index(flag) + 1]):
                return False
        for flag in ("--repair-epsilon", "--repair-clean-calibration"):
            expected = command[command.index(flag) + 1] if flag in command else None
            saved = result["repair"].get(flag[2:].replace("-", "_"))
            if flag == "--repair-epsilon" and expected is not None:
                expected = float(expected)
            if saved != expected:
                return False
        if not (path.with_suffix("") / "adapter_config.json").exists():
            return False
        if "--quality-samples" in command:
            return (result.get("quality_evaluated") is True and
                    result["quality"]["n_samples"] == int(command[command.index("--quality-samples") + 1]) and
                    result["quality"].get("sampling_protocol", "legacy_forward_euler") == "legacy_forward_euler" and
                    math.isfinite(result["quality"]["fid_ratio"]))
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def quality_complete(command):
    try:
        result = json.loads(Path(command[command.index("--output") + 1]).read_text())
        return (Path(result["source"]).resolve() == Path(command[command.index("--run-dir") + 1]).resolve()
                and Path(result["input_adapter"]).resolve() == Path(command[command.index("--adapter-path") + 1]).resolve()
                and result.get("quality_evaluated") is True
                and result["quality"]["sampling_protocol"] == "diffusers_sd3"
                and result["quality"]["n_samples"] == 30 and math.isfinite(result["quality"]["fid_ratio"]))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=Path("table_runs/main_architectures/sd35/hypercube32"))
    parser.add_argument("--output-root", type=Path, default=Path("table_runs/repairs/sd35/hypercube32"))
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--messages", default=",".join(MESSAGES_32))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse matching completed repair stages")
    parser.add_argument("--standard-quality-only", action="store_true",
                        help="Evaluate existing repaired adapters with the official sampling pipeline")
    parser.add_argument("--reference-adapter", type=Path,
                        help="Existing repaired adapter for the first message, if stored outside the suite")
    args = parser.parse_args()
    messages = args.messages.split(",")
    if not messages or len(set(messages)) != len(messages) or any(m not in MESSAGES_32 for m in messages):
        parser.error("Specify distinct messages from the original five-message grid")
    if args.root.resolve() == args.output_root.resolve():
        parser.error("Use a separate output root")
    if not args.standard_quality_only and (args.calibration is None or not args.calibration.is_file()):
        parser.error("Calibration JSON is missing")
    if args.standard_quality_only:
        summary = {"sampling_protocol": "diffusers_sd3", "per_message": [], "failures": []}
        quality_root = args.output_root / "standard_quality"
        for message in messages:
            adapter = (args.reference_adapter if message == MESSAGES_32[0] and args.reference_adapter is not None
                       else args.output_root / message / "final")
            output = quality_root / (message + ".json")
            cmd = [sys.executable, "-u", "diagnose_sd35_hypercube.py", "--run-dir", str(args.root / message),
                   "--adapter-path", str(adapter), "--output", str(output), "--queries", "256", "--trials", "1",
                   "--batch-size", "32", "--seed", "20261015", "--quality-samples", "30", "--quality-protocol", "standard"]
            if not args.run:
                print(subprocess.list2cmdline(cmd))
                continue
            if output.exists() and not args.skip_existing:
                parser.error(f"Quality output exists: {output}; use --skip-existing or a new root")
            quality_root.mkdir(parents=True, exist_ok=True)
            if not (args.skip_existing and quality_complete(cmd)):
                print(f"{message}: standard quality; log={output.with_suffix('.log')}", flush=True)
                with output.with_suffix(".log").open("w") as log:
                    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    summary["failures"].append({"message": message, "returncode": result.returncode})
                    (quality_root / "quality_results.json").write_text(json.dumps(summary, indent=2))
                    continue
            result = json.loads(output.read_text())
            summary["per_message"].append({"message": message, "adapter": str(adapter), "quality": result["quality"]})
            (quality_root / "quality_results.json").write_text(json.dumps(summary, indent=2))
            print(f"{message}: standard quality={result['quality']}", flush=True)
        if summary["failures"]:
            raise SystemExit(1)
        return
    summary = {"extra_updates_per_message": 200, "effective_batch_size": 4, "per_message": [], "failures": []}
    for message in messages:
        output = args.output_root / message
        jobs = commands(args.root / message, output, args.calibration)
        if not args.run:
            for cmd in jobs:
                print(subprocess.list2cmdline(cmd))
            continue
        if output.exists() and any(output.iterdir()) and not args.skip_existing:
            parser.error(f"Output already contains files: {output}; use a new output root")
        output.mkdir(parents=True, exist_ok=True)
        rebuilt = False
        for stage, cmd in zip(("warmup", "compensated", "final"), jobs):
            if args.skip_existing and not rebuilt and stage_complete(cmd):
                print(f"{message}: reuse {stage}", flush=True)
                continue
            rebuilt = True
            print(f"{message}: {stage}; log={output / (stage + '.log')}", flush=True)
            with (output / (stage + ".log")).open("w") as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                summary["failures"].append({"message": message, "stage": stage, "returncode": result.returncode})
                break
        else:
            result = json.loads((output / "final.json").read_text())
            row = {"message": message, "wm": result["wm"], "clean": result["clean"],
                   "quality": result.get("quality"), "source": str(args.root / message), "result": str(output / "final.json")}
            summary["per_message"].append(row)
            quality = row["quality"]
            if quality is None or not math.isfinite(quality["fid_ratio"]):
                summary["failures"].append({"message": message, "stage": "quality"})
            print(f"{message}: exact={row['wm']['exact_recovery']}% quality={quality}", flush=True)
        (args.output_root / "suite_results.json").write_text(json.dumps(summary, indent=2))
    if summary["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
