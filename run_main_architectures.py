"""Generate the main 5-bit orthogonal / 32-bit hypercube grid and matched encoding ablation."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

from run_table_experiments import Job, flow_unet_base, sd35_base, classic_mlp_base, set_arg, quote_cmd, has_completed_result


MESSAGES_5 = ["00000", "00111", "01010", "10101", "11001"]
MESSAGES_32 = ["01100011100000010111101001111110", "01000100001011101011011000110111",
               "11100000100111110110101001100101", "01110000101110011010011100101010",
               "00110101011110010100111101110101"]
SCHEMES = {"orthogonal5": ("orthogonal", MESSAGES_5), "hypercube32": ("hypercube", MESSAGES_32),
           "hypercube5": ("hypercube", MESSAGES_5)}


def pools(values):
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError("Pools must be unique DATASET=PATH entries")
        result[name] = path
    return result


def build_jobs(root, families, schemes, audio_pools=None, stable_pools=None):
    jobs = []
    for scheme in schemes:
        mode, messages = SCHEMES[scheme]
        table = "tab:encoding" if scheme == "hypercube5" else "tab:main-architectures"
        if "mlp" in families:
            cmd = classic_mlp_base(str(root))
            set_arg(cmd, "--output_dir", root / "main_architectures" / "mlp" / scheme)
            set_arg(cmd, "--steps", 5000)
            set_arg(cmd, "--wm_bits", len(messages[0]))
            set_arg(cmd, "--wm_K", 32)
            set_arg(cmd, "--wm_messages", ",".join(messages))
            set_arg(cmd, "--codebook_mode", mode)
            jobs.append(Job(f"main_{scheme}_mlp_mnist", table, cmd,
                            "Scratch training: five messages, 5,000 updates per model; legacy MNIST feature-space quality metric."))
        for message in messages:
            if "unet" in families:
                for dataset in ("mnist", "cifar10", "celeba64"):
                    cmd = flow_unet_base(str(root), dataset, message, subdir=f"main_architectures/unet/{scheme}")
                    set_arg(cmd, "--wm_K", 32)
                    set_arg(cmd, "--codebook_mode", mode)
                    jobs.append(Job(f"main_{scheme}_unet_{dataset}_{message}", table, cmd))
            if "sd35" in families:
                cmd = sd35_base(str(root), message, "500")
                set_arg(cmd, "--output_dir", root / "main_architectures" / "sd35" / scheme / message)
                set_arg(cmd, "--wm_K", 32)
                set_arg(cmd, "--codebook_mode", mode)
                jobs.append(Job(f"main_{scheme}_sd35_{message}", table, cmd,
                                "DiT uses 256 queries; do not treat differing query budgets as a controlled architecture comparison."))
            if "audio" in families:
                audio_jobs = [(dataset, path, architecture) for dataset, path in (audio_pools or {}).items()
                              for architecture in ("mlp", "transformer")]
                audio_jobs.extend((dataset, path, "stable") for dataset, path in (stable_pools or {}).items())
                for dataset, path, architecture in audio_jobs:
                    cmd = ["python", "watermark_audio_flow.py", "--latents", path, "--dataset", dataset,
                           "--architecture", architecture, "--wm_message", message, "--codebook_mode", mode,
                           "--wm_K", "32", "--steps", "1000", "--n_queries", "4096",
                           "--n_detect_trials", "20", "--n_fid_samples", "500", "--n_sample_steps", "100",
                           "--output_dir", str(root / "main_architectures" / "audio" / architecture / scheme),
                           "--base_dir", str(root / "audio_bases")]
                    jobs.append(Job(f"main_{scheme}_audio_{architecture}_{dataset}_{message}", table, cmd,
                                    "FD-CLAP measures audio quality; do not label it image FID."))
    return jobs


def complete(command):
    if not has_completed_result(command):
        return False
    if command[1] == "flow_watermark_mnist_mlp.py":
        root = Path(command[command.index("--output_dir") + 1])
        result = json.loads((root / "results.json").read_text())
        return (result["config"].get("wm_bits") == int(command[command.index("--wm_bits") + 1])
                and result["config"].get("wm_messages") == command[command.index("--wm_messages") + 1].split(","))
    if command[1] != "watermark_audio_flow.py":
        return True
    root = Path(command[command.index("--output_dir") + 1])
    dataset = command[command.index("--dataset") + 1]
    message = command[command.index("--wm_message") + 1]
    result = json.loads((root / dataset / message / "results.json").read_text())
    return (result["metrics"].get("quality_complete") is True
            and all(str(result["config"].get(flag[2:])) == command[command.index(flag) + 1]
                    for flag in ("--architecture", "--latents", "--n_detect_trials", "--n_fid_samples", "--n_sample_steps")))


def compatible_legacy_result(command, path):
    filename = "sweep_results.json" if command[1] == "watermark_sd35.py" else "results.json"
    try:
        payload = json.loads((path / filename).read_text())
    except (OSError, ValueError):
        return False
    config = payload.get("config", {})
    metrics = payload.get("metrics") or payload.get("results")
    if not metrics:
        return False
    message = config.get("wm_message", "")
    bits = config.get("wm_bits", len(message))
    mode = config.get("codebook_mode", "auto")
    if mode == "auto" and 0 < bits <= 5 and config.get("wm_K") == 32:
        mode = "orthogonal"
    config = config | {"codebook_mode": mode}
    if "wm_messages" in config:
        config["wm_messages"] = ",".join(config["wm_messages"])
    ignored = {"output_dir", "skip_plots"}
    minimum = {"n_detect_trials", "n_detect_seeds"}
    index = 2
    while index < len(command):
        key = command[index][2:]
        if key == "skip_plots":
            index += 1
            continue
        value = command[index + 1]
        index += 2
        if key in ignored:
            continue
        saved = config.get(key)
        if key in minimum:
            if saved is None or int(saved) < int(value):
                return False
        elif isinstance(saved, (int, float)) and not isinstance(saved, bool):
            try:
                if saved != float(value):
                    return False
            except ValueError:
                return False
        elif saved is None or str(saved) != value:
            return False
    rows = payload.get("results", [payload.get("metrics", {})])
    return any(row.get("fid_ratio") is not None and
               ("wm_acc" in row or "detection_accuracy_wm" in row) and
               (command[1] != "watermark_sd35.py" or str(row.get("steps")) == config["steps"])
               for row in rows)


def reuse_legacy(job, root):
    command = job.command
    script = command[1]
    if script == "watermark_hf_flow_unet.py":
        dataset = command[command.index("--dataset") + 1]
        message = command[command.index("--wm_message") + 1]
        candidates = [root / folder / dataset / message
                      for folder in ("hf_flow_unet", "flow_unet_payload_hypercube")]
    elif script == "watermark_sd35.py":
        message = command[command.index("--wm_message") + 1]
        candidates = [root / f"sd35_msg_{message}"]
    elif script == "flow_watermark_mnist_mlp.py":
        candidates = [root / "classic_mlp" / "mnist"]
    else:
        return None
    for path in candidates:
        if compatible_legacy_result(command, path):
            output = path.parent.parent if script == "watermark_hf_flow_unet.py" else path
            set_arg(command, "--output_dir", output)
            job.note += f" Reused measured results from {path}; retain the saved evaluation trial count."
            return str(path)
    return None


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out_root", type=Path, default=Path("table_runs"))
    parser.add_argument("--manifest", type=Path, default=Path("table_runs/manifest_main_architectures.json"))
    parser.add_argument("--families", nargs="+", choices=["mlp", "unet", "sd35", "audio"], default=["mlp", "unet", "sd35"])
    parser.add_argument("--schemes", nargs="+", choices=list(SCHEMES), default=list(SCHEMES))
    parser.add_argument("--audio-pools", nargs="*", default=[], metavar="DATASET=PATH")
    parser.add_argument("--stable-pools", nargs="*", default=[], metavar="DATASET=PATH")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--reuse_legacy", action="store_true", help="Reuse compatible results in older experiment directories")
    parser.add_argument("--continue_on_error", action="store_true", help="Attempt remaining jobs if one fails; exit nonzero afterward")
    args = parser.parse_args()
    try:
        audio_pools, stable_pools = pools(args.audio_pools), pools(args.stable_pools)
    except ValueError as exc:
        parser.error(str(exc))
    if "audio" in args.families and not (audio_pools or stable_pools):
        parser.error("Audio jobs require --audio-pools or --stable-pools; prepare the latent caches first")
    jobs = build_jobs(args.out_root, set(args.families), list(dict.fromkeys(args.schemes)), audio_pools, stable_pools)
    for job in jobs:
        job.command[0] = sys.executable
    reused = {job.name: path for job in jobs if args.reuse_legacy and (path := reuse_legacy(job, args.out_root))}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps([asdict(job) | {"command_str": quote_cmd(job.command),
                                                     "reuse_source": reused.get(job.name)} for job in jobs], indent=2) + "\n")
    print(f"Prepared {len(jobs)} jobs. Manifest: {args.manifest}")
    print(f"Reused {len(reused)} compatible legacy results", flush=True)
    failures = []
    for job in jobs:
        print(f"\n# {job.name}\n{quote_cmd(job.command)}")
        if args.run:
            if job.name in reused:
                print(f"Reusing {reused[job.name]}", flush=True)
                continue
            if args.skip_existing and complete(job.command):
                print("Skipping completed result")
                continue
            result = subprocess.run(job.command)
            if result.returncode:
                failures.append(job.name)
                print(f"FAILED: {job.name} (exit {result.returncode})", flush=True)
                if not args.continue_on_error:
                    raise SystemExit(result.returncode)
    if failures:
        print(f"Failed jobs: {', '.join(failures)}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
