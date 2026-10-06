"""Generate or run the remote experiment grid for the paper tables.

This is intentionally a lightweight orchestrator. The heavy work still lives
in the model-specific experiment scripts:

  - watermark_sd35.py
  - watermark_hf_flow_unet.py
  - flow_watermark_mnist_mlp.py
  - flow_watermark_unet_lora.py

By default this script prints commands and writes a manifest. Use --run to
execute them directly on a GPU machine.
"""

from __future__ import annotations

import argparse
import json
import random
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


SD35_MESSAGES_5BIT = ["00000", "00111", "01010", "10101", "11001"]
SD35_CANONICAL_MESSAGE = "10101"
FLOW_UNET_MESSAGES_5BIT = ["00000", "00111", "01010", "10101", "11001"]
CLASSIC_MESSAGES_5BIT = ["00000", "00111", "01010", "10101", "11001"]
PAYLOAD_MESSAGES = {
    1: ["0", "1"],
    3: ["000", "011", "101", "110", "111"],
    5: ["00000", "00111", "01010", "10101", "11001"],
    8: ["00000000", "00110101", "01011010", "10100101", "11110000"],
    12: ["000000000000", "001101011010", "010110101101", "101001010010", "111100001111"],
}
PAYLOAD_STRESS_BITS = (4, 8, 16, 32)


def payload_stress_messages(bits: int) -> list[str]:
    indices = random.Random(2605 + bits).sample(range(1 << bits), 5)
    return [format(index, f"0{bits}b") for index in indices]


@dataclass
class Job:
    name: str
    table: str
    command: list[str]
    note: str = ""


def py(script: str, *args: object) -> list[str]:
    return ["python", script, *[str(a) for a in args]]


def add_arg(cmd: list[str], name: str, value: object | None = None) -> list[str]:
    cmd.append(name)
    if value is not None:
        cmd.append(str(value))
    return cmd


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def has_completed_result(cmd: list[str]) -> bool:
    if "--output_dir" not in cmd:
        return False
    out_dir = Path(cmd[cmd.index("--output_dir") + 1])
    if "--dataset" in cmd and "--wm_message" in cmd:
        dataset = cmd[cmd.index("--dataset") + 1]
        message = cmd[cmd.index("--wm_message") + 1]
        out_dir = out_dir / dataset / message
    result_name = "sweep_results.json" if cmd[1] == "watermark_sd35.py" else "results.json"
    result_path = out_dir / result_name
    try:
        with open(result_path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    config = payload.get("config", {})
    for flag in ("--dataset", "--wm_message", "--codebook_mode", "--wm_K", "--n_queries", "--steps"):
        if flag in cmd and str(config.get(flag[2:])) != cmd[cmd.index(flag) + 1]:
            return False
    return bool(payload.get("metrics") or payload.get("results"))


def sd35_base(out_root: str, message: str = "10101", steps: str = "500") -> list[str]:
    cmd = py(
        "watermark_sd35.py",
        "--output_dir", f"{out_root}/sd35_msg_{message}",
        "--wm_message", message,
        "--steps", steps,
        "--n_detect_seeds", 20,
        "--n_detect_queries", 256,
        "--n_fid_samples", 30,
        "--n_fid_real", 30,
        "--lora_rank", 16,
        "--lora_targets", "mlp",
        "--wm_eps", 1.5,
        "--wm_lambda", 0.5,
    )
    return cmd


def flow_unet_base(
    out_root: str,
    dataset: str,
    message: str = "10101",
    *,
    subdir: str = "hf_flow_unet",
) -> list[str]:
    cmd = py(
        "watermark_hf_flow_unet.py",
        "--output_dir", f"{out_root}/{subdir}",
        "--dataset", dataset,
        "--wm_message", message,
        "--objective", "full",
        "--detect_distribution", "train_interp",
        "--steps", 1000,
        "--lora_rank", 16,
        "--lora_alpha", 16,
        "--lora_targets", "both",
        "--train_extra", "time_embed,out.",
        "--batch_size", 64,
        "--lr", 1e-4,
        "--wm_eps", 1.5,
        "--wm_lambda", 1.0,
        "--wm_proj_weight", 10.0,
        "--wm_tanh_scale", 2.0,
        "--n_detect_trials", 20,
        "--n_queries", 4096,
        "--n_fid_samples", 500,
        "--n_sample_steps", 100,
    )
    if dataset == "celeba64":
        cmd.extend([
            "--data_source", "base_samples",
            "--base_sample_pool", f"{out_root}/cache/celeba64_base_samples.pt",
            "--fid_reference", "base_samples",
            "--fid_feature", "inception",
            "--fid_batch_size", "4",
        ])
    return cmd


def classic_mlp_base(out_root: str) -> list[str]:
    return py(
        "flow_watermark_mnist_mlp.py",
        "--output_dir", f"{out_root}/classic_mlp/mnist",
        "--steps", 10000,
        "--wm_bits", 5,
        "--wm_messages", ",".join(CLASSIC_MESSAGES_5BIT),
        "--n_queries", 4096,
        "--skip_plots",
    )


def classic_unet_base(out_root: str, dataset: str) -> list[str]:
    return py(
        "flow_watermark_unet_lora.py",
        "--dataset", dataset,
        "--output_dir", f"{out_root}/classic_unet/{dataset}",
        "--checkpoint_dir", f"{out_root}/checkpoints/classic_unet_{dataset}",
        "--target_steps", 5000,
        "--lora_steps", 500,
        "--wm_bits", 5,
        "--wm_messages", ",".join(CLASSIC_MESSAGES_5BIT),
        "--n_queries", 4096,
        "--skip_plots",
    )


def set_arg(cmd: list[str], name: str, value: object) -> list[str]:
    if name in cmd:
        cmd[cmd.index(name) + 1] = str(value)
    else:
        cmd.extend([name, str(value)])
    return cmd


def canonical_message(bits: int) -> str:
    messages = PAYLOAD_MESSAGES.get(bits)
    if not messages:
        return "1" * bits
    return messages[min(3, len(messages) - 1)]


def build_jobs(selected: set[str], out_root: str, capacity_bits: tuple[int, ...] = PAYLOAD_STRESS_BITS) -> list[Job]:
    jobs: list[Job] = []

    if "sd35-main" in selected:
        for msg in SD35_MESSAGES_5BIT:
            jobs.append(Job(
                name=f"sd35_main_5bit_{msg}",
                table="tab:main, tab:app-main",
                command=sd35_base(out_root, msg, "500"),
                note="Primary SD3.5 5-bit row; aggregate across messages with collect_results.py.",
            ))

    if "sd35-cross" in selected:
        for msg in SD35_MESSAGES_5BIT:
            jobs.append(Job(
                name=f"sd35_cross_{msg}",
                table="tab:app-cross",
                command=sd35_base(f"{out_root}/cross", msg, "500"),
                note="Run all listed messages, then compare detector scores across codewords.",
            ))

    if "sd35-query" in selected:
        # This retrains once and evaluates one query budget per run. It is more
        # expensive than evaluating all budgets from one checkpoint, but it keeps
        # the existing SD3 script untouched and reproducible.
        for n in [16, 32, 64, 128, 256, 512, 1024]:
            for msg in SD35_MESSAGES_5BIT:
                cmd = sd35_base(f"{out_root}/query", msg, "500")
                cmd[cmd.index("--output_dir") + 1] = f"{out_root}/query/sd35_N_{n}_msg_{msg}"
                cmd[cmd.index("--n_detect_queries") + 1] = str(n)
                jobs.append(Job(
                    name=f"sd35_query_{n}_{msg}",
                    table="tab:app-ablation(c)",
                    command=cmd,
                    note="Query-budget ablation; report mean/std across messages.",
                ))

    if "sd35-epsilon" in selected:
        for eps in [0.1, 0.5, 1.0, 1.5, 3.0, 5.0]:
            for msg in SD35_MESSAGES_5BIT:
                cmd = sd35_base(f"{out_root}/epsilon", msg, "500")
                cmd[cmd.index("--output_dir") + 1] = f"{out_root}/epsilon/sd35_eps_{eps}_msg_{msg}"
                cmd[cmd.index("--wm_eps") + 1] = str(eps)
                jobs.append(Job(
                    name=f"sd35_epsilon_{eps}_{msg}",
                    table="tab:app-ablation(d)",
                    command=cmd,
                    note="Perturbation strength ablation; report mean/std across messages.",
                ))

    if "sd35-rank" in selected:
        for rank in [4, 8, 16, 32, 64]:
            for msg in SD35_MESSAGES_5BIT:
                cmd = sd35_base(f"{out_root}/rank", msg, "500")
                cmd[cmd.index("--output_dir") + 1] = f"{out_root}/rank/sd35_rank_{rank}_msg_{msg}"
                cmd[cmd.index("--lora_rank") + 1] = str(rank)
                jobs.append(Job(
                    name=f"sd35_rank_{rank}_{msg}",
                    table="tab:app-ablation(e)",
                    command=cmd,
                    note="LoRA-rank ablation; report mean/std across messages.",
                ))

    if "sd35-steps" in selected:
        for msg in SD35_MESSAGES_5BIT:
            cmd = sd35_base(f"{out_root}/steps", msg, "50,100,300,500,1000,2000")
            cmd[cmd.index("--output_dir") + 1] = f"{out_root}/steps/sd35_steps_msg_{msg}"
            jobs.append(Job(
                name=f"sd35_steps_{msg}",
                table="tab:app-ablation(f), tab:app-finetune",
                command=cmd,
                note="Cumulative run; aggregate each target step across messages.",
            ))

    if "flow-unet-main" in selected:
        for dataset in ["mnist", "cifar10", "celeba64"]:
            for msg in FLOW_UNET_MESSAGES_5BIT:
                jobs.append(Job(
                    name=f"flow_unet_{dataset}_{msg}",
                    table="tab:app-main",
                    command=flow_unet_base(out_root, dataset, msg),
                    note=(
                        "Public pretrained flow-matching UNet. Aggregate across "
                        "messages with collect_results.py."
                    ),
                ))

    if "classic-main" in selected:
        jobs.append(Job(
            name="classic_mlp_mnist",
            table="tab:main, tab:app-main",
            command=classic_mlp_base(out_root),
            note="Original MNIST MLP experiment rerun with the same explicit five-message protocol.",
        ))
        for dataset in ["mnist", "cifar10"]:
            jobs.append(Job(
                name=f"classic_unet_{dataset}",
                table="tab:main, tab:app-main",
                command=classic_unet_base(out_root, dataset),
                note="Original small UNet LoRA experiment rerun with the same explicit five-message protocol.",
            ))

    if "flow-unet-query" in selected:
        cmd = flow_unet_base(out_root, "cifar10", subdir="flow_unet_query")
        cmd.extend(["--sweep", "queries"])
        jobs.append(Job(
            name="flow_unet_cifar_query",
            table="tab:app-ablation(c)",
            command=cmd,
            note="HF flow-UNet query-budget sweep.",
        ))

    if "flow-unet-payload" in selected:
        for dataset in ["cifar10", "celeba64"]:
            for bits, messages in PAYLOAD_MESSAGES.items():
                for msg in messages:
                    cmd = flow_unet_base(out_root, dataset, msg, subdir="flow_unet_payload")
                    jobs.append(Job(
                        name=f"flow_unet_payload_{dataset}_{bits}bit_{msg}",
                        table="tab:app-ablation(a)",
                        command=cmd,
                        note="Payload-capacity ablation; aggregate by bit length and dataset.",
                    ))

    if selected & {"flow-unet-payload-capacity", "flow-unet-payload-stress"}:
        for bits in capacity_bits:
            for msg in payload_stress_messages(bits):
                cmd = flow_unet_base(
                    out_root,
                    "cifar10",
                    msg,
                    subdir="flow_unet_payload_hypercube",
                )
                set_arg(cmd, "--wm_K", 32)
                set_arg(cmd, "--codebook_mode", "hypercube")
                set_arg(cmd, "--n_queries", 4096)
                set_arg(cmd, "--n_detect_trials", 40)
                jobs.append(Job(
                    name=f"flow_unet_payload_capacity_cifar10_{bits}bit_{msg}",
                    table="tab:app-payload",
                    command=cmd,
                    note="Fixed K=32 and N=4096; vary payload length over the implicit hypercube codebook.",
                ))

    if "flow-unet-epsilon" in selected:
        for eps in [0.1, 0.5, 1.0, 1.5, 3.0, 5.0]:
            for msg in FLOW_UNET_MESSAGES_5BIT:
                cmd = flow_unet_base(out_root, "cifar10", msg, subdir=f"flow_unet_epsilon/eps_{eps}")
                set_arg(cmd, "--wm_eps", eps)
                jobs.append(Job(
                    name=f"flow_unet_epsilon_{eps}_{msg}",
                    table="tab:app-ablation(d)",
                    command=cmd,
                    note="HF flow-UNet perturbation-strength ablation on CIFAR-10.",
                ))

    if "flow-unet-rank" in selected:
        for rank in [4, 8, 16, 32, 64]:
            for msg in FLOW_UNET_MESSAGES_5BIT:
                cmd = flow_unet_base(out_root, "cifar10", msg, subdir=f"flow_unet_rank/rank_{rank}")
                set_arg(cmd, "--lora_rank", rank)
                set_arg(cmd, "--lora_alpha", rank)
                jobs.append(Job(
                    name=f"flow_unet_rank_{rank}_{msg}",
                    table="tab:app-ablation(e)",
                    command=cmd,
                    note="HF flow-UNet LoRA-rank ablation on CIFAR-10.",
                ))

    if "flow-unet-finetune" in selected:
        for post_steps in [0, 100, 500, 1000]:
            for msg in FLOW_UNET_MESSAGES_5BIT:
                cmd = flow_unet_base(out_root, "cifar10", msg, subdir=f"flow_unet_finetune/post_{post_steps}")
                set_arg(cmd, "--post_ft_steps", post_steps)
                jobs.append(Job(
                    name=f"flow_unet_finetune_{post_steps}_{msg}",
                    table="tab:app-finetune",
                    command=cmd,
                    note="Robustness to clean continued flow-matching fine-tuning.",
                ))

    if "sd35-payload" in selected:
        for bits, messages in PAYLOAD_MESSAGES.items():
            for msg in messages:
                cmd = sd35_base(f"{out_root}/sd35_payload/{bits}bit", msg, "500")
                jobs.append(Job(
                    name=f"sd35_payload_{bits}bit_{msg}",
                    table="tab:app-ablation(a)",
                    command=cmd,
                    note="SD3.5 payload-capacity ablation.",
                ))

    return jobs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", default="table_runs")
    parser.add_argument("--manifest", default="table_runs/manifest.json")
    parser.add_argument("--run", action="store_true", help="Execute jobs instead of only printing commands.")
    parser.add_argument("--only", default="all", help="Comma-separated job groups, or all.")
    parser.add_argument("--capacity_bits", default="4,8,16,32", help="Comma-separated bit lengths for the flow-UNet payload-capacity group.")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--slurm", default=None, help="Write one SLURM-style shell script with all commands.")
    return parser.parse_args()


def main():
    args = parse_args()
    all_groups = {
        "sd35-main",
        "sd35-cross",
        "sd35-query",
        "sd35-epsilon",
        "sd35-rank",
        "sd35-steps",
        "flow-unet-main",
        "classic-main",
        "flow-unet-query",
        "flow-unet-payload",
        "flow-unet-payload-capacity",
        "flow-unet-payload-stress",
        "flow-unet-epsilon",
        "flow-unet-rank",
        "flow-unet-finetune",
        "sd35-payload",
    }
    selected = all_groups if args.only == "all" else {x.strip() for x in args.only.split(",") if x.strip()}
    unknown = selected - all_groups
    if unknown:
        raise SystemExit(f"Unknown group(s): {sorted(unknown)}")

    try:
        capacity_bits = tuple(int(value.strip()) for value in args.capacity_bits.split(",") if value.strip())
    except ValueError as exc:
        raise SystemExit("--capacity_bits must be a comma-separated list of integers") from exc
    if not capacity_bits or len(set(capacity_bits)) != len(capacity_bits) or set(capacity_bits) - set(PAYLOAD_STRESS_BITS):
        raise SystemExit(f"--capacity_bits must contain distinct values from {PAYLOAD_STRESS_BITS}")

    jobs = build_jobs(selected, args.out_root, capacity_bits)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump([asdict(job) | {"command_str": quote_cmd(job.command)} for job in jobs], f, indent=2)

    print(f"Prepared {len(jobs)} jobs. Manifest: {manifest_path}")
    for job in jobs:
        print(f"\n# {job.name} [{job.table}]")
        if job.note:
            print(f"# {job.note}")
        print(quote_cmd(job.command))

    if args.slurm:
        slurm_path = Path(args.slurm)
        slurm_path.parent.mkdir(parents=True, exist_ok=True)
        with open(slurm_path, "w") as f:
            f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
            for job in jobs:
                f.write(f"echo '=== {job.name} ==='\n")
                f.write(quote_cmd(job.command) + "\n\n")
        print(f"\nWrote shell script: {slurm_path}")

    if args.run:
        for job in jobs:
            if args.skip_existing and has_completed_result(job.command):
                print(f"Skipping existing job: {job.name}")
                continue
            print(f"\nRunning {job.name}")
            subprocess.run(job.command, check=True)


if __name__ == "__main__":
    main()
