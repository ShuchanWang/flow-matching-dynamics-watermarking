"""Measure trained RF-UNet watermark strength on held-out flow paths."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from watermark_hf_flow_unet import (
    DEFAULT_CONFIGS,
    add_lora,
    build_model,
    download_checkpoint,
    hypercube_code,
    load_adapter,
    load_state_dict_flexible,
    message_index,
    model_velocity,
    sample,
    seed_everything,
)


def real_endpoint_loader(dataset: str, batch_size: int, n_queries: int):
    channels = 1 if dataset == "mnist" else 3
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * channels, (0.5,) * channels),
    ])
    if dataset == "mnist":
        data = datasets.MNIST("./data", train=False, download=True, transform=transform)
    elif dataset == "cifar10":
        data = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    else:
        return None
    if n_queries > len(data):
        raise ValueError(f"Requested {n_queries} held-out images, but only {len(data)} exist")
    return DataLoader(Subset(data, range(n_queries)), batch_size=batch_size, shuffle=False)


@torch.no_grad()
def measure_run(run_dir: Path, options, device: torch.device):
    with (run_dir / "results.json").open() as handle:
        args = SimpleNamespace(**json.load(handle)["config"])
    cfg = DEFAULT_CONFIGS[args.dataset]
    seed_everything(options.seed)
    checkpoint = options.checkpoint or download_checkpoint(args.repo_id, args.checkpoint or cfg.filename)
    base = build_model(cfg, device)
    load_state_dict_flexible(base, checkpoint, device)
    base.eval()
    marked = build_model(cfg, device)
    load_state_dict_flexible(marked, checkpoint, device)
    add_lora(marked, args.lora_rank, args.lora_alpha, args.lora_targets)
    adapter = run_dir / f"adapter_post_ft_{args.post_ft_steps}.pt" if args.post_ft_steps else run_dir / "adapter_final.pt"
    load_adapter(marked, adapter, device)
    marked.eval()
    key = torch.load(run_dir / "watermark_key.pt", map_location="cpu", weights_only=False)
    if key["message"] != args.wm_message:
        raise ValueError(f"Key does not match message in {run_dir}")
    P, codes = key["P"].to(device), key["codes"].to(device)
    bits = tuple(map(int, args.wm_message))
    codebook_mode = getattr(args, "codebook_mode", "auto")
    code = hypercube_code(bits, codes) if codebook_mode == "hypercube" else codes[message_index(bits)]
    direction = P @ code
    loader = real_endpoint_loader(args.dataset, options.batch_size, options.n_queries)
    loader_iter = iter(loader) if loader is not None else None

    base_sq = delta_sq = target_sq = 0.0
    base_proj_sq = delta_proj_sq = 0.0
    carrier_sq = carrier_delta = carrier_base = 0.0
    count = 0
    while count < options.n_queries:
        n = min(options.batch_size, options.n_queries - count)
        if loader is None:
            endpoint = sample(base, cfg, args, device, n)
            labels = None
            endpoint_type = "base_generated"
        else:
            endpoint, labels = next(loader_iter)
            endpoint = endpoint.to(device)
            labels = labels.to(device) if cfg.class_cond else None
            n = endpoint.shape[0]
            endpoint_type = "held_out_real"
        x0 = torch.randn_like(endpoint)
        t = torch.rand(n, device=device)
        x_t = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * endpoint
        v0 = model_velocity(base, x_t, t, labels).float().flatten(1)
        delta = model_velocity(marked, x_t, t, labels).float().flatten(1) - v0
        target = args.wm_eps * torch.sin(2 * math.pi * t)
        base_proj = v0 @ direction
        delta_proj = delta @ direction
        carrier = torch.sin(2 * math.pi * t)
        base_sq += v0.square().sum().item()
        delta_sq += delta.square().sum().item()
        target_sq += target.square().sum().item()
        base_proj_sq += base_proj.square().sum().item()
        delta_proj_sq += delta_proj.square().sum().item()
        carrier_sq += carrier.square().sum().item()
        carrier_delta += (carrier * delta_proj).sum().item()
        carrier_base += (carrier * base_proj).sum().item()
        count += n

    d = math.prod(cfg.dim)
    amplitude = carrier_delta / carrier_sq
    result = {
        "dataset": args.dataset,
        "bits": len(bits),
        "k": args.wm_K,
        "codebook_mode": codebook_mode,
        "message": args.wm_message,
        "endpoint_type": endpoint_type,
        "n_queries": count,
        "base_velocity_rms_l2": math.sqrt(base_sq / count),
        "model_change_rms_l2": math.sqrt(delta_sq / count),
        "model_change_to_base_rms": math.sqrt(delta_sq / base_sq),
        "target_carrier_rms_l2": math.sqrt(target_sq / count),
        "target_to_base_rms": math.sqrt(target_sq / base_sq),
        "base_keyed_coordinate_rms": math.sqrt(base_proj_sq / count),
        "model_change_keyed_coordinate_rms": math.sqrt(delta_proj_sq / count),
        "keyed_change_to_base_coordinate_rms": math.sqrt(delta_proj_sq / base_proj_sq),
        "carrier_aligned_change": carrier_delta / count,
        "carrier_aligned_base": carrier_base / count,
        "carrier_amplitude_estimate": amplitude,
        "carrier_component_to_base_velocity_rms": abs(amplitude) * math.sqrt(carrier_sq / base_sq),
        "carrier_component_to_base_keyed_rms": abs(amplitude) * math.sqrt(carrier_sq / base_proj_sq),
        "velocity_dimension": d,
    }
    path = run_dir / "signal_strength.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--n_queries", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default=None, help="Optional local base checkpoint")
    parser.add_argument("--summary_csv", type=Path, default=Path("table_runs/signal_strength_summary.csv"))
    options = parser.parse_args()
    if options.n_queries < 1 or options.batch_size < 1:
        parser.error("--n_queries and --batch_size must be positive")
    device = torch.device(options.device)
    rows = []
    for run_dir in options.run_dirs:
        result = measure_run(run_dir, options, device)
        rows.append({"run_dir": str(run_dir), **result})
        print(f"{run_dir}: change/base={result['model_change_to_base_rms']:.4f}, "
              f"carrier/base={result['target_to_base_rms']:.4f}, "
              f"aligned={result['carrier_aligned_change']:.4f}", flush=True)
    options.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with options.summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary: {options.summary_csv}")

    groups = {}
    for row in rows:
        key = (row["dataset"], row["k"], row["codebook_mode"], row["bits"], row["endpoint_type"])
        groups.setdefault(key, []).append(row)
    measures = ("model_change_to_base_rms", "carrier_component_to_base_velocity_rms",
                "carrier_component_to_base_keyed_rms", "carrier_amplitude_estimate",
                "base_velocity_rms_l2")
    grouped_rows = []
    for key, members in sorted(groups.items()):
        entry = dict(zip(("dataset", "k", "codebook_mode", "bits", "endpoint_type"), key))
        entry["n_messages"] = len(members)
        for measure in measures:
            values = [member[measure] for member in members]
            entry[f"{measure}_mean"] = statistics.mean(values)
            entry[f"{measure}_std"] = statistics.pstdev(values)
        grouped_rows.append(entry)
    grouped_path = options.summary_csv.with_name(options.summary_csv.stem + "_grouped.csv")
    with grouped_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=grouped_rows[0].keys())
        writer.writeheader()
        writer.writerows(grouped_rows)
    print(f"Grouped mean/std: {grouped_path}")


if __name__ == "__main__":
    main()
