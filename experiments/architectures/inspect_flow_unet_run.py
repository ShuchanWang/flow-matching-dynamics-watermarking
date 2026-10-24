"""Inspect a completed HF flow UNet watermark run without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torchvision.utils import make_grid, save_image

from watermark_hf_flow_unet import (
    DEFAULT_CONFIGS,
    add_lora,
    build_model,
    detect,
    download_checkpoint,
    labels_for_batch,
    load_adapter,
    load_state_dict_flexible,
    model_velocity,
    seed_everything,
)


@torch.no_grad()
def sample_from_noise(model, noise, labels, steps):
    x = noise.clone()
    for i in range(steps):
        t = torch.full((x.shape[0],), i / steps, device=x.device)
        x = (x + model_velocity(model, x, t, labels) / steps).clamp(-5, 5)
    return x.clamp(-1, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing results.json, adapter_final.pt, watermark_key.pt")
    parser.add_argument("--n_images", type=int, default=8)
    parser.add_argument("--image_batch_size", type=int, default=4)
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--n_queries", type=int, default=4096)
    parser.add_argument("--query_batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default=None, help="Local base checkpoint; otherwise use the run's HF repository")
    options = parser.parse_args()
    if min(options.n_images, options.image_batch_size, options.n_trials, options.n_queries, options.query_batch_size) < 1:
        parser.error("Image, trial, query, and batch counts must be positive")

    run_dir = options.run_dir
    with (run_dir / "results.json").open() as handle:
        saved = json.load(handle)
    args = SimpleNamespace(**saved["config"])
    cfg = DEFAULT_CONFIGS[args.dataset]
    device = torch.device(options.device)
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

    base_rows, marked_rows = [], []
    for start in range(0, options.n_images, options.image_batch_size):
        count = min(options.image_batch_size, options.n_images - start)
        noise = torch.randn(count, *cfg.dim, device=device)
        labels = labels_for_batch(cfg, count, args, device)
        base_rows.append(sample_from_noise(base, noise, labels, args.n_sample_steps).cpu())
        marked_rows.append(sample_from_noise(marked, noise, labels, args.n_sample_steps).cpu())
    clean = torch.cat(base_rows)
    wm = torch.cat(marked_rows)
    # The top row is the base model; the bottom row is the watermarked model.
    grid = make_grid(torch.cat((clean, wm)), nrow=options.n_images, normalize=True, value_range=(-1, 1))
    image_path = run_dir / "paired_base_wm.png"
    save_image(grid, image_path)

    key = torch.load(run_dir / "watermark_key.pt", map_location="cpu", weights_only=False)
    if key["message"] != args.wm_message:
        raise ValueError("Watermark key and results.json disagree on the message")
    P, codes = key["P"].to(device), key["codes"].to(device)
    args.detect_distribution = "noise"
    args.batch_size = options.query_batch_size
    bits = tuple(map(int, args.wm_message))
    records = []
    for trial in range(options.n_trials):
        row = {"trial": trial + 1}
        for name, model in (("watermarked", marked), ("base", base)):
            decoded, scores, _ = detect(model, cfg, P, codes, args, device, n_queries=options.n_queries)
            row[name] = {"decoded": "".join(map(str, decoded)), "target_match": decoded == bits}
            if args.codebook_mode == "hypercube":
                signs = 2 * np.asarray(bits) - 1
                margins = signs * scores
                row[name]["min_signed_bit_margin"] = float(margins.min())
                row[name]["mean_signed_bit_margin"] = float(margins.mean())
                row[name]["bit_errors"] = int(np.count_nonzero(margins <= 0))
        records.append(row)
        print(f"Noise-only trial {trial + 1}/{options.n_trials}: "
              f"WM errors={row['watermarked'].get('bit_errors', 'n/a')}, "
              f"base target match={row['base']['target_match']}", flush=True)
    report = {"run_dir": str(run_dir), "seed": options.seed, "n_images": options.n_images,
              "n_queries_per_trial": options.n_queries, "detection_distribution": "noise",
              "paired_grid": str(image_path), "trials": records}
    report_path = run_dir / "inspection.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Paired image grid: {image_path}")
    print(f"Noise-only detection report: {report_path}")


if __name__ == "__main__":
    main()
