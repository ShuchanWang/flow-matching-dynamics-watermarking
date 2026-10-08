"""Checkpoint-only CIFAR-10 RF-UNet watermark modification study.

Each attack starts from the same saved watermarked state. Pruning and
quantization affect only parameters saved in adapter_final.pt; quantization
is simulated quantize/dequantize, not a compressed inference backend.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import watermark_hf_flow_unet as wm


MESSAGES = ("00000", "00111", "01010", "10101", "11001")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_root", type=Path, default=Path("table_runs/hf_flow_unet/cifar10"))
    parser.add_argument("--messages", nargs="+", default=["10101"],
                        help="Saved message directories, or 'all' for the five stage-1 messages.")
    parser.add_argument("--output_dir", type=Path, default=Path("table_runs/robustness_cifar10"))
    parser.add_argument("--ft_steps", type=int, nargs="*", default=[100],
                        help="Clean flow-matching continuation steps; use an empty list to omit.")
    parser.add_argument("--ft_lr", type=float, default=1e-5)
    parser.add_argument("--prune", type=float, nargs="*", default=[0.25],
                        help="Global magnitude-pruned fractions of saved trainable weights.")
    parser.add_argument("--quant_bits", type=int, nargs="*", default=[8],
                        help="Signed symmetric per-tensor fake-quantization bit widths.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fid_batch_size", type=int, default=4)
    parser.add_argument("--n_queries", type=int, default=4096)
    parser.add_argument("--n_detect_trials", type=int, default=20)
    parser.add_argument("--n_fid_samples", type=int, default=500)
    parser.add_argument("--n_sample_steps", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def attack_weights(model, names, kind, value):
    params = dict(model.named_parameters())
    selected = [params[name] for name in names if name in params and params[name].is_floating_point()]
    if not selected:
        raise RuntimeError("The saved adapter contains no matching floating-point parameters.")
    with torch.no_grad():
        if kind == "prune":
            count = sum(p.numel() for p in selected)
            n_remove = round(count * value)
            if n_remove == 0:
                return 0
            magnitudes = torch.cat([p.detach().abs().reshape(-1) for p in selected])
            threshold = torch.kthvalue(magnitudes, n_remove).values
            removed = 0
            for p in selected:
                mask = p.abs() <= threshold
                removed += int(mask.sum().item())
                p.masked_fill_(mask, 0)
            return removed / count
        if kind == "quant":
            qmax = 2 ** (value - 1) - 1
            for p in selected:
                max_abs = p.abs().max()
                if max_abs > 0:
                    scale = max_abs / qmax
                    p.copy_((p / scale).round().clamp(-qmax, qmax) * scale)
            return value
    raise ValueError(kind)


def conditions(cli):
    yield "baseline", None, 0
    for steps in cli.ft_steps:
        if steps <= 0:
            raise ValueError("Fine-tuning steps must be positive.")
        yield f"clean_ft_{steps}", "ft", steps
    for fraction in cli.prune:
        if not 0 < fraction < 1:
            raise ValueError("Pruning fractions must be between 0 and 1.")
        yield f"prune_{fraction:g}", "prune", fraction
    for bits in cli.quant_bits:
        if not 2 <= bits <= 16:
            raise ValueError("Quantization bits must be between 2 and 16.")
        yield f"quant_{bits}bit", "quant", bits


def main():
    cli = parse_args()
    names = MESSAGES if cli.messages == ["all"] else cli.messages
    if cli.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(cli.device)
    cfg = wm.DEFAULT_CONFIGS["cifar10"]
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    base_model = None
    checkpoint = None

    for message in names:
        run_dir = cli.runs_root / message
        for filename in ("results.json", "watermark_key.pt", "adapter_final.pt"):
            if not (run_dir / filename).is_file():
                raise FileNotFoundError(run_dir / filename)
        source = json.loads((run_dir / "results.json").read_text())
        saved = source["config"]
        if saved["dataset"] != "cifar10" or saved["wm_message"] != message:
            raise ValueError(f"Mismatched run configuration in {run_dir}")
        args = SimpleNamespace(**saved)
        args.codebook_mode = getattr(args, "codebook_mode", "auto")
        args.detect_distribution = "train_interp"
        args.batch_size = cli.batch_size
        args.fid_batch_size = cli.fid_batch_size
        args.n_queries = cli.n_queries
        args.n_detect_trials = cli.n_detect_trials
        args.n_fid_samples = cli.n_fid_samples
        args.n_sample_steps = cli.n_sample_steps
        args.num_workers = cli.num_workers
        args.grad_accum = 1
        args.lr = cli.ft_lr
        args.post_ft_steps = 0
        if checkpoint is None:
            checkpoint = saved.get("checkpoint") or cfg.filename
            checkpoint = wm.download_checkpoint(saved["repo_id"], checkpoint)
            base_model = wm.build_model(cfg, device)
            wm.load_state_dict_flexible(base_model, checkpoint, device)
            base_model.eval()
            for p in base_model.parameters():
                p.requires_grad_(False)
        model = wm.build_model(cfg, device)
        wm.load_state_dict_flexible(model, checkpoint, device)
        wm.add_lora(model, args.lora_rank, args.lora_alpha, args.lora_targets)
        model.to(device)
        wm.load_adapter(model, run_dir / "adapter_final.pt", device)
        wm.freeze_non_lora(model, args.train_extra)
        original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        saved_weights = torch.load(run_dir / "adapter_final.pt", map_location="cpu")
        attack_names = tuple(saved_weights)
        key = torch.load(run_dir / "watermark_key.pt", map_location="cpu")
        if key["message"] != message:
            raise ValueError(f"Key message mismatch in {run_dir}")
        P, codes = key["P"].to(device), key["codes"].to(device)
        train_loader, real_images = wm.load_real_data(cfg, args, device)
        if train_loader is None:
            raise RuntimeError("CIFAR-10 training data are required for on-path detection and clean fine-tuning.")

        for label, kind, severity in conditions(cli):
            result_path = cli.output_dir / message / f"{label}.json"
            if cli.skip_existing and result_path.exists():
                rows.append(json.loads(result_path.read_text()))
                print(f"Skipping {message}/{label}: saved result exists", flush=True)
                continue
            model.load_state_dict(original, strict=True)
            model.eval()
            applied = None
            if kind == "ft":
                args.post_ft_steps = severity
                wm.seed_everything(args.seed + 1000)
                history = wm.clean_finetune(model, train_loader, cfg, args, device)
                applied = history[-1]["clean_finetune_mse"]
                args.post_ft_steps = 0
            elif kind in ("prune", "quant"):
                applied = attack_weights(model, attack_names, kind, severity)
            model.eval()
            wm.seed_everything(args.seed + 2000)
            metrics = wm.evaluate(model, base_model, cfg, P, codes, message, args,
                                  device, real_images, train_loader)
            row = {
                "message": message, "condition": label, "attack": kind or "none",
                "severity": severity, "applied": applied,
                "attack_scope": "saved trainable parameters" if kind in ("prune", "quant") else "clean flow-matching continuation",
                "finetune_lr": cli.ft_lr if kind == "ft" else None,
                "n_queries": args.n_queries, "n_detect_trials": args.n_detect_trials,
                "n_fid_samples": args.n_fid_samples, "fid_feature": args.fid_feature,
                "fid_reference": "CIFAR-10 test split", "detect_distribution": args.detect_distribution,
                **metrics,
            }
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(row, indent=2) + "\n")
            rows.append(row)
            print(f"{message}/{label}: recovery={metrics['wm_acc']:.1f}% "
                  f"clean_fp={metrics['clean_fp']:.1f}% "
                  f"FID ratio={metrics.get('fid_ratio', float('nan')):.3f}", flush=True)

        del model, original, saved_weights, train_loader, real_images
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fields = sorted({key for row in rows for key in row})
    with (cli.output_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {cli.output_dir / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
