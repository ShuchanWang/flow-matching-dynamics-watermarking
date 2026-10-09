"""Evaluate saved adapters with their original keys without changing source runs."""

import argparse
import json
import numpy as np
from pathlib import Path
from types import SimpleNamespace

import torch

import watermark_hf_flow_unet as wm


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def refine_checkpoints(source_root, output_root, device):
    """Replay the first 100 updates with the original 1,000-step LR schedule."""
    for source in sorted(source_root.iterdir()):
        if not (source / "watermark_key.pt").exists():
            continue
        destination = output_root / source.name
        destination.mkdir(parents=True, exist_ok=True)
        key = torch.load(source / "watermark_key.pt", map_location="cpu", weights_only=False)
        args = SimpleNamespace(**key["config"])
        args.codebook_mode = getattr(args, "codebook_mode", "auto")
        args.save_every = 10
        cfg = wm.DEFAULT_CONFIGS[args.dataset]
        wm.seed_everything(args.seed)
        checkpoint = wm.download_checkpoint(args.repo_id, args.checkpoint or cfg.filename)
        base = wm.build_model(cfg, device)
        wm.load_state_dict_flexible(base, checkpoint, device)
        base.eval().requires_grad_(False)
        model = wm.build_model(cfg, device)
        wm.load_state_dict_flexible(model, checkpoint, device)
        wm.add_lora(model, args.lora_rank, args.lora_alpha, args.lora_targets)
        P, codes = wm.make_codebook(int(np.prod(cfg.dim)), len(source.name), args.wm_K,
                                    device, args.codebook_mode)
        if not torch.equal(P.cpu(), key["P"]) or not torch.equal(codes.cpu(), key["codes"]):
            raise RuntimeError("Recreated RNG sequence does not match the original saved key")
        loader, _ = wm.load_real_data(cfg, args, device)
        bits = tuple(int(b) for b in source.name)
        code = wm.hypercube_code(bits, codes) if args.codebook_mode == "hypercube" else codes[wm.message_index(bits)]
        history = wm.train_lora(model, base, loader, cfg, P, code, args, device,
                                destination, stop_after=100)
        torch.save(key, destination / "watermark_key.pt")
        original = torch.load(source / "adapter_step_100.pt", map_location="cpu", weights_only=False)
        replay = torch.load(destination / "adapter_step_100.pt", map_location="cpu", weights_only=False)
        max_difference = max(float((original[k] - replay[k]).abs().max()) for k in original)
        write_json(destination / "replay.json", {
            "source": str(source), "schedule_horizon": args.steps, "updates": 100,
            "original_step_100_max_parameter_difference": max_difference, "history": history,
        })
        print(f"Replay {source.name}: step-100 max parameter difference={max_difference:.8g}", flush=True)
        del model, base, loader, P, codes
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, default=Path("table_runs/hf_flow_unet/cifar10"))
    parser.add_argument("--output", type=Path, default=Path("table_runs/embedding_budget/cifar10"))
    parser.add_argument("--steps", default="100,200,300,400,500,600,700,800,900,1000")
    parser.add_argument("--refine", action="store_true", help="Replay the first 100 updates and test every 10")
    cli = parser.parse_args()
    cli.output.mkdir(parents=True, exist_ok=True)
    messages = ["00000", "00111", "01010", "10101", "11001"]
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    if cli.refine:
        refined_source = cli.output / "checkpoints"
        refine_checkpoints(cli.source, refined_source, device)
        cli.source = refined_source
        cli.steps = "10,20,30,40,50,60,70,80,90,100"
    rows = []
    earliest = None
    for step in sorted(set(int(s) for s in cli.steps.split(","))):
        step_rows = []
        for message in messages:
            source = cli.source / message
            key_path = source / "watermark_key.pt"
            adapter = source / f"adapter_step_{step}.pt"
            key = torch.load(key_path, map_location="cpu", weights_only=False)
            if key["message"] != message:
                raise ValueError(f"Key message mismatch: {key_path}")
            args = SimpleNamespace(**key["config"])
            args.codebook_mode = getattr(args, "codebook_mode", "auto")
            args.n_queries = 4096
            args.n_detect_trials = 20
            args.num_workers = 0
            cfg = wm.DEFAULT_CONFIGS[args.dataset]
            wm.seed_everything(args.seed)
            checkpoint = wm.download_checkpoint(args.repo_id, args.checkpoint or cfg.filename)
            base = wm.build_model(cfg, device)
            wm.load_state_dict_flexible(base, checkpoint, device)
            base.eval().requires_grad_(False)
            model = wm.build_model(cfg, device)
            wm.load_state_dict_flexible(model, checkpoint, device)
            wm.add_lora(model, args.lora_rank, args.lora_alpha, args.lora_targets)
            wm.load_adapter(model, adapter, device)
            model.eval().requires_grad_(False)
            loader, _ = wm.load_real_data(cfg, args, device)
            P, codes = key["P"].to(device), key["codes"].to(device)
            wm.seed_everything(args.seed + 10000)
            metrics = wm.evaluate(model, base, cfg, P, codes, message, args, device,
                                  train_loader=loader)
            row = {"step": step, "message": message, "adapter": str(adapter),
                   "key": str(key_path), "n_queries": args.n_queries,
                   "n_trials": args.n_detect_trials, **metrics}
            rows.append(row)
            step_rows.append(row)
            write_json(cli.output / "recovery.json", rows)
            print(f"step={step} message={message} recovery={metrics['wm_acc']:.0f}% "
                  f"separation={metrics['sep_sigma']:.2f}", flush=True)
            del model, base, loader, P, codes
            torch.cuda.empty_cache()
        if all(row["wm_acc"] == 100.0 for row in step_rows):
            earliest = step
            break

    quality = []
    if earliest is not None:
        for message in messages:
            source = cli.source / message
            key = torch.load(source / "watermark_key.pt", map_location="cpu", weights_only=False)
            args = SimpleNamespace(**key["config"])
            args.num_workers = 0
            cfg = wm.DEFAULT_CONFIGS[args.dataset]
            wm.seed_everything(args.seed)
            checkpoint = wm.download_checkpoint(args.repo_id, args.checkpoint or cfg.filename)
            base = wm.build_model(cfg, device)
            wm.load_state_dict_flexible(base, checkpoint, device)
            base.eval().requires_grad_(False)
            model = wm.build_model(cfg, device)
            wm.load_state_dict_flexible(model, checkpoint, device)
            wm.add_lora(model, args.lora_rank, args.lora_alpha, args.lora_targets)
            wm.load_adapter(model, source / f"adapter_step_{earliest}.pt", device)
            model.eval().requires_grad_(False)
            loader, real = wm.load_real_data(cfg, args, device)
            # Pair clean and watermarked sampling with the same random stream.
            wm.seed_everything(args.seed + 20000)
            clean = wm.sample_for_fid(base, cfg, args, device, args.n_fid_samples)
            wm.seed_everything(args.seed + 20000)
            marked = wm.sample_for_fid(model, cfg, args, device, args.n_fid_samples)
            clean_fid = wm.compute_fid(real, clean, args, device)
            marked_fid = wm.compute_fid(real, marked, args, device)
            quality.append({"step": earliest, "message": message,
                            "n_samples": args.n_fid_samples, "feature": args.fid_feature,
                            "sampling_steps": args.n_sample_steps,
                            "fid_clean": clean_fid, "fid_wm": marked_fid,
                            "fid_ratio": marked_fid / clean_fid})
            write_json(cli.output / "quality.json", quality)
            print(f"FID step={earliest} message={message}: clean={clean_fid:.3f} "
                  f"wm={marked_fid:.3f} ratio={marked_fid / clean_fid:.4f}", flush=True)
            del model, base, loader, real, clean, marked
            torch.cuda.empty_cache()
    write_json(cli.output / "summary.json", {
        "earliest_tested_success": earliest, "recovery": rows, "quality": quality,
        "interpretation": "Earliest tested checkpoint; does not establish a minimum.",
        "recovery_seed": 10042, "quality_seed": 20042,
    })


if __name__ == "__main__":
    main()
