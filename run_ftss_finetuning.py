"""Apply the sibling FTSS repository's MNIST diagnostics to saved MLPs."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torchvision.datasets import MNIST

from repair_mlp_watermarks import FlowModel, signatures
from watermark_codebooks import decode_signature, message_code


def load_ftss(repo):
    path = repo / "finetune_memorization_ftss.py"
    spec = importlib.util.spec_from_file_location("external_ftss_finetune", path)
    module = importlib.util.module_from_spec(spec)
    # Keep the sibling repository read-only, including Python bytecode caches.
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_args():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--ftss-repo", type=Path, default=Path("../ftss-memorization"))
    p.add_argument("--source", type=Path,
                   default=Path("table_runs/main_architectures/mlp/orthogonal5"))
    p.add_argument("--marked-checkpoint", type=Path,
                   default=Path("table_runs/repairs/mlp/orthogonal5/wm_model_00111_0.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("table_runs/ftss_finetuning"))
    p.add_argument("--sizes", type=int, nargs="+", default=[50, 1000])
    p.add_argument("--steps", type=int, nargs="+", default=[0, 100, 300])
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-generated", type=int, default=256)
    p.add_argument("--n-probes", type=int, default=15)
    p.add_argument("--time-points", type=int, default=20)
    p.add_argument("--ode-steps", type=int, default=200)
    p.add_argument("--epsilon-scale", type=float, default=0.1)
    p.add_argument("--wm-epsilon", type=float, default=0.2)
    p.add_argument("--wm-lambda", type=float, default=0.018)
    p.add_argument("--message", default="00111")
    p.add_argument("--queries", type=int, default=4096)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if (min(args.sizes) < 2 or max(args.sizes) * 2 + 1000 > 10000
            or min(args.steps) < 0 or max(args.steps) < 1
            or min(args.runs, args.n_generated, args.n_probes, args.queries, args.trials) < 1
            or args.n_generated < 2 or args.time_points < 2 or args.ode_steps < 2
            or args.time_points > args.ode_steps or args.epsilon_scale <= 0
            or args.lr <= 0 or args.batch_size < 1):
        p.error("Invalid data sizes, steps, or evaluation settings")
    args.steps = sorted(set(args.steps) | {0})
    return args


def main():
    args = parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this study does not fall back to CPU")
    ftss = load_ftss(args.ftss_repo.resolve())
    clean_path = args.source / "clean_model_0.pt"
    key_path = args.source / "watermark_key.pt"
    key = torch.load(key_path, map_location=device, weights_only=True)
    mode = key["codebook_mode"]
    if len(args.message) != key["wm_bits"] or set(args.message) - {"0", "1"}:
        raise ValueError("Message does not match the saved key")
    projection, codes = key["P"], key["codes"]
    direction = projection @ message_code(args.message, codes, mode)
    clean_state = torch.load(clean_path, map_location="cpu", weights_only=True)
    marked_state = torch.load(args.marked_checkpoint, map_location="cpu", weights_only=True)
    images = MNIST("data", train=False, download=False).data.flatten(1).float() / 127.5 - 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "config.json").exists():
        raise FileExistsError("Use a fresh output directory; existing studies are preserved")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(
        checkpoint_sha256=digest(clean_path), marked_sha256=digest(args.marked_checkpoint),
        key_sha256=digest(key_path), ftss_source_sha256=digest(Path(ftss.__file__)),
        ftss_commit=subprocess.check_output(
            ["git", "-C", str(args.ftss_repo.resolve()), "rev-parse", "HEAD"], text=True).strip(),
        dataset="MNIST test split, unseen during original base training",
        ftss_definition="RMS finite-difference terminal gain; original sibling evaluator",
        replication_definition=("Generated sample closer to the tuned pool than to an equal-size "
                                "untouched pool, below the 1% calibration-neighbor distance"),
        ratio_baseline="Same existing pretrained clean checkpoint; not a new full-data FT control",
        optimizer="AdamW, weight_decay=0.01, fixed learning rate; paired minibatches",
        time_indices=np.linspace(0, args.ode_steps - 1, args.time_points, dtype=int).tolist(),
        source_evaluator=str(Path(ftss.__file__).resolve()))
    write_json(args.output_dir / "config.json", config)
    status = {"state": "running", "measurements": [], "elapsed_seconds": 0}
    begin = time.monotonic()
    try:
        for run in range(args.runs):
            seed = args.seed + 1000 * run
            order = np.random.default_rng(seed).permutation(len(images))
            rng = torch.Generator(device=device).manual_seed(seed)
            generation_noise = torch.randn(args.n_generated, 784, device=device, generator=rng)
            probe_noise = torch.randn(args.n_probes, 784, device=device, generator=rng)
            directions = torch.randn(args.n_probes, 784, device=device, generator=rng)
            loss_noise = torch.randn(max(args.sizes), 784, device=device, generator=rng)
            loss_times = torch.rand(max(args.sizes), 1, device=device, generator=rng)
            baseline_min = None
            for size in args.sizes:
                split = {"tuned": order[:size].tolist(), "reference": order[size:2 * size].tolist(),
                         "calibration": order[2 * size:2 * size + 1000].tolist()}
                write_json(args.output_dir / f"indices_n{size}_run{run}.json", split)
                tuned, reference, calibration = [images[split[k]] for k in
                                                   ("tuned", "reference", "calibration")]
                pool = tuned.to(device)
                clean_start = None

                def record(model, arm, step):
                    nonlocal baseline_min, clean_start
                    metrics = ftss.evaluate(model, generation_noise, probe_noise, directions,
                                            tuned, reference, calibration, loss_noise,
                                            loss_times, args)
                    if baseline_min is None:
                        baseline_min = metrics["g_min"]
                    if clean_start is None:
                        clean_start = metrics
                    values = signatures(model, projection, args.queries, args.trials, seed + 9000)
                    truth = tuple(map(int, args.message))
                    decoded = [decode_signature(v, codes, len(truth), mode)[0] for v in values]
                    row = {"run": run, "size": size, "arm": arm, "step": step, **metrics,
                           "ftss_ratio_to_pretrained_base": metrics["g_min"] / baseline_min,
                           "replication_change_from_clean": metrics["replication_rate"]
                           - clean_start["replication_rate"],
                           "wm_recovery_pct": 100 * sum(d == truth for d in decoded) / len(decoded)}
                    status["measurements"].append(row)
                    status["elapsed_seconds"] = time.monotonic() - begin
                    write_json(args.output_dir / "results.json", status)
                    print(f"run={run} n={size} {arm} step={step} "
                          f"FTSS/base={row['ftss_ratio_to_pretrained_base']:.4f} "
                          f"copy={row['replication_rate']:.4f} "
                          f"control={row['reference_match_rate']:.4f} "
                          f"WM={row['wm_recovery_pct']:.1f}%", flush=True)
                    return row

                for arm in ("clean_ft", "watermark_ft", "saved_watermark"):
                    torch.manual_seed(seed)
                    state = marked_state if arm == "saved_watermark" else clean_state
                    model = FlowModel(state).to(device)
                    record(model, arm, 0)
                    if arm == "saved_watermark":
                        del model
                        continue
                    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
                    training_rng = torch.Generator(device=device).manual_seed(seed + 100)
                    for step in range(1, max(args.steps) + 1):
                        model.train()
                        indices = torch.randint(size, (min(args.batch_size, size),),
                                                device=device, generator=training_rng)
                        x1 = pool[indices]
                        x0 = torch.randn(x1.shape, device=device, generator=training_rng)
                        t = torch.rand(len(x1), 1, device=device, generator=training_rng)
                        carrier = torch.sin(2 * math.pi * t)
                        target = x1 - x0
                        if arm == "watermark_ft":
                            target = target + args.wm_epsilon * carrier * direction
                        pred = model((1 - t) * x0 + t * x1, t)
                        loss = (pred - target).square().mean()
                        if arm == "watermark_ft":
                            loss = loss - args.wm_lambda * (carrier * (pred @ direction[:, None])).mean()
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                        if step in args.steps:
                            record(model, arm, step)
                    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                               args.output_dir / f"{arm}_n{size}_run{run}_final.pt")
                    del model, optimizer
        status["state"] = "complete"
    except Exception as exc:
        status.update(state="failed", error=str(exc))
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - begin
        write_json(args.output_dir / "results.json", status)


if __name__ == "__main__":
    main()
