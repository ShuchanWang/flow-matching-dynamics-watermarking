"""Execute original FTSS MLP definitions, then matched clean-model fine-tuning."""

import argparse
import ast
import contextlib
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ftss_matched_mlp_matplotlib")
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from evaluate_generative_generalization import (DigitEvaluator, balanced_part, features, prdc,
                                                semantic_diversity, training_preference)
from run_ftss_finetuning import digest, load_ftss, write_json
from run_unmarked_memorization import balanced_split
from ftss_batched import estimate_ftss_batched


ROOT = Path(__file__).resolve().parent


def execution_only_update(existing, current):
    normalized = dict(existing)
    for key in ("runner_sha256", "ftss_backend", "batched_source_sha256"):
        normalized[key] = current[key]
    return normalized == current


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError(f"Missing source definitions in {path}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def original_definitions(repo, device, dataset, steps=20000):
    namespace = {"__name__": "matched_original_ftss", "torch": torch, "nn": nn, "F": F,
                 "np": np, "random": random, "os": os, "plt": plt, "copy": copy, "time": time,
                 "DataLoader": DataLoader, "Subset": Subset, "SEED": 42, "D": 784,
                 "HIDDEN_DIM": 2048, "STEPS": steps, "LR": 1e-4, "EMA_DECAY": 0.999,
                 "USE_AMP": device.type == "cuda", "BATCH_SIZE": 64, "device": device,
                 "train_ds": dataset, "DATASET_NAME": "mnist", "IN_CHANNELS": 1, "IMG_SIZE": 28}
    extract(repo / "train_memorization_mlp.py",
            ["MLPFlow", "update_ema", "visualize_samples", "train_with_tracking"], namespace)
    extract(repo / "test_memorization_trend_mlp.py", ["estimate_ftss"], namespace)
    return namespace


def planned_subsets(sizes, runs, count=60000):
    # The original script seeds Python once, then samples across size/run order.
    rng = random.Random(42)
    result = {}
    for size in sizes:
        for run in range(runs):
            state = rng.getstate()
            indices = rng.sample(range(count), size)
            result[size, run] = (state, indices)
    return result


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def numeric(value):
    if isinstance(value, dict):
        return {key: numeric(v) for key, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [numeric(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class Measurement:
    def __init__(self, args, namespace, device):
        self.args, self.namespace, self.device = args, namespace, device
        self.ftss_helpers = load_ftss(args.repo)
        self.dataset = datasets.MNIST(str(ROOT / "data"), train=False, download=False)
        self.images = self.dataset.data.flatten(1).float() / 127.5 - 1
        saved = torch.load(args.classifier, map_location="cpu", weights_only=True)
        self.classifier = DigitEvaluator().to(device)
        self.classifier.load_state_dict(saved["model"])
        self.classifier.eval().requires_grad_(False)
        self.real = features(self.classifier, self.dataset.data[:, None].float() / 255, device)

    def __call__(self, model, name, split, seed):
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            started = time.monotonic()
            estimator = (estimate_ftss_batched if self.args.ftss_backend == "batched"
                         else self.namespace["estimate_ftss"])
            result = numeric(estimator(
                model, self.device, n_samples=self.args.probes, n_steps=self.args.ode_steps))
            result["ftss_backend"] = self.args.ftss_backend
            torch.manual_seed(seed + 1)
            noise = torch.randn(256, 784, device=self.device)
            samples = self.ftss_helpers.generate(model, noise, self.args.ode_steps, 64)
            generated = features(self.classifier, samples.clamp(-1, 1).reshape(-1, 1, 28, 28).add(1).div(2), self.device)
            real_indices = balanced_part(split["calibration"], 0, 200)
            predicted = generated["probabilities"].argmax(1)
            result.update(prdc(self.real["spatial"][real_indices], generated["spatial"][:200], 3))
            result.update(semantic_diversity(generated["spatial"], predicted,
                                             self.real["spatial"][split["calibration"]],
                                             self.dataset.targets.numpy()[split["calibration"]]))
            result.update(training_preference(generated["spatial"], self.real["spatial"][split["tuned"]],
                                               self.real["spatial"][split["reference"]]))
            result["measurement_seconds"] = time.monotonic() - started
            np.savez_compressed(self.args.output / f"{name}_audit.npz", generated_samples=samples.numpy(),
                                **generated)
            print(f"EVALUATED {name}: g_min={result['min']:.5f} coverage={result['coverage']:.3f} "
                  f"precision={result['precision']:.3f} style={result['style_diversity_ratio']:.3f}", flush=True)
            return result


def report_markdown(output, report):
    lines = ["# Matched FTSS MLP Reproduction and Fine-Tuning", "",
             f"State: {report['state']}. Phase: {report['phase']}. Tables update as measurements finish.", "",
             "Original training/model/evaluator function bodies are extracted without edits from the "
             "sibling repository. The selected from-scratch data sizes are endpoint checks, not the full "
             "published sweep. Subset RNG states preserve the original selected-size/run order on restart.", "",
             "Training: four 2048-wide hidden layers, LayerNorm, dropout 0.1, Xavier initialization, "
             "AdamW, OneCycle peak LR 1e-4, batch 64, gradient clipping 5, EMA 0.999, CUDA AMP. "
             "Paper estimator: 15 trajectories and 200 Euler steps by default, 20 injection times through "
             "0.95, fresh directions at each time. A validated batched backend preserves those probes; "
             "backend and execution-change provenance are recorded. "
             "The denominator below is the mean minimum gain of "
             "the largest from-scratch subset across runs, matching the original evaluation script.", "",
             "## From Scratch", "", "| Images | Run | Minimum FTSS | Paper ratio |",
             "|---:|---:|---:|---:|"]
    scratch = report["scratch"]
    baseline_rows = [r for r in scratch if r["size"] == max(report["config"]["scratch_sizes"])]
    denominator = np.mean([r["metrics"]["min"] for r in baseline_rows]) if baseline_rows else None
    for row in scratch:
        value = row["metrics"]["min"]
        ratio = f"{value / denominator:.4f}" if denominator else "pending"
        lines.append(f"| {row['size']} | {row['run']} | {value:.5f} | {ratio} |")
    lines += ["", "## Matched Fine-Tuning", "",
              "Starts from the largest-subset EMA checkpoint. New class-balanced MNIST test examples "
              "are disjoint from original training; reference/calibration pools are disjoint from tuning. "
              "Fine-tuning uses the same architecture, dropout, optimizer family, batch, peak LR, clipping "
              "and EMA, with OneCycle compressed to the requested fine-tuning duration. Ordinary and EMA "
              "weights are evaluated separately with the same paper estimator protocol. This is an extension "
              "to fine-tuning, not a fine-tuning experiment reported in the original paper.", "",
              "| Images | Run | Steps | Weights | FTSS / Before | Precision | Held-out Coverage | Style Diversity | Training Preference |",
              "|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    for row in report["finetuning"]:
        m = row["metrics"]
        lines.append(f"| {row['size']} | {row['run']} | {row['step']} | {row['weights']} | "
                     f"{row['ratio_to_before']:.4f} | {m['precision']:.3f} | {m['coverage']:.3f} | "
                     f"{m['style_diversity_ratio']:.3f} | {m['training_preference']:.3f} |")
    lines += ["", "External outcomes use the existing independent MNIST classifier and authors' PRDC "
              "implementation, with fixed spatial features and 200 real/200 generated samples. "
              "Scratch training-set preference is omitted because its evaluator pools are held out "
              "from scratch training, not members. All configurations, source hashes, and audits are saved.", ""]
    (output / "report.md").write_text("\n".join(lines))


def run(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output / ".matplotlib"))
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.MNIST(str(ROOT / "data"), train=True, download=False, transform=transform)
    namespace = original_definitions(args.repo, device, train_ds, args.scratch_steps)
    source_paths = [args.repo / name for name in ("train_memorization_mlp.py", "test_memorization_trend_mlp.py")]
    config = {"scratch_sizes": args.scratch_sizes, "runs": args.runs, "scratch_steps": args.scratch_steps,
              "fine_sizes": args.fine_sizes, "fine_steps": args.fine_steps, "check_steps": args.check_steps,
              "probes": args.probes, "ode_steps": args.ode_steps, "device": args.device,
              "source_sha256": {str(p): digest(p) for p in source_paths},
              "runner_sha256": digest(Path(__file__)),
              "ftss_backend": args.ftss_backend,
              "batched_source_sha256": digest(ROOT / "ftss_batched.py") if args.ftss_backend == "batched" else None,
              "classifier_sha256": digest(args.classifier),
              "paper_training_recipe": args.scratch_steps == 20000,
              "paper_estimator_parameters": args.probes == 15 and args.ode_steps == 200,
              "evaluation_seeds": "scratch 90000 + run; fine-tuning 100000 + run; same probes within each comparison",
              "source_execution": "Unedited AST function bodies; EMA observer adds progress logging only",
              "selected_endpoint_reproduction": True}
    config_path, results_path = args.output / "config.json", args.output / "results.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if not args.resume:
            raise ValueError("Existing run requires --resume with the identical protocol")
        report = json.loads(results_path.read_text())
        if existing != config:
            if not args.accept_execution_update or not execution_only_update(existing, config):
                raise ValueError("Protocol changed; only a validated execution-only update may resume")
            validation = json.loads(args.accept_execution_update.read_text())
            tested_sizes = {r["size"] for r in validation["models"] if r["passed"]}
            if (args.ftss_backend != "batched" or not validation["all_passed"]
                    or validation["candidate_source_sha256"] != config["batched_source_sha256"]
                    or not tested_sizes.issuperset(args.scratch_sizes)):
                raise ValueError("Missing real-model equivalence validation for the batched evaluator")
            report.setdefault("execution_updates", []).append({
                "previous_config": existing, "new_config": config,
                "validation_sha256": digest(args.accept_execution_update),
                "reason": "Batch independent FTSS branches; preserve paper training and measurement protocol"})
            report["config"] = config
            write_json(config_path, config)
            print("Accepted validated batched evaluator update; preserving completed results", flush=True)
    else:
        write_json(config_path, config)
        report = {"state": "running", "phase": "scratch", "scratch": [], "finetuning": [], "config": config}
    report.update(state="running", pid=os.getpid())
    report.pop("error", None)
    def save():
        write_json(results_path, report)
        report_markdown(args.output, report)
    reproduction = args.output / "reproduction"
    for subdir in ("checkpoints", "figures"):
        (reproduction / subdir).mkdir(parents=True, exist_ok=True)
    plans = planned_subsets(args.scratch_sizes, args.runs)
    measurement = Measurement(args, namespace, device)
    try:
        for size in args.scratch_sizes:
            for run_id in range(args.runs):
                if any((r["size"], r["run"]) == (size, run_id) for r in report["scratch"]):
                    continue
                report.update(phase="scratch", current={"size": size, "run": run_id})
                save()
                state, indices = plans[size, run_id]
                write_json(args.output / f"scratch_n{size}_run{run_id}_indices.json", indices)
                random.setstate(state)
                np.random.seed(42)
                torch.manual_seed(42)
                print(f"SCRATCH N={size} run={run_id}, original recipe {args.scratch_steps} steps", flush=True)
                original_update = namespace["update_ema"]
                progress = {"step": 0, "start": time.monotonic()}
                def observed_update(ema_model, training_model, decay):
                    original_update(ema_model, training_model, decay)
                    progress["step"] += 1
                    if progress["step"] % 1000 == 0:
                        elapsed = time.monotonic() - progress["start"]
                        print(f"SCRATCH N={size} run={run_id} step={progress['step']}/{args.scratch_steps} "
                              f"elapsed={elapsed:.0f}s", flush=True)
                        report["current"]["step"] = progress["step"]
                        save()
                namespace["update_ema"] = observed_update
                try:
                    with working_directory(reproduction):
                        model, _, _ = namespace["train_with_tracking"](size, run_id)
                finally:
                    namespace["update_ema"] = original_update
                checkpoint = reproduction / "checkpoints" / f"mlp_mnist_n{size}_run{run_id}.pt"
                split = balanced_split(measurement.dataset.targets.numpy(), 50, 42 + 1000 * run_id)
                metrics = measurement(model, f"scratch_n{size}_run{run_id}", split, 90000 + run_id)
                for key in ("training_preference", "training_distance_squared", "training_anchor_advantage"):
                    metrics[key.replace("training_", "heldout_pool_")] = metrics.pop(key)
                report["scratch"].append({"size": size, "run": run_id, "checkpoint": str(checkpoint),
                                          "checkpoint_sha256": digest(checkpoint), "metrics": metrics})
                save()
                del model
        for run_id in range(args.runs):
            checkpoint = reproduction / "checkpoints" / f"mlp_mnist_n{max(args.scratch_sizes)}_run{run_id}.pt"
            for size in args.fine_sizes:
                prefix = f"fine_n{size}_run{run_id}"
                rows = [r for r in report["finetuning"] if r["size"] == size and r["run"] == run_id]
                if all(any(r["step"] == args.fine_steps and r["weights"] == kind for r in rows)
                       for kind in ("raw", "ema")):
                    continue
                report.update(phase="finetuning", current={"size": size, "run": run_id})
                save()
                model = namespace["MLPFlow"]().to(device)
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
                model.requires_grad_(True)
                ema = copy.deepcopy(model).eval().requires_grad_(False)
                split = balanced_split(measurement.dataset.targets.numpy(), size, 42 + 1000 * run_id)
                write_json(args.output / f"{prefix}_indices.json", split)
                pool = measurement.images[split["tuned"]].to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999))
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4,
                                                               total_steps=args.fine_steps, pct_start=0.1)
                scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
                torch.manual_seed(20000 + 1000 * run_id)
                snapshot = args.output / f"{prefix}_latest.pt"
                step = 0
                if snapshot.exists():
                    saved = torch.load(snapshot, map_location="cpu", weights_only=True)
                    model.load_state_dict(saved["model"])
                    ema.load_state_dict(saved["ema"])
                    optimizer.load_state_dict(saved["optimizer"])
                    scheduler.load_state_dict(saved["scheduler"])
                    scaler.load_state_dict(saved["scaler"])
                    torch.set_rng_state(saved["cpu_rng"])
                    if device.type == "cuda":
                        torch.cuda.set_rng_state(saved["cuda_rng"], device)
                    step = saved["step"]
                def measure(current):
                    kinds = ("ema",) if current == 0 else ("raw", "ema")
                    for kind in kinds:
                        if any(r["step"] == current and r["weights"] == kind for r in rows):
                            continue
                        metrics = measurement(ema if kind == "ema" else model,
                                              f"{prefix}_step{current}_{kind}", split, 100000 + run_id)
                        baseline = next((r["metrics"]["min"] for r in rows if r["step"] == 0), metrics["min"])
                        row = {"size": size, "run": run_id, "step": current, "weights": kind,
                               "metrics": metrics, "ratio_to_before": metrics["min"] / baseline}
                        rows.append(row)
                        report["finetuning"].append(row)
                        save()
                measure(step)
                for current in range(step + 1, args.fine_steps + 1):
                    model.train()
                    index = torch.randperm(len(pool))[:min(64, len(pool))]
                    target = pool[index]
                    noise = torch.randn_like(target)
                    times = torch.rand(len(target), 1, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device.type, enabled=device.type == "cuda"):
                        loss = F.mse_loss(model((1-times)*noise + times*target, times), target-noise)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Nonfinite fine-tuning loss: {prefix} {current}")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    namespace["update_ema"](ema, model, 0.999)
                    if current % 500 == 0:
                        print(f"FINE {prefix} {current}/{args.fine_steps} loss={float(loss):.6f}", flush=True)
                    if current in args.check_steps:
                        saved = {"model": model.state_dict(), "ema": ema.state_dict(),
                                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                 "scaler": scaler.state_dict(), "step": current, "cpu_rng": torch.get_rng_state()}
                        if device.type == "cuda":
                            saved["cuda_rng"] = torch.cuda.get_rng_state(device)
                        temporary = snapshot.with_suffix(".tmp")
                        torch.save(saved, temporary)
                        temporary.replace(snapshot)
                        measure(current)
                del model, ema, optimizer, pool
                torch.cuda.empty_cache() if device.type == "cuda" else None
        report.update(state="complete", phase="complete")
        report.pop("current", None)
    except Exception as exc:
        report.update(state="failed", error=str(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--repo", type=Path, default=ROOT.parent / "ftss-memorization")
    p.add_argument("--output", type=Path, default=ROOT / "table_runs/ftss_mlp_matched")
    p.add_argument("--classifier", type=Path, default=ROOT / "table_runs/generative_generalization/feature_evaluator.pt")
    p.add_argument("--scratch-sizes", type=int, nargs="+", default=[50, 12000])
    p.add_argument("--scratch-steps", type=int, default=20000)
    p.add_argument("--fine-sizes", type=int, nargs="+", default=[50, 1000])
    p.add_argument("--fine-steps", type=int, default=3000)
    p.add_argument("--check-steps", type=int, nargs="+", default=[1000, 3000])
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--probes", type=int, default=15)
    p.add_argument("--ode-steps", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--ftss-backend", choices=["literal", "batched"], default="literal")
    p.add_argument("--accept-execution-update", type=Path)
    args = p.parse_args()
    if (min(args.scratch_steps, args.runs, args.probes, args.ode_steps) < 1 or args.fine_steps < 10
            or any(n not in (50, 100, 200, 500, 1000, 2000, 5000, 12000) for n in args.scratch_sizes)
            or any(n < 10 or n > 4000 or n % 10 for n in args.fine_sizes)):
        p.error("Invalid matched-reproduction parameters")
    args.scratch_sizes = sorted(set(args.scratch_sizes))
    args.fine_sizes = sorted(set(args.fine_sizes))
    args.check_steps = sorted({step for step in args.check_steps if 0 < step <= args.fine_steps} | {args.fine_steps})
    args.output = args.output.resolve()
    args.repo = args.repo.resolve()
    args.classifier = args.classifier.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
