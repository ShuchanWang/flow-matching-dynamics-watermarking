"""Preliminary paired modification/transfer screen of saved perfect 32-bit models."""
from repo_layout import ROOT, source_path

import argparse
import copy
from dataclasses import replace
import fcntl
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10

import run_baseline_first as baseline
import run_watermark_security as security
from run_embedding_strategy_robustness import save_torch, whole_model_attack


CONDITIONS = ["baseline", "prune25", "prune50", "quant8", "quant4", "quant2", "ft_low", "ft_high",
              "same_velocity", "same_sample", "small_velocity", "small_sample"]


def selected_rows(source):
    results = json.loads((source / "tune/results.json").read_text())
    rows = []
    for method in baseline.METHODS:
        candidates = [r for r in results["rows"] if r["payload"] == 32 and r["method"] == method
                      and r["metrics"]["recovery_percent"] == 100]
        if not candidates:
            raise ValueError(f"No perfectly decoded 32-bit development model for {method}")
        rows.append(min(candidates, key=lambda r: (r["metrics"]["fid_ratio_control"], r["trial"])))
    return rows


def condition_spec(name):
    if name.startswith("prune"):
        return "prune", int(name[5:]) / 100
    if name.startswith("quant"):
        return "quant", int(name[5:])
    if name.startswith("ft_"):
        return "full_finetune", 1e-5 if name == "ft_low" else 1e-4
    if name in CONDITIONS:
        return name, None
    raise ValueError(name)


def summarize(output, results):
    security.write_json(output / "results.json", results)
    lines = ["# Saved 32-Bit Modification and Transfer Screen", "", f"State: {results['state']}", "",
             "One development key per method, selected for perfect original recovery. "
             "Fresh-query baselines are retained regardless of their recovery. "
             "No five-bit or five-key confirmation is running. Quality and transfer "
             "must be interpreted jointly; weight extraction is N/A on the smaller architecture.", "",
             "| Method | Condition | Exact % | Bits % | FID/modified control | FID/original control | Parameter ratio |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in results["rows"]:
        m = row["metrics"]
        exact = f"{m['recovery_percent']:.1f}" if m["recovery_percent"] is not None else "N/A"
        bits = f"{m['bit_accuracy_percent']:.1f}" if m["bit_accuracy_percent"] is not None else "N/A"
        lines.append(f"| {row['method']} | {row['condition']} | {exact} | {bits} | "
                     f"{m['fid_ratio_modified_control']:.4f} | {m['fid_ratio_original_control']:.4f} | {m['parameter_ratio']:.3f} |")
    (output / "report.md").write_text('\n'.join(lines) + '\n')


def run(args):
    from huggingface_hub import hf_hub_download
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    source_config = json.loads((args.source / "tune/config.json").read_text())
    rows = [r for r in selected_rows(args.source) if r["method"] in args.methods]
    originals = []
    for row in rows:
        directory = args.source / "tune/32/0" / row["method"] / f"trial_{row['trial']}"
        originals.extend([directory / "marked/model.pt", directory / "control/model.pt"])
        originals.append(directory.parent.parent / "key.pt")
    sources = [Path(__file__), Path(baseline.__file__), Path(security.__file__), Path(security.flow.__file__),
               source_path('run_embedding_strategy_robustness.py'), source_path('run_flow_unet_robustness.py'),
               source_path('run_secret_carrier_pilot.py'), baseline.ROOT / "docs/BASELINE_SCREEN.md"]
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    selection=[{k: r[k] for k in ("method", "seed", "trial", "strength")} for r in rows],
                    source_hashes={str(p): security.digest(p) for p in sources},
                    artifact_hashes={str(p): security.digest(p) for p in originals},
                    original_config_sha256=security.digest(args.source / "tune/config.json"))
    path = args.output / "config.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Resume requires identical selection, models, sources and settings")
    security.write_json(path, manifest)
    path = args.output / "results.json"
    results = json.loads(path.read_text()) if path.exists() else dict(rows=[])
    results.update(state="running", expected_evaluations=len(rows) * len(args.conditions),
                   samples=args.samples, queries=args.queries, trials=args.trials,
                   fidelity="one-key 5000-image screen" if args.samples >= 5000 else "smoke/non-publication")
    results.pop("error", None)
    summarize(args.output, results)
    cfg = security.flow.DEFAULT_CONFIGS["cifar10"]
    checkpoint = Path(hf_hub_download("WayBob/FlowMatching-Unet-Celeb-64x64", cfg.filename, local_files_only=True))
    if security.digest(checkpoint) != source_config["base_sha256"]:
        raise ValueError("Clean pretrained checkpoint changed")
    base = security.flow.build_model(cfg, device)
    security.flow.load_state_dict_flexible(base, str(checkpoint), device)
    base.eval().requires_grad_(False)
    dataset = CIFAR10(str(baseline.ROOT / "data"), train=True, download=False)
    pool = torch.from_numpy(dataset.data[:20000]).permute(0, 3, 1, 2).to(device).float() / 127.5 - 1
    labels = torch.tensor(dataset.targets[:20000], device=device)
    evaluation = argparse.Namespace(**vars(args))
    verification = argparse.Namespace(**vars(args))
    verification.eval_seed = args.eval_seed + 200000
    reference_path = args.output / "real_features.pt"
    try:
        if reference_path.exists():
            real = torch.load(reference_path, weights_only=True, map_location="cpu")
        elif args.samples == source_config["arguments"]["samples"]:
            real = torch.load(args.source / "real_features.pt", weights_only=True, map_location="cpu")
            save_torch(reference_path, real)
        else:
            test = CIFAR10(str(baseline.ROOT / "data"), train=False, download=False)
            indices = np.concatenate([np.flatnonzero(np.asarray(test.targets) == y)[:args.samples // 10] for y in range(10)])
            images = torch.from_numpy(test.data[indices]).permute(0, 3, 1, 2).float() / 127.5 - 1
            real = security.flow.inception_features(images, device, args.feature_batch)
            save_torch(reference_path, real)
        # Finish each condition across methods before advancing to the next transformation.
        for condition in args.conditions:
            kind, severity = condition_spec(condition)
            for row in rows:
                method = row["method"]
                if any(r["method"] == method and r["condition"] == condition for r in results["rows"]):
                    continue
                results["current"] = dict(method=method, condition=condition)
                summarize(args.output, results)
                directory = args.output / method / condition
                directory.mkdir(parents=True, exist_ok=True)
                original = args.source / "tune/32/0" / method / f"trial_{row['trial']}"
                key = torch.load(original.parent.parent / "key.pt", weights_only=True, map_location=device)
                pair = {}
                for role in ("control", "marked"):
                    model = copy.deepcopy(base)
                    model.load_state_dict(torch.load(original / role / "model.pt", weights_only=True, map_location=device))
                    model.eval().requires_grad_(False)
                    if kind in ("prune", "quant"):
                        scope = whole_model_attack(model, kind, severity)
                        security.write_json(directory / (role + "_scope.json"), scope)
                    elif kind != "baseline":
                        settings = argparse.Namespace(steps=args.steps, lr=severity if kind == "full_finetune" else 1e-4,
                                                      batch_size=args.batch, seed=row["seed"] + 10000)
                        small = condition.startswith("small_")
                        torch.manual_seed(settings.seed)
                        student = (security.flow.build_model(replace(cfg, num_channels=32), device)
                                   if small else copy.deepcopy(base))
                        if kind == "full_finetune":
                            student = copy.deepcopy(model)
                            arm, training_pool, training_labels = "full_finetune", pool, labels
                        elif condition.endswith("velocity"):
                            arm, training_pool, training_labels = "velocity", pool, labels
                        else:
                            arm = "sample_only"
                            sample_path = directory / (role + "_teacher_samples.pt")
                            if sample_path.exists():
                                saved = torch.load(sample_path, weights_only=True, map_location=device)
                            else:
                                images, yy = security.generate(model, args.teacher_samples, args.sample_steps, args.sample_batch,
                                                                args.eval_seed + 50000, device)
                                saved = dict(images=images, labels=yy)
                                save_torch(sample_path, saved)
                            training_pool, training_labels = saved["images"].to(device), saved["labels"].to(device)
                        training_dir = directory / role
                        training_dir.mkdir(exist_ok=True)
                        model = security.train_student(student, model, training_pool, training_labels, arm, settings, training_dir)
                    pair[role] = model.eval().requires_grad_(False)
                control_dir, marked_dir = directory / "control", directory / "marked"
                control_dir.mkdir(exist_ok=True)
                marked_dir.mkdir(exist_ok=True)
                reuse = (condition == "baseline" and all(getattr(args, k) == source_config["arguments"][k]
                         for k in ("samples", "sample_steps", "sample_batch", "feature_batch", "eval_seed", "kid_subset", "kid_subsets")))
                if reuse:
                    control_quality = json.loads((original / "control/quality.json").read_text())
                    marked_quality = json.loads((original / "marked/quality.json").read_text())
                    control_features = torch.load(original / "control/features.pt", weights_only=True, map_location="cpu")
                    marked_features = torch.load(original / "marked/features.pt", weights_only=True, map_location="cpu")
                else:
                    control_quality, control_features = baseline.quality(pair["control"], real, evaluation, control_dir)
                    marked_quality, marked_features = baseline.quality(pair["marked"], real, evaluation, marked_dir)
                if method == "weight" and condition.startswith("small_"):
                    metrics = dict(recovery_percent=None, bit_accuracy_percent=None,
                                   reason="N/A: selected white-box tensor and projection do not match the smaller architecture")
                else:
                    metrics = baseline.verify(pair["marked"], method, key, pool, labels, verification)
                metrics.update(marked_quality, control_quality=control_quality,
                               fid_ratio_modified_control=marked_quality["fid"] / control_quality["fid"],
                               fid_ratio_original_control=marked_quality["fid"] / row["metrics"]["control_quality"]["fid"],
                               kid_modified_control=baseline.kid(marked_features, control_features, evaluation, paired=True),
                               parameter_ratio=sum(p.numel() for p in pair["marked"].parameters()) / sum(p.numel() for p in base.parameters()))
                security.write_json(directory / "metrics.json", metrics)
                results["rows"].append(dict(method=method, condition=condition, metrics=metrics))
                summarize(args.output, results)
                print(f"RESULT {method}/{condition}: recovery={metrics['recovery_percent']} FID/control={metrics['fid_ratio_modified_control']:.4f}", flush=True)
                del pair, model
                torch.cuda.empty_cache()
        results.update(state="complete", current=None)
    except Exception as exc:
        results.update(state="failed", error=repr(exc))
        raise
    finally:
        summarize(args.output, results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", type=Path, default=baseline.ROOT / "table_runs/baseline_first")
    p.add_argument("--output", type=Path, default=baseline.ROOT / "table_runs/baseline_screen32")
    p.add_argument("--methods", nargs="+", choices=baseline.METHODS, default=list(baseline.METHODS))
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--teacher-samples", type=int, default=2000)
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--sample-steps", type=int, default=100)
    p.add_argument("--sample-batch", type=int, default=128)
    p.add_argument("--feature-batch", type=int, default=32)
    p.add_argument("--queries", type=int, default=4096)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--eval-batch", type=int, default=512)
    p.add_argument("--kid-subset", type=int, default=1000)
    p.add_argument("--kid-subsets", type=int, default=20)
    p.add_argument("--eval-seed", type=int, default=930042)
    a = p.parse_args()
    if min(a.steps, a.batch, a.teacher_samples, a.samples, a.sample_steps, a.sample_batch, a.feature_batch,
           a.queries, a.trials, a.eval_batch, a.kid_subset, a.kid_subsets) < 1 or a.samples % 10 or a.samples > 10000 or a.kid_subset < 2:
        p.error("Invalid evaluation counts")
    if len(set(a.conditions)) != len(a.conditions) or len(set(a.methods)) != len(a.methods):
        p.error("Duplicate conditions/methods")
    a.source, a.output = a.source.resolve(), a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / "study.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(a)
