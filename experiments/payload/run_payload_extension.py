"""Complete fixed-K payload and original-key prefix-query studies without replacing old runs."""
from repo_layout import ROOT, source_path

import argparse
import contextlib
import copy
import fcntl
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torchvision.datasets import CIFAR10

import watermark_hf_flow_unet as flow
from artifact_io import digest, write_json
from finalize_payload_paper import block_hash, payload_block, finalize


FIXED = ("repo_id", "dataset", "checkpoint", "seed", "wm_K", "codebook_mode", "wm_eps",
         "wm_lambda", "wm_proj_weight", "wm_tanh_scale", "objective", "detect_distribution",
         "lora_rank", "lora_alpha", "lora_targets", "train_extra", "steps", "post_ft_steps",
         "batch_size", "grad_accum", "lr", "weight_decay", "n_train_samples", "data_source",
         "n_fid_samples", "fid_batch_size", "fid_feature", "n_sample_steps", "eval_class",
         "n_queries", "n_detect_trials", "num_workers")


def make_plan(source, output):
    existing = []
    for path in source.rglob("results.json"):
        saved = json.loads(path.read_text())
        c = saved["config"]
        if len(c["wm_message"]) in (64, 128):
            existing.append({"run_dir": str(path.parent.resolve()), "config": c,
                             "original_metrics": saved["metrics"]})
    existing.sort(key=lambda row: (len(row["config"]["wm_message"]), row["config"]["wm_message"]))
    if len(existing) != 10 or any(sum(len(r["config"]["wm_message"]) == n for r in existing) != 5 for n in (64, 128)):
        raise ValueError("Expected five completed messages at each original payload")
    reference = existing[0]["config"]
    if (reference["wm_K"], reference["wm_eps"], reference["steps"], reference["seed"]) != (128, 1.5, 1000, 42):
        raise ValueError("Unexpected source recipe")
    for row in existing:
        if any(row["config"][key] != reference[key] for key in FIXED):
            raise ValueError("Original payload runs are not a controlled comparison")
        row["original_key_sha256"] = digest(Path(row["run_dir"]) / "watermark_key.pt")
        row["original_adapter_sha256"] = digest(Path(row["run_dir"]) / "adapter_final.pt")
    additions = []
    for row in existing:
        if len(row["config"]["wm_message"]) != 128:
            continue
        config = copy.deepcopy(reference)
        config.update(wm_message=row["config"]["wm_message"][:96], output_dir=str(output / "trained96"))
        additions.append({"run_dir": str(output / "trained96" / "cifar10" / config["wm_message"]),
                          "config": config, "paired_128_message": row["config"]["wm_message"]})
    return additions + existing


def score_prefixes(signed_coordinates, truth=None):
    signed = np.asarray(signed_coordinates, dtype=np.float64)
    if signed.ndim != 2 or len(signed) < 1:
        raise ValueError("Expected trials by bit coordinates")
    correct = signed > 0
    if truth is not None:
        correct |= (signed == 0) & (np.asarray(truth) == 0)
    return {"exact_recovery_percent": float(100 * np.all(correct, axis=1).mean()),
            "bit_accuracy_percent": float(100 * correct.mean()),
            "mean_signed_bit_margin": float(signed.mean()),
            "minimum_signed_bit_margin_mean": float(signed.min(axis=1).mean()),
            "per_bit_signed_mean": signed.mean(axis=0).tolist(),
            "per_bit_signed_std": signed.std(axis=0).tolist(),
            "bits_with_nonpositive_trial_mean": int((signed.mean(axis=0) <= 0).sum())}


def orthogonal_covariance(codes):
    mean = codes.mean(axis=0)
    return codes.T @ codes / len(codes) - np.outer(mean, mean)


@torch.no_grad()
def query_trial(model, cfg, pool, labels, projection, axes, query_counts, batch_size, seed, device,
                precision="float32"):
    generator = torch.Generator(device=device).manual_seed(seed)
    total = torch.zeros(projection.shape[1], dtype=torch.float64, device=device)
    squares = torch.zeros_like(total)
    result = []
    count = 0
    for endpoint in query_counts:
        while count < endpoint:
            n = min(batch_size, endpoint - count)
            xs, ts, ys = [], [], []
            # Query generation is independent of the forward-pass batch size.
            for offset in range(0, n, 512):
                b = min(512, n-offset)
                index = torch.randint(len(pool), (b,), device=device, generator=generator)
                target = pool[index]
                noise = torch.randn(target.shape, device=device, generator=generator)
                t = torch.rand(b, device=device, generator=generator)
                view = t[:, None, None, None]
                xs.append((1 - view) * noise + view * target)
                ts.append(t)
                ys.append(labels[index])
            x, t = torch.cat(xs), torch.cat(ts)
            y = torch.cat(ys) if cfg.class_cond else None
            with torch.autocast(device.type, dtype=torch.float16, enabled=precision == "float16"):
                velocity = flow.model_velocity(model, x, t, y).flatten(1).float()
            demodulated = torch.sin(2 * math.pi * t)[:, None] * (velocity @ projection)
            total += demodulated.double().sum(0)
            squares += demodulated.double().square().sum(0)
            count += n
        signature = total / count
        coordinates = signature @ axes.double().T
        result.append(coordinates.cpu().numpy())
    return np.stack(result), (total / count).cpu().numpy(), (squares / count).cpu().numpy()


def original_gpu_state_for_quality(rng, cfg, config, device):
    torch.cuda.set_rng_state(rng, device)
    # Advance the original marked/clean detection draws, without their redundant forwards.
    for _ in range(config["n_detect_trials"] * 2):
        for start in range(0, config["n_queries"], config["batch_size"]):
            n = min(config["batch_size"], config["n_queries"] - start)
            torch.randn(n, *cfg.dim, device=device)
            torch.rand(n, device=device)


@torch.no_grad()
def quality96(model, base, real_images, cfg, args, run_dir, output, device):
    path = run_dir / "quality.json"
    if path.exists():
        return json.loads(path.read_text())
    rng = torch.load(run_dir / "quality_rng.pt", map_location="cpu", weights_only=True)
    with torch.random.fork_rng(devices=[device.index or torch.cuda.current_device()]):
        original_gpu_state_for_quality(rng, cfg, vars(args), device)
        cache_path = output / "quality_reference.pt"
        state_hash = hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest()
        if cache_path.exists():
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            if cache["rng_sha256"] != state_hash:
                raise ValueError("Quality-generation RNG differs across matched runs")
            for start in range(0, args.n_fid_samples, args.fid_batch_size):
                torch.randn(min(args.fid_batch_size, args.n_fid_samples-start), *cfg.dim, device=device)
        else:
            print("Generating shared 500-image clean FID reference", flush=True)
            clean = flow.sample_for_fid(base, cfg, args, device, args.n_fid_samples)
            real_features = flow.inception_features(real_images[:args.n_fid_samples], device, args.fid_batch_size)
            clean_features = flow.inception_features(clean, device, args.fid_batch_size)
            cache = {"rng_sha256": state_hash, "real_features": real_features, "clean_features": clean_features}
            cache["fid_clean"] = flow.compute_fid_from_features(real_features, clean_features)
            torch.save(cache, cache_path)
        print("Generating watermarked FID samples", flush=True)
        marked = flow.sample_for_fid(model, cfg, args, device, args.n_fid_samples)
        marked_features = flow.inception_features(marked, device, args.fid_batch_size)
        fid = flow.compute_fid_from_features(cache["real_features"], marked_features)
    result = {"fid_clean": cache["fid_clean"], "fid_wm": fid, "fid_ratio": fid / cache["fid_clean"],
              "samples": args.n_fid_samples, "sample_steps": args.n_sample_steps,
              "reference": "Same CIFAR-10 test reference and original Inception feature implementation",
              "rng_protocol": "Original post-training CUDA state advanced through original detection draws"}
    write_json(path, result)
    return result


def prepare96(row, output, checkpoint, device):
    run_dir = Path(row["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    config = row["config"]
    args = SimpleNamespace(**config)
    cfg = flow.DEFAULT_CONFIGS[args.dataset]
    flow.seed_everything(args.seed)
    base = flow.build_model(cfg, device)
    flow.load_state_dict_flexible(base, checkpoint, device)
    base.eval().requires_grad_(False)
    model = flow.build_model(cfg, device)
    flow.load_state_dict_flexible(model, checkpoint, device)
    flow.add_lora(model, args.lora_rank, args.lora_alpha, args.lora_targets)
    model.to(device)
    projection, axes = flow.make_codebook(int(np.prod(cfg.dim)), 96, args.wm_K, device, "hypercube")
    train_loader, real_images = flow.load_real_data(cfg, args, device)
    training_path = run_dir / "training.json"
    if training_path.exists():
        training = json.loads(training_path.read_text())
        if training["config"] != config:
            raise ValueError("96-bit training recipe changed")
        flow.load_adapter(model, run_dir / "adapter_final.pt", device)
    else:
        print(f"TRAIN96 {args.wm_message}", flush=True)
        key = {"P": projection.cpu(), "codes": axes.cpu(), "message": args.wm_message, "config": config}
        torch.save(key, run_dir / "watermark_key.pt")
        started = time.monotonic()
        history = flow.train_lora(model, base, train_loader, cfg, projection,
                                  flow.hypercube_code(tuple(map(int, args.wm_message)), axes),
                                  args, device, run_dir)
        torch.save(torch.cuda.get_rng_state(device), run_dir / "quality_rng.pt")
        flow.save_adapter(model, run_dir / "adapter_final.pt")
        write_json(training_path, {"config": config, "history": history,
                                  "training_seconds": time.monotonic() - started,
                                  "adapter_sha256": digest(run_dir / "adapter_final.pt"),
                                  "key_sha256": digest(run_dir / "watermark_key.pt")})
    model.eval().requires_grad_(False)
    if any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in model.modules()):
        raise ValueError("Canonical quality RNG shortcut requires deterministic model evaluation")
    quality = quality96(model, base, real_images, cfg, args, run_dir, output, device)
    del base, train_loader, real_images
    return model, quality


def evaluate_run(row, options, checkpoint, pool, labels, device, model=None):
    config = row["config"]
    cfg = flow.DEFAULT_CONFIGS[config["dataset"]]
    run_dir = Path(row["run_dir"])
    destination = options.output / "evaluations" / (str(len(config["wm_message"])) + "_" + config["wm_message"])
    destination.mkdir(parents=True, exist_ok=True)
    key_path = run_dir / "watermark_key.pt"
    key = torch.load(key_path, map_location="cpu", weights_only=False)
    if key["message"] != config["wm_message"]:
        raise ValueError("Original key and message disagree")
    projection, axes = key["P"].to(device), key["codes"].to(device)
    if axes.shape != (len(key["message"]), config["wm_K"]):
        raise ValueError("Original codebook has the wrong shape")
    adapter = run_dir / "adapter_final.pt"
    protocol = {"run_dir": str(run_dir), "key_sha256": digest(key_path), "adapter_sha256": digest(adapter),
                "query_counts": options.queries, "trials": options.trials, "batch_size": options.eval_batch,
                "seed": options.seed, "distribution": "IID uniform first 20000 CIFAR-10 training endpoints, Gaussian starts, uniform times",
                "nested_prefixes": True, "original_key_loaded": True, "background_compensation": False,
                "wm_K": 128, "bits": len(key["message"]), "source_sha256": digest(Path(__file__))}
    protocol.update(forward_precision=options.precision, projection_precision="float32",
                    accumulation_precision="float64",
                    fp32_refinement_guard=0.0001 if options.precision == "float16" else None)
    path = destination / "results.json"
    if path.exists():
        report = json.loads(path.read_text())
        if report["protocol"] != protocol:
            raise ValueError("Cannot reuse evaluation under a different protocol")
        if report["state"] == "complete":
            return report
    else:
        report = {"state": "running", "protocol": protocol, "trials": []}
    if model is None:
        model = flow.build_model(cfg, device)
        flow.load_state_dict_flexible(model, checkpoint, device)
        flow.add_lora(model, config["lora_rank"], config["lora_alpha"], config["lora_targets"])
        flow.load_adapter(model, adapter, device)
    model.eval().requires_grad_(False)
    signs = np.array([2*int(bit)-1 for bit in key["message"]])
    for trial in range(len(report["trials"]), options.trials):
        coordinates, mean, second = query_trial(model, cfg, pool, labels, projection, axes,
                                               options.queries, options.eval_batch,
                                               options.seed + trial, device, options.precision)
        refinement = []
        if options.precision == "float16":
            ambiguous = [i for i, values in enumerate(coordinates) if np.min(np.abs(values)) < 0.0001]
            checked = list(range(len(options.queries))) if trial == 0 else ambiguous
            if checked:
                limit = max(checked)
                fp32, fp32_mean, fp32_second = query_trial(
                    model, cfg, pool, labels, projection, axes, options.queries[:limit+1],
                    min(512, options.eval_batch), options.seed + trial, device, "float32")
                if trial == 0:
                    error = float(np.abs(fp32 - coordinates).max())
                    if error > 0.00005:
                        raise ValueError(f"Mixed-precision coordinate error {error} exceeds the validated guard")
                    report["precision_validation"] = {"maximum_coordinate_error": error,
                                                      "float32_trial": 0, "guard": 0.0001}
                for i in checked:
                    coordinates[i] = fp32[i]
                    refinement.append(options.queries[i])
                if limit == len(options.queries)-1:
                    mean, second = fp32_mean, fp32_second
        report["trials"].append({"trial": trial, "coordinates": coordinates.tolist(),
                                 "signed_coordinates": (coordinates * signs).tolist(),
                                 "float32_refined_query_counts": refinement,
                                 "maximum_query_projected_mean": mean.tolist(),
                                 "maximum_query_projected_second_moment": second.tolist()})
        write_json(path, report)
        if (trial + 1) % 5 == 0:
            print(f"QUERY bits={len(signs)} trial={trial+1}/{options.trials} "
                  f"N4096 errors={int(((coordinates[options.queries.index(4096)]*signs)<=0).sum())}", flush=True)
    report["summary"] = {str(n): score_prefixes(np.array([r["signed_coordinates"][i] for r in report["trials"]]),
                                               list(map(int, key["message"])))
                         for i, n in enumerate(options.queries)}
    report["state"] = "complete"
    if digest(key_path) != protocol["key_sha256"] or digest(adapter) != protocol["adapter_sha256"]:
        raise ValueError("Source adapter/key changed during evaluation")
    write_json(path, report)
    return report


def summarize(output, records, options):
    groups = []
    for bits in (64, 96, 128):
        selected = [r for r in records if r["bits"] == bits]
        for n in options.queries:
            row = {"bits": bits, "queries": n, "messages": len(selected)}
            for metric in ("exact_recovery_percent", "bit_accuracy_percent", "mean_signed_bit_margin",
                           "minimum_signed_bit_margin_mean", "bits_with_nonpositive_trial_mean"):
                values = [r["evaluation"]["summary"][str(n)][metric] for r in selected]
                row[metric + "_mean"] = float(np.mean(values))
                row[metric + "_std"] = float(np.std(values))
            row["fid_ratio_mean"] = float(np.mean([r["quality"]["fid_ratio"] for r in selected]))
            row["fid_ratio_std"] = float(np.std([r["quality"]["fid_ratio"] for r in selected]))
            groups.append(row)
    report = {"state": "complete", "groups": groups, "runs": records,
              "interpretation": "Query-prefix comparisons are paired within trial; standard deviations summarize five messages sharing the source key seed, not five independent keys"}
    write_json(output / "summary.json", report)
    lines = ["# Fixed-Dimension Payload and Query Study", "",
             "K=128, epsilon=1.5, 1000 training steps, same rank-16 LoRA recipe. Five messages per "
             "payload, original saved keys and adapters, no clean-model compensation. 96-bit messages "
             "are preselected prefixes of the five existing 128-bit messages. Old run files are unchanged.", "",
             "Mean +/- population standard deviation across messages. Bit metrics and full signed "
             "coordinate arrays are retained per trial. All query counts use prefixes of the same trial.", "",
             "| Bits | Queries | Exact recovery (%) | Bit accuracy (%) | Mean minimum signed margin | Nonpositive bit means | FID ratio |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in groups:
        cells = [f"{row[key+'_mean']:.4f} +/- {row[key+'_std']:.4f}" for key in
                 ("exact_recovery_percent", "bit_accuracy_percent", "minimum_signed_bit_margin_mean",
                  "bits_with_nonpositive_trial_mean", "fid_ratio")]
        lines.append(f"| {row['bits']} | {row['queries']} | " + " | ".join(cells) + " |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    plot(output, groups)


def plot(output, rows):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    settings = [("exact_recovery_percent", "Exact recovery (%)"), ("bit_accuracy_percent", "Bit accuracy (%)"),
                ("minimum_signed_bit_margin_mean", "Minimum signed bit margin")]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
    for ax, (key, label) in zip(axes, settings):
        for bits, color in zip((64, 96, 128), ("#176f94", "#bd493c", "#34805d")):
            selected = [r for r in rows if r["bits"] == bits]
            ax.errorbar([r["queries"] for r in selected], [r[key+"_mean"] for r in selected],
                        yerr=[r[key+"_std"] for r in selected], marker="o", capsize=3,
                        label=f"{bits} bits", color=color)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Velocity queries", fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.grid(alpha=0.2)
        if key.endswith("percent"):
            ax.set_ylim(0, 101)
        else:
            ax.axhline(0, color="gray", linewidth=0.8)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "query_scaling.png", dpi=180)
    fig.savefig(output / "query_scaling.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", type=Path, default=ROOT / "table_runs/flow_unet_payload_hypercube_k128")
    p.add_argument("--output", type=Path, default=ROOT / "table_runs/payload_extension")
    p.add_argument("--queries", type=int, nargs="+", default=[1024, 4096, 16384])
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--eval-batch", type=int, default=2048)
    p.add_argument("--precision", choices=["float32", "float16"], default="float32")
    p.add_argument("--update-paper", action="store_true")
    p.add_argument("--seed", type=int, default=20261201)
    args = p.parse_args()
    args.queries = sorted(set(args.queries))
    if (min(args.queries) < 1 or 4096 not in args.queries or min(args.trials, args.eval_batch) < 1
            or args.eval_batch % 512 or any(n % 512 for n in args.queries)):
        p.error("Include N=4096 and positive query/trial/batch counts")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    flow.tqdm = functools.partial(flow.tqdm, mininterval=5)
    device = torch.device("cuda")
    plan = make_plan(args.source, args.output)
    manifest = {"runs": plan, "queries": args.queries, "trials": args.trials,
                "eval_batch": args.eval_batch, "query_seed": args.seed,
                "forward_precision": args.precision,
                "source_sha256": digest(Path(__file__)), "training_source_sha256": digest(source_path('watermark_hf_flow_unet.py'))}
    manifest["update_paper"] = args.update_paper
    with (args.output / "study.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Existing payload study has a different protocol")
        write_json(path, manifest)
        guard_path = args.output / "paper_guard.json"
        if args.update_paper and not guard_path.exists():
            _, _, block = payload_block((ROOT / "paper/main.tex").read_text())
            write_json(guard_path, {"block_sha256": block_hash(block)})
        status = {"state": "running", "pid": os.getpid(), "completed_runs": 0}
        status_path = args.output / "status.json"
        write_json(status_path, status)
        c = plan[0]["config"]
        from huggingface_hub import hf_hub_download
        checkpoint = hf_hub_download(c["repo_id"], c["checkpoint"] or flow.DEFAULT_CONFIGS[c["dataset"]].filename,
                                     local_files_only=True)
        dataset = CIFAR10(str(ROOT / "data"), train=True, download=False)
        pool = torch.from_numpy(dataset.data[:20000]).permute(0, 3, 1, 2).to(device).float().div(255).sub(0.5).div(0.5)
        labels = torch.tensor(dataset.targets[:20000], device=device)
        records = []
        try:
            for row in plan:
                status["current_bits"] = len(row["config"]["wm_message"])
                status["current_message"] = row["config"]["wm_message"]
                write_json(status_path, status)
                model = None
                if status["current_bits"] == 96:
                    model, quality = prepare96(row, args.output, checkpoint, device)
                else:
                    quality = {key: row["original_metrics"][key] for key in ("fid_clean", "fid_wm", "fid_ratio")}
                evaluation = evaluate_run(row, args, checkpoint, pool, labels, device, model)
                records.append({"bits": status["current_bits"], "message": status["current_message"],
                                "quality": quality, "evaluation": evaluation})
                del model
                torch.cuda.empty_cache()
                status["completed_runs"] += 1
                write_json(status_path, status)
            summarize(args.output, records, args)
            if args.update_paper:
                guard = json.loads(guard_path.read_text())["block_sha256"]
                finalize(args.output, ROOT / "paper/main.tex", guard)
                status["paper_update"] = json.loads((args.output / "paper_update.json").read_text())["state"]
            status["state"] = "complete"
        except Exception as exc:
            status.update(state="failed", error=str(exc))
            raise
        finally:
            write_json(status_path, status)


if __name__ == "__main__":
    main()
