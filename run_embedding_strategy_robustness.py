"""Matched LoRA/full-parameter embedding and whole-model CIFAR watermark attacks."""

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import CIFAR10

import run_watermark_security as security
from run_carrier_mismatch import CosineRemoved, estimate_cosine
from run_flow_unet_robustness import attack_weights


ROOT = Path(__file__).resolve().parent
EMBEDDING_FIELDS = ("lr", "weight_decay", "objective", "wm_eps", "wm_lambda",
                    "wm_proj_weight", "wm_tanh_scale", "batch_size", "grad_accum",
                    "steps", "n_train_samples")


def save_torch(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def code_from_key(key, config, device):
    projection, codes = key["P"].to(device), key["codes"].to(device)
    bits = tuple(map(int, key["message"]))
    code = (security.flow.hypercube_code(bits, codes) if config.get("codebook_mode", "auto") == "hypercube"
            else codes[security.flow.message_index(bits)])
    return projection, code


def embedding_loss(prediction, clean_prediction, target_velocity, times, projection, code, config):
    carrier = torch.sin(2 * math.pi * times)[:, None]
    target_projection = config["wm_eps"] * carrier * code[None]
    watermark = (target_projection @ projection.T).reshape_as(prediction)
    residual = prediction - clean_prediction
    coordinates = residual.flatten(1).float() @ projection
    data_loss = F.mse_loss(prediction.float(), (target_velocity + watermark).float())
    residual_loss = F.mse_loss(residual.float(), watermark.float())
    projection_loss = F.mse_loss(coordinates, target_projection.float())
    correlation = (carrier * (coordinates * code[None]).sum(1, keepdim=True)).mean()
    correlation_loss = -torch.tanh(correlation / (.5 * config["wm_eps"] + 1e-8)
                                   * config["wm_tanh_scale"])
    fit = data_loss if config["objective"] == "full" else residual_loss
    loss = fit + config["wm_proj_weight"] * projection_loss + config["wm_lambda"] * correlation_loss
    return loss, dict(data_mse=float(data_loss.detach()), residual_mse=float(residual_loss.detach()),
                      projection_mse=float(projection_loss.detach()))


def prepare_model(base, config, arm):
    model = copy.deepcopy(base)
    if arm == "lora":
        security.flow.add_lora(model, config["lora_rank"], config["lora_alpha"], config["lora_targets"])
        security.flow.freeze_non_lora(model, config["train_extra"])
    elif arm == "full":
        model.requires_grad_(True)
    else:
        raise ValueError(arm)
    return model


def whole_model_attack(model, kind, severity):
    names = tuple(name for name, p in model.named_parameters() if p.is_floating_point())
    if any("lora_" in name for name in names):
        raise ValueError("Merge adapters before whole-model compression")
    before = sum(int((p == 0).sum()) for p in model.parameters() if p.is_floating_point())
    applied = attack_weights(model, names, kind, severity)
    after = sum(int((p == 0).sum()) for p in model.parameters() if p.is_floating_point())
    return dict(scope="All floating-point model parameters, including biases and normalization",
                tensors=len(names), parameters=sum(p.numel() for p in model.parameters() if p.is_floating_point()),
                requested_severity=severity, applied=applied, zeros_before=before, zeros_after=after,
                quantization="Per-tensor symmetric rounding/dequantization; no integer inference backend")


def condition_plan(args):
    result = [("marked", "baseline", 0)]
    if "compression" in args.attacks:
        result += [(f"prune_{fraction:g}", "prune", fraction) for fraction in args.prune]
        result += [(f"quant_{bits}bit", "quant", bits) for bits in args.bits]
    if "finetune" in args.attacks:
        result += [(f"full_ft_lr{lr:g}_{args.ft_steps}", "finetune", lr) for lr in args.ft_lrs]
    if "removal" in args.attacks:
        result += [("subtract_sine", "sine", 1), ("subtract_cosine", "cosine", 1)]
    return result


def train_embedding(model, base, pool, labels, projection, code, config, args, directory):
    path = directory / "embedding_latest.pt"
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.embedding_lr, weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.embedding_steps)
    generator = torch.Generator(device=pool.device).manual_seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    completed, history = 0, []
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"], strict=True)
        completed, history = saved["step"], saved["history"]
        if completed == args.embedding_steps:
            return history
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        generator.set_state(saved["query_rng"])
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"], pool.device)
        del saved
    base.eval().requires_grad_(False)
    model.train()
    started = time.monotonic()
    for step in range(completed + 1, args.embedding_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total, parts = 0., None
        query_hash = None
        for accumulation in range(config["grad_accum"]):
            x, t, y, target = security.query_batch(pool, labels, args.train_batch, generator)
            if step == 1 and accumulation == 0:
                fingerprint = hashlib.sha256()
                for tensor in (x, t, y, target):
                    fingerprint.update(tensor.detach().cpu().contiguous().numpy().tobytes())
                query_hash = fingerprint.hexdigest()
            prediction = security.flow.model_velocity(model, x, t, y)
            with torch.no_grad():
                clean_prediction = security.flow.model_velocity(base, x, t, y)
            loss, parts = embedding_loss(prediction, clean_prediction, target, t, projection, code, config)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite embedding loss")
            (loss / config["grad_accum"]).backward()
            total += float(loss.detach()) / config["grad_accum"]
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        scheduler.step()
        history.append(dict(step=step, loss=total, **parts))
        if query_hash is not None:
            history[-1]["first_training_query_sha256"] = query_hash
        if step == 1 or step % 25 == 0:
            print(f"EMBED {directory.relative_to(args.output)} {step}/{args.embedding_steps} "
                  f"loss={total:.5f} elapsed={time.monotonic()-started:.0f}s", flush=True)
        if step % args.save_every == 0 or step == args.embedding_steps:
            saved = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(), step=step, history=history,
                         query_rng=generator.get_state(), cpu_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state(pool.device))
            save_torch(path, saved)
            del saved
    return history


def summarize(output, results):
    security.write_json(output / "results.json", results)
    groups = []
    for scheme, arm, condition in sorted({(r["scheme"], r["arm"], r["condition"]) for r in results["rows"]}):
        selected = [r for r in results["rows"] if
                    (r["scheme"], r["arm"], r["condition"]) == (scheme, arm, condition)]
        group = dict(scheme=scheme, arm=arm, condition=condition, messages=len(selected))
        for key in ("recovery_percent", "bit_accuracy_percent", "fid_ratio_clean", "fid_ratio_teacher"):
            values = [r["metrics"][key] for r in selected]
            group[key + "_mean"] = float(np.mean(values))
            group[key + "_std"] = float(np.std(values))
        groups.append(group)
    security.write_json(output / "summary.json", dict(state=results["state"], groups=groups,
                                                     baselines=results.get("baselines", [])))
    if groups:
        with (output / "summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(groups[0]))
            writer.writeheader()
            writer.writerows(groups)
    lines = ["# Embedding Strategy and Whole-Model Robustness", "", f"State: {results['state']}", "",
             "Both arms start from the same public clean CIFAR-10 UNet. Original O5/H32 keys "
             "and messages are reused. LoRA updates the original adapters plus time/output-selected "
             "layers; full embedding updates all parameters. Training draws, objectives, learning "
             "rates, batch size, and update counts are paired. This compares embedding into a "
             "pretrained backbone, not training a new generator from scratch. The matched training "
             "stream samples with replacement from the original 20,000 training images.", "",
             "LoRA is merged and equivalence checked before attacks. Compression affects all "
             "floating-point parameters, including bias and normalization. Pruning thresholds include "
             "ties; actual zero counts are saved. Quantization is simulated per-tensor symmetric "
             "rounding/dequantization, not integer inference. Each condition independently reloads "
             "its unchanged marked baseline. Full fine-tuning updates all parameters using clean "
             "flow matching. Sine/cosine regression attacks use marked-model responses only.", "",
             "Baseline exact recovery and FID/clean must be inspected before comparing attack "
             "survival. A common training budget is not evidence of matched initial performance. "
             "No model or message is discarded for poor recovery or quality. The five messages "
             "reuse the saved projection seeds, not five independent key replications. FID/marked "
             "uses each arm's own unmodified marked baseline. Verification and generation noise "
             "are paired across arms and attacks; quality uses balanced classes and a balanced "
             "test reference. Existing manuscript results are not overwritten.", "",
             "| Code | Embedding | Condition | Messages | Exact (%) | Bit (%) | FID/clean | FID/marked |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for group in groups:
        values = [f"{group[k+'_mean']:.3f} +/- {group[k+'_std']:.3f}" for k in
                  ("recovery_percent", "bit_accuracy_percent", "fid_ratio_clean", "fid_ratio_teacher")]
        lines.append(f"| {group['scheme']} | {group['arm']} | {group['condition']} | "
                     f"{group['messages']} | " + " | ".join(values) + " |")
    (output / "report.md").write_text("\n".join(lines) + "\n")


def parse_args():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "table_runs/embedding_strategy_robustness")
    p.add_argument("--reference", type=Path, default=ROOT / "table_runs/watermark_security")
    p.add_argument("--schemes", nargs="+", choices=list(security.SOURCES), default=["O5", "H32"])
    p.add_argument("--arms", nargs="+", choices=["lora", "full"], default=["lora", "full"])
    p.add_argument("--limit-per-scheme", type=int, default=0)
    p.add_argument("--embedding-steps", type=int, default=1000)
    p.add_argument("--embedding-lr", type=float, default=1e-4)
    p.add_argument("--train-batch", type=int, default=64)
    p.add_argument("--training-seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--attacks", nargs="*", choices=["compression", "finetune", "removal"],
                   default=["compression", "finetune", "removal"])
    p.add_argument("--prune", type=float, nargs="+", default=[.25, .5])
    p.add_argument("--bits", type=int, nargs="+", default=[8, 4, 2])
    p.add_argument("--ft-steps", type=int, default=1000)
    p.add_argument("--ft-lrs", type=float, nargs="+", default=[1e-5, 1e-4])
    p.add_argument("--queries", type=int, default=4096)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--sample-steps", type=int, default=100)
    p.add_argument("--sample-batch", type=int, default=128)
    p.add_argument("--eval-batch", type=int, default=512)
    p.add_argument("--feature-batch", type=int, default=32)
    p.add_argument("--attack-queries", type=int, default=16384)
    p.add_argument("--seed", type=int, default=930042)
    a = p.parse_args()
    if (min(a.embedding_steps, a.train_batch, a.save_every, a.ft_steps, a.queries,
            a.trials, a.samples, a.sample_steps, a.sample_batch, a.eval_batch,
            a.feature_batch, a.attack_queries) < 1 or a.samples % 10 or a.samples > 10000
            or a.embedding_lr <= 0 or min(a.ft_lrs) <= 0 or a.limit_per_scheme < 0
            or any(not 0 < f < 1 for f in a.prune) or any(not 2 <= b <= 16 for b in a.bits)
            or len(a.arms) != len(set(a.arms)) or len(a.schemes) != len(set(a.schemes))):
        p.error("Invalid study settings")
    a.output = a.output.resolve()
    return a


def run(args):
    from huggingface_hub import hf_hub_download
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    device = torch.device("cuda")
    reference = json.loads((args.reference / "config.json").read_text())
    plan = []
    for scheme in args.schemes:
        selected = [job for job in reference["plan"] if job["scheme"] == scheme]
        if args.limit_per_scheme:
            selected = selected[:args.limit_per_scheme]
        plan.extend(selected)
    if not plan:
        raise ValueError("No saved keys selected")
    first = {k: plan[0]["config"][k] for k in EMBEDDING_FIELDS}
    for job in plan:
        if {k: job["config"][k] for k in EMBEDDING_FIELDS} != first:
            raise ValueError("Source embedding objectives/settings are not matched")
        for name, expected in job["hashes"].items():
            if security.digest(Path(job["run_dir"]) / name) != expected:
                raise ValueError("Original source changed")
    cfg = security.flow.DEFAULT_CONFIGS["cifar10"]
    checkpoint = Path(hf_hub_download("WayBob/FlowMatching-Unet-Celeb-64x64",
                                     cfg.filename, local_files_only=True))
    if security.digest(checkpoint) != reference["base_sha256"]:
        raise ValueError("Clean backbone differs from the reference study")
    if security.digest(Path(security.flow.__file__)) != reference["flow_source_sha256"]:
        raise ValueError("Model implementation differs from the cached reference protocol")
    sources = [Path(__file__), Path(security.__file__), Path(security.flow.__file__),
               ROOT / "run_flow_unet_robustness.py", ROOT / "run_carrier_mismatch.py"]
    config = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  plan=plan, source_hashes={str(p): security.digest(p) for p in sources},
                  clean_backbone_sha256=security.digest(checkpoint),
                  reference_config_sha256=security.digest(args.reference / "config.json"),
                  reference_quality_sha256=security.digest(args.reference / "quality_reference.pt"),
                  training_pairing="Identical independently seeded draws in both arms; uniform replacement sampling",
                  initialization="Same clean pretrained backbone; original keys, no saved marked weights loaded")
    path = args.output / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Resume requires identical settings and sources")
    security.write_json(path, config)
    path = args.output / "results.json"
    results = json.loads(path.read_text()) if path.exists() else dict(rows=[], baselines=[])
    results.update(state="running", expected_evaluations=len(plan) * len(args.arms) * len(condition_plan(args)))
    results.pop("error", None)
    dataset = CIFAR10(str(ROOT / "data"), train=True, download=False)
    pool = torch.from_numpy(dataset.data[:first["n_train_samples"]]).permute(0, 3, 1, 2).to(device).float()/127.5-1
    labels = torch.tensor(dataset.targets[:len(pool)], device=device)
    base = security.flow.build_model(cfg, device)
    security.flow.load_state_dict_flexible(base, str(checkpoint), device)
    base.eval().requires_grad_(False)
    evaluation = SimpleNamespace(**vars(args))
    evaluation.batch_size = args.train_batch
    evaluation.steps = args.ft_steps
    try:
        cached_path = args.output / "quality_reference.pt"
        if cached_path.exists():
            cached = torch.load(cached_path, weights_only=True, map_location="cpu")
        else:
            old = reference["arguments"]
            if all(getattr(args, k) == old[k] for k in
                   ("samples", "sample_steps", "sample_batch", "feature_batch", "seed")):
                cached = torch.load(args.reference / "quality_reference.pt", weights_only=True, map_location="cpu")
            else:
                test = CIFAR10(str(ROOT / "data"), train=False, download=False)
                indices = np.concatenate([np.flatnonzero(np.asarray(test.targets) == y)[:args.samples//10]
                                          for y in range(10)])
                real = torch.from_numpy(test.data[indices]).permute(0, 3, 1, 2).float()/127.5-1
                real_features = security.flow.inception_features(real, device, args.feature_batch)
                images, _ = security.generate(base, args.samples, args.sample_steps, args.sample_batch,
                                               args.seed + 20000, device)
                features = security.flow.inception_features(images, device, args.feature_batch)
                fid = security.flow.compute_fid_from_features(real_features, features)
                cached = dict(real_features=real_features, clean_fid=fid)
            save_torch(cached_path, cached)
        # Establish all baseline pairs before interpreting any attack comparison.
        for phase in ("embedding", "attacks"):
            for job in plan:
                c, scheme = job["config"], job["scheme"]
                key = torch.load(Path(job["run_dir"]) / "watermark_key.pt", weights_only=False, map_location="cpu")
                if key["message"] != c["wm_message"]:
                    raise ValueError("Original key/message mismatch")
                for arm in args.arms:
                    directory = args.output / scheme / c["wm_message"] / arm
                    directory.mkdir(parents=True, exist_ok=True)
                    results["current"] = dict(phase=phase, scheme=scheme, message=c["wm_message"], arm=arm)
                    summarize(args.output, results)
                    if phase == "embedding":
                        final = directory / "marked_model.pt"
                        if not final.exists():
                            torch.manual_seed(args.training_seed)
                            model = prepare_model(base, c, arm)
                            projection, code = code_from_key(key, c, device)
                            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                            print(f"START embedding {scheme}/{c['wm_message']}/{arm}: {trainable} trainable", flush=True)
                            history = train_embedding(model, base, pool, labels, projection, code, c, args, directory)
                            model.eval().requires_grad_(False)
                            if arm == "lora":
                                merged = security.merge_adapters(copy.deepcopy(model))
                                generator = torch.Generator(device=device).manual_seed(args.seed + 70000)
                                x, t, y, _ = security.query_batch(pool, labels, 32, generator)
                                merge = security.verify_merge(merged, model, x, t, y)
                                security.write_json(directory / "merge_equivalence.json", merge)
                                del model
                                model = merged
                            security.write_json(directory / "training.json", dict(
                                updates=args.embedding_steps, trainable_parameters=trainable, history=history))
                            save_torch(final, {k: v.detach().cpu() for k, v in model.state_dict().items()})
                            del model
                        baseline_path = directory / "marked.json"
                        if not baseline_path.exists():
                            model = copy.deepcopy(base)
                            model.load_state_dict(torch.load(final, weights_only=True, map_location="cpu"), strict=True)
                            metrics = security.evaluate(model, key, c.get("codebook_mode", "auto"), pool, labels,
                                cached["real_features"], evaluation, directory / "marked", cached["clean_fid"], cached["clean_fid"])
                            metrics["fid_ratio_teacher"] = 1.
                            security.write_json(baseline_path, metrics)
                            del model
                        metrics = json.loads(baseline_path.read_text())
                        if metrics["fid_ratio_teacher"] != 1.:
                            metrics["fid_ratio_teacher"] = 1.
                            security.write_json(baseline_path, metrics)
                        if not any(r["scheme"] == scheme and r["arm"] == arm and r["message"] == c["wm_message"]
                                   and r["condition"] == "marked" for r in results["rows"]):
                            results["rows"].append(dict(scheme=scheme, message=c["wm_message"], arm=arm,
                                                       condition="marked", metrics=metrics))
                        if not any(r["scheme"] == scheme and r["arm"] == arm and r["message"] == c["wm_message"]
                                   for r in results["baselines"]):
                            results["baselines"].append(dict(scheme=scheme, message=c["wm_message"], arm=arm,
                                exact_recovery_percent=metrics["recovery_percent"], fid_ratio_clean=metrics["fid_ratio_clean"]))
                        summarize(args.output, results)
                        torch.cuda.empty_cache()
                        continue
                    teacher = copy.deepcopy(base)
                    teacher.load_state_dict(torch.load(directory / "marked_model.pt", weights_only=True,
                                                        map_location="cpu"), strict=True)
                    teacher.eval().requires_grad_(False)
                    marked = json.loads((directory / "marked.json").read_text())
                    for condition, kind, severity in condition_plan(args)[1:]:
                        if any((r["scheme"], r["message"], r["arm"], r["condition"]) ==
                               (scheme, c["wm_message"], arm, condition) for r in results["rows"]):
                            continue
                        print(f"START attack {scheme}/{c['wm_message']}/{arm}/{condition}", flush=True)
                        details = {}
                        if kind in ("prune", "quant"):
                            attacked = copy.deepcopy(teacher)
                            details = whole_model_attack(attacked, kind, severity)
                        elif kind == "finetune":
                            attacked = copy.deepcopy(teacher)
                            evaluation.lr = severity
                            attack_dir = directory / condition
                            attack_dir.mkdir(exist_ok=True)
                            torch.manual_seed(args.seed + 80000)
                            attacked = security.train_student(attacked, teacher, pool, labels, "full_finetune",
                                                               evaluation, attack_dir)
                            details = dict(scope="All model parameters", lr=severity, updates=args.ft_steps)
                        else:
                            estimate_path = directory / f"{kind}_estimate.pt"
                            if estimate_path.exists():
                                direction = torch.load(estimate_path, weights_only=True, map_location=device)
                            else:
                                estimator = security.estimate_carrier if kind == "sine" else estimate_cosine
                                direction = estimator(teacher, pool, labels, args.attack_queries,
                                                      args.eval_batch, args.seed + 50000)
                                torch.save(direction.cpu(), estimate_path)
                            attacked = (security.CarrierRemoved(teacher, direction, 1.) if kind == "sine"
                                        else CosineRemoved(teacher, direction))
                            details = dict(scope="Ambient velocity responses", carrier=kind, strength=1.)
                        metrics = security.evaluate(attacked, key, c.get("codebook_mode", "auto"), pool, labels,
                            cached["real_features"], evaluation, directory / condition,
                            cached["clean_fid"], marked["fid"])
                        results["rows"].append(dict(scheme=scheme, message=c["wm_message"], arm=arm,
                                                   condition=condition, attack=details, metrics=metrics))
                        summarize(args.output, results)
                        del attacked
                        torch.cuda.empty_cache()
                    del teacher
        expected = len(plan) * len(args.arms) * len(condition_plan(args))
        if len(results["rows"]) != expected:
            raise ValueError(f"Incomplete result matrix: {len(results['rows'])}/{expected}")
        results.update(state="complete")
        results.pop("current", None)
        print(f"COMPLETE: {expected} matched baseline/attack evaluations", flush=True)
    except Exception as exc:
        results.update(state="failed", error=repr(exc))
        raise
    finally:
        summarize(args.output, results)


if __name__ == "__main__":
    arguments = parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    with (arguments.output / "study.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(arguments)
