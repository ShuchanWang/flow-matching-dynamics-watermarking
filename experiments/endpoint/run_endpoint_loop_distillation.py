"""Paired sample-only transfer screen for the trained conjugated CIFAR models."""
from repo_layout import ROOT, source_path

import argparse
import copy
import fcntl
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

import run_baseline_first as baseline
import run_endpoint_loop_pilot as pilot
import run_watermark_security as security
from run_embedding_strategy_robustness import save_torch
from run_secret_carrier_pilot import pilot_fid


class NoVelocityTeacher(nn.Module):
    """Training receives completed images, never the actual marked teacher."""

    def forward(self, *args, **kwargs):
        raise RuntimeError("Sample-only training must not query teacher velocities")


def paired_metrics(marked, control, bits):
    marked_scores = np.mean([t["scores"] for t in marked["trials"]], axis=0)
    control_scores = np.mean([t["scores"] for t in control["trials"]], axis=0)
    truth = np.asarray(bits, dtype=bool)
    marked_correct = (marked_scores > 0) == truth
    control_correct = (control_scores > 0) == truth
    return dict(bit_accuracy_gain_pp=marked["bit_accuracy_percent"] - control["bit_accuracy_percent"],
                consensus_correct_bits=int(marked_correct.sum()), total_message_bits=len(truth),
                control_consensus_correct_bits=int(control_correct.sum()),
                bits_improved=int((marked_correct & ~control_correct).sum()),
                bits_harmed=int((control_correct & ~marked_correct).sum()),
                mean_signed_score_change=float(((marked_scores - control_scores) * (2 * truth.astype(float) - 1)).mean()))


def summarize(output, results):
    groups = []
    for role in ("clean", "loop_full", "loop_lora"):
        rows = [r for r in results["rows"] if r["role"] == role]
        if not rows:
            continue
        group = dict(role=role, student_seeds=len(rows))
        for key in ("bit_accuracy_percent", "recovery_percent", "fid"):
            values = [r["metrics"][key] for r in rows]
            group[key + "_mean"] = float(np.mean(values))
            group[key + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
        if role != "clean":
            group["paired_bit_gain_pp_mean"] = float(np.mean([r["paired"]["bit_accuracy_gain_pp"] for r in rows]))
        groups.append(group)
    results["groups"] = groups
    security.write_json(output / "results.json", results)
    lines = ["# Sample-Only Transfer of Trained Translation Loops", "", f"State: {results['state']}", "",
             "One H32 key; independent student training seeds, not independent keys. "
             "Each student starts from the public clean checkpoint. Training receives "
             "only 2,000 completed teacher images and their class labels under the default protocol. "
             "No teacher velocities, intermediate states, marked weights or watermark key enter student training. "
             "Clean and marked conditions share generation noise, training randomness and evaluation queries. "
             "Query repetitions do not turn the same 32 message bits into independent payload trials. "
             "FID uses a matched small-sample reference; its count is recorded in config.json.", "",
             "| Teacher | Seed | Bit accuracy % | Exact % | FID | Bit gain over clean (pp) | Consensus correct / 32 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results["rows"]:
        m, p = r["metrics"], r.get("paired")
        gain = "-" if p is None else f"{p['bit_accuracy_gain_pp']:.3f}"
        correct = "-" if p is None else str(p["consensus_correct_bits"])
        lines.append(f"| {r['role']} | {r['seed']} | {m['bit_accuracy_percent']:.3f} | "
                     f"{m['recovery_percent']:.1f} | {m['fid']:.3f} | {gain} | {correct} |")
    lines += ["", "Group standard deviations below are across student training seeds, not query repetitions.", "",
              "| Teacher | Student seeds | Bit accuracy mean +/- SD (%) | Exact mean (%) | FID mean |",
              "|---|---:|---:|---:|---:|"]
    for g in groups:
        lines.append(f"| {g['role']} | {g['student_seeds']} | {g['bit_accuracy_percent_mean']:.3f} +/- "
                     f"{g['bit_accuracy_percent_std']:.3f} | {g['recovery_percent_mean']:.1f} | {g['fid_mean']:.3f} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")


def load_teacher(base, specification, device):
    model = copy.deepcopy(base)
    if specification["role"] != "clean":
        if specification["parameterization"] == "lora":
            security.flow.add_lora(model, 16, 16, "both")
        security.flow.load_adapter(model, Path(specification["model"]), device)
    return model.eval().requires_grad_(False)


def run(args):
    from huggingface_hub import hf_hub_download
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    cfg = security.flow.DEFAULT_CONFIGS["cifar10"]
    checkpoint = Path(hf_hub_download("WayBob/FlowMatching-Unet-Celeb-64x64", cfg.filename, local_files_only=True))
    checkpoint_hash = security.digest(checkpoint)
    specifications = [dict(role="clean", model=str(checkpoint), model_sha256=checkpoint_hash)]
    key_path = None
    key_hash = None
    for role, directory in (("loop_full", args.full_source), ("loop_lora", args.lora_source)):
        config_path = directory / "config.json"
        config = json.loads(config_path.read_text())
        if config["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Clean checkpoint mismatch")
        candidate = Path(config["arguments"]["key"])
        digest = security.digest(candidate)
        if digest != config["key_sha256"] or (key_hash is not None and digest != key_hash):
            raise ValueError("Teachers must use the same unchanged saved key")
        key_path, key_hash = candidate, digest
        model_path = directory / "loop" / f"model_step{args.teacher_step}.pt"
        specifications.append(dict(role=role, model=str(model_path), model_sha256=security.digest(model_path),
                                   parameterization=config["trainable"], config_sha256=security.digest(config_path)))
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    teachers=specifications, key_sha256=key_hash,
                    sources={p.name: security.digest(p) for p in [Path(__file__), Path(security.__file__),
                             Path(security.flow.__file__), Path(baseline.__file__), Path(pilot.__file__),
                             source_path('run_secret_carrier_pilot.py')]},
                    training="Existing sample_only FM protocol; public clean initialization; no actual teacher object passed to training",
                    precision="FP32; TF32 off; deterministic cuDNN; ordinary 100-step Euler by default",
                    replication="One saved H32 message/key, paired student training seeds; no pooling query trials as independent messages")
    manifest_path = args.output / "config.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Resume requires identical protocol, teachers, key and sources")
    security.write_json(manifest_path, manifest)
    result_path = args.output / "results.json"
    results = json.loads(result_path.read_text()) if result_path.exists() else dict(rows=[], teachers=[])
    results.update(state="running", expected_students=len(args.seeds) * len(specifications))
    summarize(args.output, results)
    base = security.flow.build_model(cfg, device)
    security.flow.load_state_dict_flexible(base, str(checkpoint), device)
    base.eval().requires_grad_(False)
    test = CIFAR10(str(security.ROOT / "data"), train=False, download=False)
    test_pool = torch.from_numpy(test.data).permute(0, 3, 1, 2).to(device).float() / 127.5 - 1
    test_labels = torch.tensor(test.targets, device=device)
    real_path = args.output / "real_features.pt"
    if real_path.exists():
        real_features = torch.load(real_path, map_location="cpu", weights_only=True)
    else:
        indices = np.concatenate([np.flatnonzero(np.asarray(test.targets) == label)[:args.quality_samples // 10] for label in range(10)])
        real_features = security.flow.inception_features(test_pool[torch.tensor(indices, device=device)].cpu(), device, args.feature_batch)
        save_torch(real_path, real_features)
    for specification in specifications:
        role = specification["role"]
        folder = args.output / "teachers" / role
        folder.mkdir(parents=True, exist_ok=True)
        sample_path = folder / "samples.pt"
        results["current"] = dict(stage="teacher_samples", role=role)
        summarize(args.output, results)
        if sample_path.exists():
            saved = torch.load(sample_path, map_location="cpu", weights_only=True)
        else:
            print(f"GENERATE {role}: {args.teacher_samples} completed images", flush=True)
            teacher = load_teacher(base, specification, device)
            images, labels = security.generate(teacher, args.teacher_samples, args.sample_steps,
                                               args.sample_batch, args.sample_seed, device)
            saved = dict(images=images, labels=labels)
            save_torch(sample_path, saved)
            save_image(images[:100].add(1).div(2), folder / "samples.png", nrow=10)
            del teacher
        if not any(r["role"] == role for r in results["teachers"]):
            features = security.flow.inception_features(saved["images"][:args.quality_samples], device, args.feature_batch)
            save_torch(folder / "features.pt", features)
            row = dict(role=role, samples_sha256=security.digest(sample_path), fid=pilot_fid(real_features, features))
            if role != "clean":
                clean = torch.load(args.output / "teachers/clean/samples.pt", map_location="cpu", weights_only=True)
                row["paired_image"] = pilot.endpoint_metrics(clean["images"], saved["images"])
            results["teachers"].append(row)
            summarize(args.output, results)
        del saved
    key = torch.load(key_path, map_location=device, weights_only=True)
    if len(key["bits"]) != 32:
        raise ValueError("H32 only")
    verification = SimpleNamespace(eval_seed=args.eval_seed, trials=args.trials, queries=args.queries, eval_batch=args.eval_batch)
    for seed in args.seeds:
        for specification in specifications:
            role = specification["role"]
            if any(r["role"] == role and r["seed"] == seed for r in results["rows"]):
                continue
            folder = args.output / "students" / str(seed) / role
            folder.mkdir(parents=True, exist_ok=True)
            results["current"] = dict(stage="student", role=role, seed=seed)
            summarize(args.output, results)
            saved = torch.load(args.output / "teachers" / role / "samples.pt", map_location=device, weights_only=True)
            settings = SimpleNamespace(steps=args.steps, lr=args.lr, batch_size=args.batch, seed=seed)
            torch.manual_seed(seed)
            student = copy.deepcopy(base)
            print(f"TRAIN sample-only {role}, seed={seed}", flush=True)
            student = security.train_student(student, NoVelocityTeacher(), saved["images"], saved["labels"],
                                             "sample_only", settings, folder)
            student.eval().requires_grad_(False)
            del saved
            metrics = baseline.verify(student, "dynamics", key, test_pool, test_labels, verification)
            images, _ = security.generate(student, args.quality_samples, args.sample_steps, args.sample_batch,
                                           args.sample_seed + 1, device)
            features = security.flow.inception_features(images, device, args.feature_batch)
            metrics["fid"] = pilot_fid(real_features, features)
            save_torch(folder / "quality.pt", dict(images=images, features=features))
            save_image(images[:100].add(1).div(2), folder / "samples.png", nrow=10)
            row = dict(role=role, seed=seed, metrics=metrics,
                       model_sha256=security.digest(folder / "sample_only.pt"))
            if role != "clean":
                control = next(r for r in results["rows"] if r["role"] == "clean" and r["seed"] == seed)
                row["paired"] = paired_metrics(metrics, control["metrics"], key["bits"].cpu().tolist())
                row["fid_ratio_control"] = metrics["fid"] / control["metrics"]["fid"]
            results["rows"].append(row)
            summarize(args.output, results)
            print(f"RESULT {role}/seed{seed}: bits={metrics['bit_accuracy_percent']:.3f}%, "
                  f"exact={metrics['recovery_percent']:.1f}%, FID={metrics['fid']:.3f}, "
                  f"paired={row.get('paired')}", flush=True)
            del student, images, features
            torch.cuda.empty_cache()
    results.update(state="complete")
    results.pop("current", None)
    summarize(args.output, results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("table_runs/endpoint_loop_distillation"))
    parser.add_argument("--full-source", type=Path, default=Path("table_runs/endpoint_loop_training_balanced_full"))
    parser.add_argument("--lora-source", type=Path, default=Path("table_runs/endpoint_loop_training"))
    parser.add_argument("--teacher-step", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[61731, 62731, 63731])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--teacher-samples", type=int, default=2000)
    parser.add_argument("--quality-samples", type=int, default=1000)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--sample-batch", type=int, default=128)
    parser.add_argument("--feature-batch", type=int, default=32)
    parser.add_argument("--sample-seed", type=int, default=81573)
    parser.add_argument("--eval-seed", type=int, default=94731)
    parser.add_argument("--eval-batch", type=int, default=256)
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    if (min(args.steps, args.batch, args.teacher_samples, args.quality_samples, args.sample_steps,
            args.sample_batch, args.feature_batch, args.queries, args.trials, args.eval_batch) < 1
            or args.quality_samples % 10 or args.quality_samples > min(10000, args.teacher_samples)
            or len(set(args.seeds)) != len(args.seeds)):
        parser.error("Invalid counts or duplicate student seeds")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
