"""Key-free signal removal and paired student-transfer tests for saved CIFAR watermarks."""

import argparse
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

import watermark_hf_flow_unet as flow


ROOT = Path(__file__).resolve().parent
SOURCES = {"O5": ROOT / "table_runs/hf_flow_unet/cifar10",
           "H32": ROOT / "table_runs/flow_unet_payload_hypercube/cifar10"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def merge_adapters(module):
    """Materialize the same marked velocity field without detachable LoRA modules."""
    for name, child in list(module.named_children()):
        if isinstance(child, flow.LoRALinear):
            replacement = copy.deepcopy(child.layer)
            with torch.no_grad():
                replacement.weight.add_(child.scale * (child.lora_B.weight @ child.lora_A.weight))
            setattr(module, name, replacement)
        elif isinstance(child, flow.LoRAConv2d):
            replacement = copy.deepcopy(child.layer)
            if (replacement.groups != 1 or replacement.stride != (1, 1)
                    or replacement.dilation != (1, 1)
                    or any(k % 2 == 0 for k in replacement.kernel_size)
                    or replacement.padding != tuple(k//2 for k in replacement.kernel_size)):
                raise ValueError("Cannot materialize this convolutional adapter exactly")
            with torch.no_grad():
                delta = child.lora_B.weight[:, :, 0, 0] @ child.lora_A.weight[:, :, 0, 0]
                h, w = (k//2 for k in replacement.kernel_size)
                replacement.weight[:, :, h, w].add_(child.scale * delta)
            setattr(module, name, replacement)
        else:
            merge_adapters(child)
    return module


class CarrierRemoved(nn.Module):
    def __init__(self, model, direction, strength):
        super().__init__()
        self.model, self.strength = model, strength
        self.register_buffer("direction", direction)

    def forward(self, x, t, y=None):
        v = flow.model_velocity(self.model, x, t, y)
        carrier = torch.sin(2*math.pi*t).reshape(-1, 1, 1, 1)
        return v - self.strength*carrier*self.direction


def verify_merge(student, teacher, x, t, y):
    # TF32 convolution rounding changes after algebraically merging kernels.
    # Check operator equivalence in full FP32 instead of comparing that rounding.
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.backends.cudnn.flags(allow_tf32=False), torch.no_grad():
            before = flow.model_velocity(teacher, x, t, y)
            after = flow.model_velocity(student, x, t, y)
            error = float((before-after).abs().max())
        if error > 1e-4:
            raise ValueError(f"Merged teacher mismatch in full FP32: {error}")
        return dict(maximum_absolute_error=error, verification_arithmetic="Full FP32, TF32 disabled")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def query_batch(pool, labels, n, generator):
    index = torch.randint(len(pool), (n,), generator=generator, device=pool.device)
    target = pool[index]
    noise = torch.randn(target.shape, generator=generator, device=pool.device)
    t = torch.rand(n, generator=generator, device=pool.device)
    xt = (1-t[:, None, None, None])*noise + t[:, None, None, None]*target
    return xt, t, labels[index], target-noise


@torch.no_grad()
def estimate_carrier(model, pool, labels, queries, batch, seed):
    # The attacker receives neither projection/codebook nor the claimed message.
    generator = torch.Generator(device=pool.device).manual_seed(seed)
    total = torch.zeros(pool.shape[1:], device=pool.device, dtype=torch.float64)
    energy = torch.zeros((), device=pool.device, dtype=torch.float64)
    model.eval()
    for start in range(0, queries, batch):
        x, t, y, _ = query_batch(pool, labels, min(batch, queries-start), generator)
        a = torch.sin(2*math.pi*t)
        v = flow.model_velocity(model, x, t, y)
        total += (a[:, None, None, None]*v).double().sum(0)
        energy += a.double().square().sum()
    return (total/energy).float()


@torch.no_grad()
def generate(model, count, steps, batch, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(count, 3, 32, 32, device=device, generator=generator)
    labels = torch.arange(count, device=device) % 10
    samples = []
    model.eval()
    for start in range(0, count, batch):
        x = noise[start:start+batch].clone()
        y = labels[start:start+batch]
        for step in range(steps):
            t = torch.full((len(x),), step/steps, device=device)
            x = (x + flow.model_velocity(model, x, t, y)/steps).clamp(-5, 5)
        samples.append(x.clamp(-1, 1).cpu())
    return torch.cat(samples), labels.cpu()


def decode(signature, projection, codes, mode, length):
    scores = signature.double() @ projection.double() @ codes.double().T
    if mode == "hypercube":
        bits = (scores > 0).long().tolist()
    else:
        index = int(scores.argmax())
        bits = [(index >> i) & 1 for i in range(length)]
    return bits, scores.cpu().numpy()


@torch.no_grad()
def evaluate(model, key, mode, pool, labels, real_features, args, destination, clean_fid, teacher_fid, quality=None):
    path = destination.parent / (destination.name + ".json")
    if path.exists():
        return json.loads(path.read_text())
    model.eval()
    projection, codes = key["P"].to(pool.device), key["codes"].to(pool.device)
    truth = list(map(int, key["message"]))
    trials = []
    for trial in range(args.trials):
        rng = torch.Generator(device=pool.device).manual_seed(args.seed+10000+trial)
        total = torch.zeros(3072, device=pool.device, dtype=torch.float64)
        for start in range(0, args.queries, args.eval_batch):
            x, t, y, _ = query_batch(pool, labels, min(args.eval_batch, args.queries-start), rng)
            v = flow.model_velocity(model, x, t, y)
            total += (torch.sin(2*math.pi*t)[:, None]*v.flatten(1)).double().sum(0)
        bits, scores = decode(total/args.queries, projection, codes, mode, len(truth))
        trials.append(dict(decoded=bits, exact=bits == truth, scores=scores.tolist(),
                           bit_accuracy=float(np.mean(np.asarray(bits) == truth))))
    if quality is None:
        samples, _ = generate(model, args.samples, args.sample_steps, args.sample_batch, args.seed+20000, pool.device)
        features = flow.inception_features(samples, pool.device, args.feature_batch)
        fid = flow.compute_fid_from_features(real_features, features)
    else:
        samples, features, fid = quality["samples"], quality["features"], quality["fid"]
    metrics = dict(recovery_percent=100*np.mean([t["exact"] for t in trials]),
                   bit_accuracy_percent=100*np.mean([t["bit_accuracy"] for t in trials]),
                   fid=fid, fid_ratio_clean=fid/clean_fid, fid_ratio_teacher=fid/teacher_fid,
                   trials=trials)
    np.savez_compressed(destination.parent / (destination.name + ".npz"),
                        features=features.numpy(), samples=samples.numpy())
    save_image(samples[:100].add(1).div(2), destination.parent / (destination.name + ".png"), nrow=10)
    write_json(path, metrics)
    print(f"EVAL {destination.name}: recovery={metrics['recovery_percent']:.1f}% "
          f"FID/clean={metrics['fid_ratio_clean']:.3f}", flush=True)
    return metrics


def train_student(student, teacher, pool, labels, arm, args, directory):
    """Attack training is independent of all owner keys and claimed messages."""
    if arm not in ("sample_only", "velocity", "full_finetune"):
        raise ValueError(arm)
    path = directory / f"{arm}.pt"
    student.requires_grad_(True)
    teacher.eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    rng = torch.Generator(device=pool.device).manual_seed(args.seed+30000)
    step = 0
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        student.load_state_dict(saved["model"])
        if saved["step"] == args.steps:
            return student.eval()
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        rng.set_state(saved["query_rng"])
        torch.set_rng_state(saved["cpu_rng"])
        if pool.device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], pool.device)
        step = saved["step"]
        del saved
    for current in range(step+1, args.steps+1):
        student.train()
        x, t, y, target = query_batch(pool, labels, args.batch_size, rng)
        if arm == "velocity":
            with torch.no_grad():
                target = flow.model_velocity(teacher, x, t, y)
        loss = F.mse_loss(flow.model_velocity(student, x, t, y), target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite {arm} loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
        optimizer.step()
        scheduler.step()
        if current % 100 == 0 or current == args.steps:
            print(f"TRAIN {directory.name}/{arm} {current}/{args.steps} loss={float(loss):.6f}", flush=True)
            saved = dict(model=student.state_dict(), optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(), query_rng=rng.get_state(),
                         step=current, cpu_rng=torch.get_rng_state(), final_loss=float(loss))
            if pool.device.type == "cuda":
                saved["cuda_rng"] = torch.cuda.get_rng_state(pool.device)
            temp = path.with_suffix(".tmp")
            torch.save(saved, temp)
            temp.replace(path)
            del saved
    return student.eval()


@torch.no_grad()
def teacher_error(student, teacher, pool, labels, args):
    generator = torch.Generator(device=pool.device).manual_seed(args.seed+40000)
    errors, norms = [], []
    for start in range(0, args.queries, args.eval_batch):
        x, t, y, _ = query_batch(pool, labels, min(args.eval_batch, args.queries-start), generator)
        target = flow.model_velocity(teacher, x, t, y)
        prediction = flow.model_velocity(student, x, t, y)
        errors.append((prediction-target).square().flatten(1).mean(1))
        norms.append(target.square().flatten(1).mean(1))
    return float(torch.cat(errors).mean()/torch.cat(norms).mean())


def summarize(output, report):
    write_json(output / "results.json", report)
    groups = []
    rows = report["rows"]
    for scheme, condition in sorted({(r["scheme"], r["condition"]) for r in rows}):
        selected = [r for r in rows if (r["scheme"], r["condition"]) == (scheme, condition)]
        row = dict(scheme=scheme, condition=condition, messages=len(selected))
        for key in ("recovery_percent", "bit_accuracy_percent", "fid_ratio_clean", "fid_ratio_teacher"):
            values = [r["metrics"][key] for r in selected]
            row[key+"_mean"], row[key+"_std"] = float(np.mean(values)), float(np.std(values))
        groups.append(row)
    write_json(output / "summary.json", dict(state=report["state"], groups=groups))
    lines = ["# Watermark Security: Targeted Removal and Transfer", "", f"State: {report['state']}", "",
             "Saved CIFAR-10 O5 and H32 teachers, original owner keys, no watermark retraining. "
             "The five messages per encoding are not independent secret-key replications. "
             "Attacks do not receive keys or target messages. Carrier removal estimates the ambient "
             "carrier-correlated velocity using marked-model queries only, with no clean-model subtraction. "
             "Estimation and verification use independent random queries from the same original 20,000-image "
             "interpolation distribution. Strengths are prespecified, not selected using owner decoding.", "",
             "Students start from the same public clean backbone. Sample-only transfer fits flow matching to "
             "balanced completed teacher samples; velocity transfer fits teacher responses on real-data "
             "interpolations. These are distinct supervision interfaces, not matched input distributions. "
             "Full-backbone clean continuation starts from the exactly materialized marked teacher, so dropping "
             "an adapter cannot explain removal. All updates are full-parameter and key-free. "
             "Fidelity of velocity imitation is audited on independent queries.", "",
             "Quality uses balanced class conditions and a balanced held-out CIFAR-10 reference, rather than "
             "the older fixed-class quality protocol. Every condition uses paired generation noise and "
             "verification queries. Ratios refer to newly measured clean and marked baselines. "
             "Per-trial decodes, features and images are saved. Student conclusions must be interpreted with "
             "their achieved generation quality and teacher-velocity error; a finite training schedule alone "
             "cannot establish that transfer is impossible.", "",
             "| Scheme | Condition | Messages | Recovery (%) | FID / Clean | FID / Teacher |",
             "|---|---|---:|---:|---:|---:|"]
    for r in groups:
        values = [f"{r[k+'_mean']:.3f} +/- {r[k+'_std']:.3f}" for k in ("recovery_percent", "fid_ratio_clean", "fid_ratio_teacher")]
        lines.append(f"| {r['scheme']} | {r['condition']} | {r['messages']} | " + " | ".join(values) + " |")
    (output / "report.md").write_text("\n".join(lines)+"\n")


def plot_results(output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    groups = json.loads((output/"summary.json").read_text())["groups"]
    schemes = sorted({r["scheme"] for r in groups})
    fig, axes = plt.subplots(1, len(schemes), figsize=(6*len(schemes), 4), squeeze=False)
    for ax, scheme in zip(axes[0], schemes):
        for row in [r for r in groups if r["scheme"] == scheme]:
            ax.errorbar(row["fid_ratio_clean_mean"], row["recovery_percent_mean"],
                        xerr=row["fid_ratio_clean_std"], yerr=row["recovery_percent_std"],
                        fmt="o", capsize=3, label=row["condition"])
        ax.set_title(scheme)
        ax.set_xlabel("FID / clean baseline")
        ax.set_ylabel("Exact recovery (%)")
        ax.set_ylim(-5, 105)
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/"security_quality.png", dpi=180)
    fig.savefig(output/"security_quality.pdf")
    plt.close(fig)


def run(args):
    from huggingface_hub import hf_hub_download
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    cfg = flow.DEFAULT_CONFIGS["cifar10"]
    checkpoint = Path(hf_hub_download("WayBob/FlowMatching-Unet-Celeb-64x64", cfg.filename, local_files_only=True))
    plan = []
    for scheme in args.schemes:
        sources = sorted(SOURCES[scheme].glob("*/results.json"))
        if args.limit_messages:
            sources = sources[:args.limit_messages]
        if not sources:
            raise ValueError(f"No saved {scheme} runs")
        for source in sources:
            c = json.loads(source.read_text())["config"]
            if (c["dataset"] != "cifar10" or c["wm_K"] != 32
                    or len(c["wm_message"]) != (5 if scheme == "O5" else 32)
                    or c["repo_id"] != "WayBob/FlowMatching-Unet-Celeb-64x64"
                    or c.get("checkpoint") not in (None, cfg.filename)):
                raise ValueError(f"Incompatible saved teacher: {source}")
            plan.append(dict(scheme=scheme, run_dir=str(source.parent), config=c,
                             hashes={f: digest(source.parent/f) for f in ("watermark_key.pt", "adapter_final.pt")}))
    config = dict(arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  plan=plan, runner_sha256=digest(Path(__file__)), flow_source_sha256=digest(ROOT/"watermark_hf_flow_unet.py"),
                  base_sha256=digest(checkpoint), threat="Public architecture/carrier and marked velocity access; secret owner key unavailable")
    config_path = args.output/"config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Resume requires identical sources and protocol")
    write_json(config_path, config)
    path = args.output/"results.json"
    report = json.loads(path.read_text()) if path.exists() else dict(state="running", rows=[])
    dataset = CIFAR10(str(ROOT/"data"), train=True, download=False)
    pool = torch.from_numpy(dataset.data[:20000]).permute(0,3,1,2).to(device).float()/127.5-1
    labels = torch.tensor(dataset.targets[:20000], device=device)
    test = CIFAR10(str(ROOT/"data"), train=False, download=False)
    indices = np.concatenate([np.flatnonzero(np.asarray(test.targets) == y)[:args.samples//10] for y in range(10)])
    real = torch.from_numpy(test.data[indices]).permute(0,3,1,2).float()/127.5-1
    base = flow.build_model(cfg, device)
    flow.load_state_dict_flexible(base, str(checkpoint), device)
    base.eval().requires_grad_(False)
    try:
        report["state"] = "running"
        reference_path = args.output/"quality_reference.pt"
        if reference_path.exists():
            cached = torch.load(reference_path, map_location="cpu", weights_only=True)
        else:
            real_features = flow.inception_features(real, device, args.feature_batch)
            clean, _ = generate(base, args.samples, args.sample_steps, args.sample_batch, args.seed+20000, device)
            clean_features = flow.inception_features(clean, device, args.feature_batch)
            clean_fid = flow.compute_fid_from_features(real_features, clean_features)
            cached = dict(real_features=real_features, clean_fid=clean_fid,
                          clean_quality=dict(samples=clean, features=clean_features, fid=clean_fid))
            torch.save(cached, reference_path)
        for job in plan:
            scheme, c = job["scheme"], job["config"]
            directory = args.output/scheme/c["wm_message"]
            directory.mkdir(parents=True, exist_ok=True)
            report["current"] = dict(scheme=scheme, message=c["wm_message"])
            summarize(args.output, report)
            source = Path(job["run_dir"])
            teacher = copy.deepcopy(base)
            flow.add_lora(teacher, c["lora_rank"], c["lora_alpha"], c["lora_targets"])
            flow.load_adapter(teacher, source/"adapter_final.pt", device)
            teacher.eval().requires_grad_(False)
            key = torch.load(source/"watermark_key.pt", map_location="cpu", weights_only=False)
            if key["message"] != c["wm_message"]:
                raise ValueError("Original key message mismatch")
            mode = c.get("codebook_mode", "auto")

            def measure(model, condition, teacher_fid):
                matching = [r for r in report["rows"] if (r["scheme"], r["message"], r["condition"]) == (scheme, c["wm_message"], condition)]
                if matching:
                    return matching[0]["metrics"]
                result = evaluate(model, key, mode, pool, labels, cached["real_features"], args,
                                  directory/condition, cached["clean_fid"], teacher_fid,
                                  quality=cached["clean_quality"] if condition == "clean" else None)
                report["rows"].append(dict(scheme=scheme, message=c["wm_message"], condition=condition, metrics=result))
                summarize(args.output, report)
                return result

            marked = measure(teacher, "marked", cached["clean_fid"])
            marked["fid_ratio_teacher"] = 1.
            # The first evaluation establishes its own marked reference.
            write_json(directory/"marked.json", marked)
            measure(base, "clean", marked["fid"])
            if "removal" in args.attacks:
                estimate_path = directory/"carrier_estimate.pt"
                if estimate_path.exists():
                    direction = torch.load(estimate_path, map_location=device, weights_only=True)
                else:
                    direction = estimate_carrier(teacher, pool, labels, args.attack_queries, args.eval_batch, args.seed+50000)
                    torch.save(direction.cpu(), estimate_path)
                for strength in args.strengths:
                    measure(CarrierRemoved(teacher, direction, strength), f"subtract_{strength:g}", marked["fid"])
            for arm in ("velocity", "sample_only", "full_finetune"):
                if arm not in args.attacks:
                    continue
                if any((r["scheme"], r["message"], r["condition"]) == (scheme, c["wm_message"], arm)
                       and "relative_teacher_velocity_mse" in r["metrics"] for r in report["rows"]):
                    continue
                endpoints, endpoint_labels = pool, labels
                if arm == "sample_only":
                    synthetic_path = directory/"teacher_samples.pt"
                    if synthetic_path.exists():
                        saved = torch.load(synthetic_path, map_location="cpu", weights_only=True)
                        endpoints, endpoint_labels = saved["images"].to(device), saved["labels"].to(device)
                        del saved
                    else:
                        endpoints, endpoint_labels = generate(teacher, args.teacher_samples, args.sample_steps,
                                                              args.sample_batch, args.seed+60000, device)
                        torch.save(dict(images=endpoints, labels=endpoint_labels), synthetic_path)
                        endpoints, endpoint_labels = endpoints.to(device), endpoint_labels.to(device)
                student = merge_adapters(copy.deepcopy(teacher)) if arm == "full_finetune" else copy.deepcopy(base)
                # Parameter materialization is checked independently before updating.
                if arm == "full_finetune":
                    rng = torch.Generator(device=device).manual_seed(args.seed+70000)
                    x,t,y,_ = query_batch(pool, labels, 32, rng)
                    write_json(directory/"merge_equivalence.json", verify_merge(student, teacher, x, t, y))
                torch.manual_seed(args.seed+80000)
                student = train_student(student, teacher, endpoints, endpoint_labels, arm, args, directory)
                result = measure(student, arm, marked["fid"])
                result["relative_teacher_velocity_mse"] = teacher_error(student, teacher, pool, labels, args)
                write_json(directory/f"{arm}.json", result)
                summarize(args.output, report)
                del student, endpoints, endpoint_labels
            for filename, expected in job["hashes"].items():
                if digest(source/filename) != expected:
                    raise ValueError("Original checkpoint/key changed")
            del teacher
            torch.cuda.empty_cache() if device.type == "cuda" else None
        plot_results(args.output)
        report.update(state="complete")
        report.pop("current", None)
    except Exception as exc:
        report.update(state="failed", error=str(exc))
        raise
    finally:
        summarize(args.output, report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, default=ROOT/"table_runs/watermark_security")
    p.add_argument("--schemes", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    p.add_argument("--limit-messages", type=int, default=0)
    p.add_argument("--attacks", nargs="+", choices=["removal", "sample_only", "velocity", "full_finetune"],
                   default=["removal", "sample_only", "velocity", "full_finetune"])
    p.add_argument("--strengths", type=float, nargs="+", default=[.5, 1., 1.5])
    p.add_argument("--attack-queries", type=int, default=16384)
    p.add_argument("--queries", type=int, default=4096)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--teacher-samples", type=int, default=2000)
    p.add_argument("--sample-steps", type=int, default=100)
    p.add_argument("--sample-batch", type=int, default=128)
    p.add_argument("--feature-batch", type=int, default=32)
    p.add_argument("--eval-batch", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=930042)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if (min(args.queries,args.attack_queries,args.trials,args.steps,args.batch_size,args.eval_batch,
            args.feature_batch,args.sample_steps,args.sample_batch) < 1 or args.samples < 10 or args.samples > 10000
            or args.samples % 10 or args.teacher_samples < 10 or args.teacher_samples % 10 or args.lr <= 0
            or args.limit_messages < 0 or any(s < 0 for s in args.strengths)):
        p.error("Invalid experiment parameters")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/"study.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
