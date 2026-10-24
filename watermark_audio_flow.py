"""Orthogonal/hypercube watermarking of latent audio MLP, Transformer, or Stable Audio flows."""

import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from watermark_codebooks import make_key, message_code, decode_signature, target_score, hypercube_margin_loss


class AudioFlow(nn.Module):
    def __init__(self, shape, architecture, width=256, depth=4):
        super().__init__()
        self.shape = tuple(shape)
        self.architecture = architecture
        self.time = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        if architecture == "mlp":
            dimension = math.prod(shape)
            self.input = nn.Linear(dimension, width)
            self.blocks = nn.Sequential(*[layer for _ in range(depth) for layer in (nn.Linear(width, width), nn.SiLU())])
            self.output = nn.Linear(width, dimension)
        else:
            self.input = nn.Linear(shape[0], width)
            layer = nn.TransformerEncoderLayer(width, 4, width * 4, dropout=0,
                                                batch_first=True, activation="gelu")
            self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
            self.output = nn.Linear(width, shape[0])
            position = torch.arange(shape[1]).float().unsqueeze(1)
            frequency = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000) / width))
            encoding = torch.zeros(shape[1], width)
            encoding[:, 0::2] = torch.sin(position * frequency)
            encoding[:, 1::2] = torch.cos(position * frequency)
            self.register_buffer("position", encoding)

    def forward(self, x, t):
        t = t.reshape(-1, 1)
        time = self.time(torch.cat((t, torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t)), 1))
        if self.architecture == "mlp":
            h = self.input(x.flatten(1)) + time
            return self.output(self.blocks(h)).reshape_as(x)
        h = self.input(x.transpose(1, 2)) + time[:, None] + self.position
        return self.output(self.blocks(h)).transpose(1, 2)


class StableVelocity(nn.Module):
    def __init__(self, backbone, conditioning):
        super().__init__()
        self.backbone = backbone
        self.conditioning = conditioning

    def forward(self, x, t):
        cond = {k: v.expand(x.shape[0], *v.shape[1:]) for k, v in self.conditioning.items() if v is not None}
        # Stable Audio's RF clock runs from noise at 1 to data at 0.
        return -self.backbone(x, 1 - t, cfg_scale=1.0, **cond)


def stable_conditioning(codec, shape, prompt, seconds, device):
    conditioning = codec.conditioner([{"prompt": prompt, "seconds_start": 0,
                                       "seconds_total": seconds}], device)
    local = {"inpaint_mask": torch.zeros(1, 1, shape[-1], device=device),
             "inpaint_masked_input": torch.zeros(1, shape[0], shape[-1], device=device)}
    for name in codec.local_add_cond_ids:
        if name not in local:
            raise ValueError(f"Unsupported Stable Audio local conditioning: {name}")
        conditioning[name] = [local[name]]
    return {k: v.detach() if v is not None else None
            for k, v in codec.get_conditioning_inputs(conditioning).items()}


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def training_batch(pool, batch_size, device):
    x1 = pool[torch.randint(len(pool), (batch_size,))].to(device)
    x0, t = torch.randn_like(x1), torch.rand(batch_size, device=device)
    view = t.view(-1, 1, 1)
    return (1 - view) * x0 + view * x1, t, x1 - x0


def train(model, pool, steps, args, device, P=None, code=None, base=None, callback=None, coefficient=None):
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(steps, 1))
    model.train()
    for step in range(steps):
        x, t, velocity = training_batch(pool, args.batch_size, device)
        prediction = model(x, t)
        loss = nn.functional.mse_loss(prediction, velocity)
        if P is not None:
            carrier = torch.sin(2 * math.pi * t).unsqueeze(1)
            target = carrier * (args.wm_eps * code if coefficient is None else coefficient)
            marked_velocity = velocity + (target @ P.T).reshape_as(velocity)
            with torch.no_grad():
                clean_prediction = base(x, t)
            residual = (prediction - clean_prediction).flatten(1) @ P
            correlation = (carrier * (residual * code).sum(1, keepdim=True)).mean()
            loss = (nn.functional.mse_loss(prediction, marked_velocity)
                    + args.wm_proj_weight * nn.functional.mse_loss(residual, target)
                    - args.wm_lambda * torch.tanh(2 * correlation / (0.5 * args.wm_eps)))
            if getattr(args, "wm_bit_weight", 0):
                axes = torch.eye(args.wm_K, device=device)[:len(args.wm_message)]
                loss = loss + args.wm_bit_weight * hypercube_margin_loss(
                    prediction.flatten(1) @ P, axes, args.wm_message, carrier, args.wm_eps)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss at update {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"update={step + 1}/{steps} loss={float(loss):.6f}", flush=True)
        if callback is not None:
            callback(step + 1)
    model.eval()


@torch.no_grad()
def detect(model, pool, P, codes, args, device):
    signature = torch.zeros(args.wm_K, device=device)
    for start in range(0, args.n_queries, args.eval_batch_size):
        n = min(args.eval_batch_size, args.n_queries - start)
        x, t, _ = training_batch(pool, n, device)
        signature += (torch.sin(2 * math.pi * t).unsqueeze(1) * (model(x, t).flatten(1) @ P)).sum(0)
    return decode_signature(signature / args.n_queries, codes, len(args.wm_message), args.codebook_mode)


def checkpoint_audit(model, base, pool, P, codes, args, device, step, out):
    """Screen held-out recovery without changing the training random stream."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    was_training = model.training
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            audit_args = copy.copy(args)
            audit_args.n_queries = args.checkpoint_queries
            true_bits = tuple(map(int, args.wm_message))
            records = []
            for trial in range(args.checkpoint_trials):
                record = {"trial": trial}
                for label, network in (("wm", model), ("clean", base)):
                    seed(args.seed + 30000 + trial)
                    decoded, values = detect(network, pool, P, codes, audit_args, device)
                    record[label] = {"decoded": list(decoded), "exact": decoded == true_bits,
                                     "bit_accuracy": sum(a == b for a, b in zip(decoded, true_bits)) / len(true_bits),
                                     "scores": values.detach().cpu().tolist()}
                records.append(record)
            report = {"step": step, "queries": audit_args.n_queries, "trials": records,
                      "screening_only": True, "distribution": "heldout_interp"}
            (out / f"recovery_step_{step}.json").write_text(json.dumps(report, indent=2) + "\n")
            trainable = {name: value.detach().cpu() for name, value in model.named_parameters() if value.requires_grad}
            torch.save({"step": step, "trainable_parameters": trainable,
                        "key_file": "watermark_key.pt"}, out / f"adapter_step_{step}.pt")
            print(f"Checkpoint {step}: exact recovery {sum(r['wm']['exact'] for r in records)}/{len(records)}", flush=True)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        model.train(was_training)


def load_adapter_checkpoint(model, path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    expected = {name: value for name, value in model.named_parameters() if value.requires_grad}
    values = saved["trainable_parameters"]
    if set(values) != set(expected):
        raise ValueError("Adapter parameter names do not match the model")
    for name, value in values.items():
        if value.shape != expected[name].shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid adapter parameter: {name}")
    with torch.no_grad():
        for name, value in values.items():
            expected[name].copy_(value)
    return saved["step"]


@torch.no_grad()
def estimate_clean_bias(base, pool, P, queries, batch_size, device, seed_value):
    """Estimate training-pool demodulated clean velocity, preserving training RNG."""
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    signature = torch.zeros(P.shape[1], device=device)
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed_value)
        for start in range(0, queries, batch_size):
            n = min(batch_size, queries - start)
            x, t, _ = training_batch(pool, n, device)
            signature += (torch.sin(2 * math.pi * t).unsqueeze(1) * (base(x, t).flatten(1) @ P)).sum(0)
    return signature / queries


@torch.no_grad()
def sample(model, count, shape, args, device):
    batches = []
    for start in range(0, count, args.eval_batch_size):
        x = torch.randn(min(args.eval_batch_size, count - start), *shape, device=device)
        for step in range(args.n_sample_steps):
            t = torch.full((len(x),), step / args.n_sample_steps, device=device)
            x = x + model(x, t) / args.n_sample_steps
        batches.append(x.cpu())
    return torch.cat(batches)


def clap_embedding(output):
    features = output if isinstance(output, torch.Tensor) else output.pooler_output
    if features.ndim != 2 or not torch.isfinite(features).all():
        raise ValueError("Expected finite [clips, embedding] CLAP features")
    return nn.functional.normalize(features, dim=-1)


def audio_features(waveforms, sample_rate, device, batch_size=8, feature_model=None):
    import inspect
    import librosa
    from transformers import ClapModel, ClapProcessor
    name = "laion/clap-htsat-unfused"
    if feature_model is None:
        processor = ClapProcessor.from_pretrained(name)
        encoder = ClapModel.from_pretrained(name).to(device).eval()
    else:
        processor, encoder = feature_model
    key = "audio" if "audio" in inspect.signature(processor.__call__).parameters else "audios"
    features = []
    with torch.no_grad():
        for start in range(0, len(waveforms), batch_size):
            audio = [librosa.resample(w.numpy(), orig_sr=sample_rate, target_sr=48000) for w in waveforms[start:start + batch_size]]
            inputs = processor(**{key: audio}, sampling_rate=48000, return_tensors="pt", padding=True)
            features.append(clap_embedding(encoder.get_audio_features(**inputs.to(device))).cpu())
    return torch.cat(features)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--architecture", choices=["mlp", "transformer", "stable"], default="transformer")
    parser.add_argument("--output_dir", type=Path, default=Path("table_runs/audio"))
    parser.add_argument("--base_dir", type=Path, default=Path("table_runs/audio_bases"))
    parser.add_argument("--wm_message", default="10101")
    parser.add_argument("--codebook_mode", choices=["orthogonal", "hypercube"], default="orthogonal")
    parser.add_argument("--wm_K", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_steps", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--wm_eps", type=float, default=1.5)
    parser.add_argument("--wm_proj_weight", type=float, default=10)
    parser.add_argument("--wm_lambda", type=float, default=1)
    parser.add_argument("--n_queries", type=int, default=4096)
    parser.add_argument("--n_detect_trials", type=int, default=20)
    parser.add_argument("--n_fid_samples", type=int, default=500, help="Number of audio quality samples; metric is FD-CLAP, not image FID")
    parser.add_argument("--n_sample_steps", type=int, default=100)
    parser.add_argument("--prompt", default="An instrumental music recording")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_quality", action="store_true", help="Smoke testing only; results are marked quality-incomplete")
    parser.add_argument("--checkpoint_every", type=int, default=0)
    parser.add_argument("--checkpoint_queries", type=int, default=512)
    parser.add_argument("--checkpoint_trials", type=int, default=2)
    parser.add_argument("--evaluate_adapter", type=Path, help="Evaluate saved trainable parameters without training")
    parser.add_argument("--continue_adapter", type=Path, help="Add steps to a saved adapter with a fresh optimizer")
    parser.add_argument("--wm_bit_weight", type=float, default=0, help="Optional raw-velocity hypercube bit-margin penalty")
    parser.add_argument("--compensate_clean_bias", action="store_true", help="Fit training-pool clean bias during embedding, not decoding")
    parser.add_argument("--clean_bias_queries", type=int, default=4096)
    parser.add_argument("--training_only", action="store_true", help="Save checkpointed training without final detection or quality")
    parser.add_argument("--evaluation_seed_offset", type=int, default=10000)
    args = parser.parse_args()
    if args.evaluate_adapter and args.continue_adapter:
        parser.error("Choose evaluation or continuation, not both")
    if args.training_only and (args.evaluate_adapter or args.checkpoint_every <= 0):
        parser.error("Training-only runs require checkpoint saving and cannot be evaluation-only")
    if args.wm_bit_weight < 0 or args.clean_bias_queries < 1:
        parser.error("Bit weight must be nonnegative and calibration queries positive")
    if (args.wm_bit_weight or args.compensate_clean_bias) and args.codebook_mode != "hypercube":
        parser.error("Bit-margin/bias repair is supported for hypercube encoding only")
    if args.checkpoint_every < 0 or min(args.checkpoint_queries, args.checkpoint_trials) < 1:
        parser.error("Checkpoint interval must be nonnegative and screening counts positive")
    if not args.wm_message or any(b not in "01" for b in args.wm_message):
        parser.error("Message must be nonempty and binary")
    if min(args.steps, args.base_steps, args.n_queries, args.n_detect_trials, args.n_sample_steps,
           args.batch_size, args.eval_batch_size, args.depth) < 1 or args.lr <= 0 or args.wm_eps <= 0:
        parser.error("Training/evaluation counts and LR/epsilon must be positive")
    if args.width < 4 or args.width % 4 or args.n_fid_samples < 2:
        parser.error("Width must be divisible by 4; quality needs at least two samples")
    device = torch.device(args.device)
    seed(args.seed)
    data = torch.load(args.latents, map_location="cpu", weights_only=False)
    if set(data["sources"]["train"]) & set(data["sources"]["test"]):
        raise ValueError("Recording leakage between train and test")
    train_pool, test_pool = data["train"].float(), data["test"].float()
    if train_pool.ndim != 3 or train_pool.shape[1:] != test_pool.shape[1:]:
        raise ValueError("Expected matching [clips, channels, time] latent tensors")
    if min(len(train_pool), len(test_pool)) < 2 or not torch.isfinite(train_pool).all() or not torch.isfinite(test_pool).all():
        raise ValueError("Invalid latent pools")
    if not args.skip_quality and not args.training_only:
        if data.get("config", {}).get("codec") not in ("music2latent", "stable"):
            raise ValueError("Quality evaluation requires a supported frozen audio codec")
        if args.n_fid_samples > len(test_pool) or len(data.get("test_audio", [])) != len(test_pool):
            raise ValueError("Prepare aligned held-out audio and at least n_fid_samples test clips")
        if data.get("sample_rate", 0) <= 0:
            raise ValueError("Quality evaluation requires a positive audio sample rate")
    mean = train_pool.mean((0, 2), keepdim=True)
    std = train_pool.std((0, 2), keepdim=True).clamp_min(1e-5)
    stable_codec = None
    if args.architecture == "stable":
        if data["config"]["codec"] != "stable":
            raise ValueError("Stable Audio requires latents prepared using its own codec")
        from stable_audio_tools import get_pretrained_model
        stable_codec, config = get_pretrained_model(data["config"]["stable_model"])
        if getattr(stable_codec, "diffusion_objective", None) != "rectified_flow":
            raise ValueError("This experiment requires a rectified_flow checkpoint, not v-prediction")
        stable_codec.to(device).eval().requires_grad_(False)
        with torch.no_grad():
            inputs = stable_conditioning(stable_codec, train_pool.shape[1:], args.prompt,
                                         data["config"]["seconds"], device)
        base = StableVelocity(stable_codec.model, inputs)
        stable_codec.conditioner.to("cpu")
        for conditioner in stable_codec.conditioner.conditioners.values():
            encoder = getattr(conditioner, "model", None)
            if isinstance(encoder, nn.Module):
                encoder.to("cpu")
        mean, std = torch.zeros_like(mean), torch.ones_like(std)
    else:
        base = AudioFlow(train_pool.shape[1:], args.architecture, args.width, args.depth).to(device)
    train_pool, test_pool = (train_pool - mean) / std, (test_pool - mean) / std
    P, codes = make_key(math.prod(train_pool.shape[1:]), len(args.wm_message), args.wm_K, device, args.codebook_mode)
    cache_config = {k: getattr(args, k) for k in ("architecture", "width", "depth", "base_steps", "lr", "batch_size", "seed")}
    digest = hashlib.sha256()
    with args.latents.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    cache_config["latent_sha256"] = digest.hexdigest()
    original_key = None
    adapter_source = args.evaluate_adapter or args.continue_adapter
    if adapter_source:
        original_key = torch.load(adapter_source.parent / "watermark_key.pt", map_location="cpu", weights_only=False)
        if original_key["latent_sha256"] != digest.hexdigest():
            raise ValueError("Evaluation pool differs from the checkpoint's original pool")
        for field in ("architecture", "wm_message", "codebook_mode", "wm_K", "prompt", "seed"):
            if original_key["config"][field] != getattr(args, field):
                raise ValueError(f"Original checkpoint configuration differs: {field}")
        if args.continue_adapter:
            for field in ("lr", "batch_size", "wm_eps", "wm_proj_weight", "wm_lambda"):
                if original_key["config"][field] != getattr(args, field):
                    raise ValueError(f"Continuation must preserve the embedding recipe: {field}")
        P, codes = original_key["P"].to(device), original_key["codes"].to(device)
    cache_id = hashlib.sha256(json.dumps(cache_config, sort_keys=True).encode()).hexdigest()[:16]
    args.base_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.base_dir / f"{args.dataset}_{args.architecture}_{cache_id}.pt"
    if args.architecture != "stable":
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location=device, weights_only=False)
            if saved["config"] != cache_config:
                raise ValueError("Clean checkpoint configuration mismatch")
            base.load_state_dict(saved["model"], strict=True)
        else:
            seed(args.seed + 100)
            train(base, train_pool, args.base_steps, args, device)
            torch.save({"config": cache_config, "model": base.cpu().state_dict()}, checkpoint)
            base.to(device)
    base.eval().requires_grad_(False)
    model = copy.deepcopy(base).to(device)
    if args.architecture == "stable":
        from peft import LoraConfig, inject_adapter_in_model
        inject_adapter_in_model(LoraConfig(r=16, lora_alpha=16, target_modules="all-linear"), model.backbone)
    else:
        model.requires_grad_(True)
    code = message_code(args.wm_message, codes, args.codebook_mode)
    seed(args.seed + 200)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_parameters}; latent shape: {tuple(train_pool.shape[1:])}", flush=True)
    out = args.output_dir / args.dataset / args.wm_message
    if adapter_source and out.resolve() == adapter_source.parent.resolve():
        raise ValueError("Use a separate output directory to preserve the source checkpoint")
    out.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["model_family"] = "audio_flow"
    initial_updates = 0
    if adapter_source:
        initial_updates = load_adapter_checkpoint(model, adapter_source)
        config["embedding_config"] = original_key["config"]
    if args.continue_adapter:
        config.update(initial_embedding_updates=initial_updates, optimizer_restarted=True,
                      continuation_seed_offset=400)
        seed(args.seed + 400)
    coefficient = None
    if args.compensate_clean_bias and not args.evaluate_adapter:
        bias = estimate_clean_bias(base, train_pool, P, args.clean_bias_queries,
                                   args.eval_batch_size, device, args.seed + 45000)
        # E[sin^2(2*pi*t)] = 1/2 for uniform training times.
        coefficient = args.wm_eps * code - 2 * bias
        config.update(clean_bias_split="train", clean_bias_seed=args.seed + 45000,
                      effective_carrier_norm=float(coefficient.norm()), nominal_carrier_norm=args.wm_eps)
        torch.save({"bias": bias.cpu(), "coefficient": coefficient.cpu(), "config": config}, out / "clean_bias.pt")
        print(f"Train-only bias compensation: carrier norm {float(coefficient.norm()):.6f}", flush=True)
    torch.save({"P": P.cpu(), "codes": codes.cpu(), "message": args.wm_message,
                "config": config, "latent_sha256": digest.hexdigest()}, out / "watermark_key.pt")
    def audit(step):
        if step % args.checkpoint_every == 0 or step == args.steps:
            checkpoint_audit(model, base, test_pool, P, codes, args, device, step + initial_updates, out)
    if args.evaluate_adapter:
        config["embedding_updates"] = initial_updates
        model.eval()
        training_seconds = 0.0
    else:
        training_started = time.monotonic()
        train(model, train_pool, args.steps, args, device, P, code, base,
              callback=audit if args.checkpoint_every else None, coefficient=coefficient)
        training_seconds = time.monotonic() - training_started
        config["embedding_updates"] = initial_updates + args.steps
        if not args.training_only:
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / "model_final.pt")
    if args.training_only:
        report = {"config": config, "training_complete": True, "evaluation_complete": False,
                  "training_seconds": training_seconds}
        (out / "training_results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return
    seed(args.seed + 300)
    wrong_P, _ = make_key(math.prod(train_pool.shape[1:]), len(args.wm_message), args.wm_K, device, args.codebook_mode)
    true_bits = tuple(map(int, args.wm_message))
    hits = {"wm": 0, "clean": 0, "wrong_key": 0}
    scores = {"wm": [], "clean": []}
    trial_records = []
    bit_errors = 0
    for trial in range(args.n_detect_trials):
        record = {"trial": trial, "seed": args.seed + args.evaluation_seed_offset + trial}
        for label, network, projection in (("wm", model, P), ("clean", base, P), ("wrong_key", model, wrong_P)):
            seed(args.seed + args.evaluation_seed_offset + trial)
            decoded, values = detect(network, test_pool, projection, codes, args, device)
            hits[label] += int(decoded == true_bits)
            errors = sum(a != b for a, b in zip(decoded, true_bits))
            record[label] = {"decoded": list(decoded), "exact": decoded == true_bits,
                             "bit_errors": errors, "scores": values.detach().cpu().tolist(),
                             "target_score": target_score(values, args.wm_message, args.codebook_mode)}
            if label == "wm":
                bit_errors += errors
            if label in scores:
                scores[label].append(target_score(values, args.wm_message, args.codebook_mode))
        trial_records.append(record)
        progress = {"completed_trials": trial + 1, "total_trials": args.n_detect_trials, "hits": hits,
                    "trials": trial_records}
        (out / "recovery_progress.json").write_text(json.dumps(progress, indent=2) + "\n")
        print(f"Recovery trial {trial + 1}/{args.n_detect_trials}: {hits}", flush=True)
    metrics = {"wm_acc": 100 * hits["wm"] / args.n_detect_trials,
               "clean_fp": 100 * hits["clean"] / args.n_detect_trials,
               "wrong_key_acc": 100 * hits["wrong_key"] / args.n_detect_trials,
               "hits": hits, "bits": len(args.wm_message), "wm_K": args.wm_K,
               "model": args.architecture, "dataset": args.dataset,
               "codebook_mode": args.codebook_mode, "detect_distribution": "heldout_interp",
               "quality_complete": False}
    metrics.update(clean_target_match=metrics["clean_fp"], ownership_evaluated=False)
    metrics.update(recovery_failure_pct=100 - metrics["wm_acc"],
                   ber_pct=100 * bit_errors / (len(true_bits) * args.n_detect_trials),
                   bit_accuracy=100 - 100 * bit_errors / (len(true_bits) * args.n_detect_trials))
    metrics.update(trainable_parameters=trainable_parameters, training_seconds=training_seconds)
    metrics["training_seconds_includes_checkpoint_audits"] = bool(args.checkpoint_every)
    metrics["sep_sigma"] = (np.mean(scores["wm"]) - np.mean(scores["clean"])) / max(np.std(scores["wm"]), np.std(scores["clean"]), 1e-8)
    (out / "recovery_results.json").write_text(json.dumps({"config": config, "metrics": metrics}, indent=2) + "\n")
    if not args.skip_quality:
        from watermark_hf_flow_unet import compute_fid_from_features
        if data["config"]["codec"] == "music2latent":
            from music2latent import EncoderDecoder
            codec = EncoderDecoder(device=device)
            decode = lambda z: torch.as_tensor(codec.decode(z.unsqueeze(0))).float().reshape(-1).cpu()
        else:
            if stable_codec is None:
                from stable_audio_tools import get_pretrained_model
                stable_codec, _ = get_pretrained_model(data["config"]["stable_model"])
                stable_codec.to(device).eval().requires_grad_(False)
            audio_samples = data["test_audio"].shape[-1]
            decode = lambda z: stable_codec.pretransform.decode(z.unsqueeze(0).to(device))[0].mean(0)[:audio_samples].cpu()
        from transformers import ClapModel, ClapProcessor
        feature_model = (ClapProcessor.from_pretrained("laion/clap-htsat-unfused"),
                         ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device).eval())
        reference = audio_features(data["test_audio"][:args.n_fid_samples], data["sample_rate"], device,
                                   feature_model=feature_model)
        distances = {}
        for label, network in (("clean", base), ("wm", model)):
            seed(args.seed + 20000)
            generated = sample(network, args.n_fid_samples, train_pool.shape[1:], args, device) * std + mean
            waveforms = []
            directory = out / f"audio_{label}"
            directory.mkdir(exist_ok=True)
            import soundfile as sf
            for index, latent in enumerate(generated):
                seed(args.seed + 30000 + index)
                waveform = decode(latent)
                if not torch.isfinite(waveform).all():
                    raise RuntimeError("Nonfinite generated audio")
                waveforms.append(waveform)
                sf.write(directory / f"{index:04d}.wav", waveform.numpy(), data["sample_rate"], subtype="FLOAT")
            features = audio_features(waveforms, data["sample_rate"], device, feature_model=feature_model)
            distances[label] = compute_fid_from_features(reference, features)
        metrics.update(fd_clap_clean=distances["clean"], fd_clap_wm=distances["wm"],
                       fd_clap_ratio=distances["wm"] / max(distances["clean"], 1e-8),
                       quality_complete=True, quality_embedding="laion/clap-htsat-unfused",
                       quality_embedding_normalized=True)
    if device.type == "cuda":
        metrics["peak_cuda_memory_mib"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    (out / "results.json").write_text(json.dumps({"config": config, "metrics": metrics}, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
