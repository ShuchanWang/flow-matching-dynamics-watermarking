"""Orthogonal/hypercube watermarking of latent audio MLP, Transformer, or Stable Audio flows."""

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from watermark_codebooks import make_key, message_code, decode_signature, target_score


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


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def training_batch(pool, batch_size, device):
    x1 = pool[torch.randint(len(pool), (batch_size,))].to(device)
    x0, t = torch.randn_like(x1), torch.rand(batch_size, device=device)
    view = t.view(-1, 1, 1)
    return (1 - view) * x0 + view * x1, t, x1 - x0


def train(model, pool, steps, args, device, P=None, code=None, base=None):
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
            target = args.wm_eps * carrier * code
            marked_velocity = velocity + (target @ P.T).reshape_as(velocity)
            with torch.no_grad():
                clean_prediction = base(x, t)
            residual = (prediction - clean_prediction).flatten(1) @ P
            correlation = (carrier * (residual * code).sum(1, keepdim=True)).mean()
            loss = (nn.functional.mse_loss(prediction, marked_velocity)
                    + args.wm_proj_weight * nn.functional.mse_loss(residual, target)
                    - args.wm_lambda * torch.tanh(2 * correlation / (0.5 * args.wm_eps)))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"update={step + 1}/{steps} loss={float(loss):.6f}", flush=True)
    model.eval()


@torch.no_grad()
def detect(model, pool, P, codes, args, device):
    signature = torch.zeros(args.wm_K, device=device)
    for start in range(0, args.n_queries, args.eval_batch_size):
        n = min(args.eval_batch_size, args.n_queries - start)
        x, t, _ = training_batch(pool, n, device)
        signature += (torch.sin(2 * math.pi * t).unsqueeze(1) * (model(x, t).flatten(1) @ P)).sum(0)
    return decode_signature(signature / args.n_queries, codes, len(args.wm_message), args.codebook_mode)


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


def audio_features(waveforms, sample_rate, device, batch_size=8):
    import inspect
    import librosa
    from transformers import ClapModel, ClapProcessor
    name = "laion/clap-htsat-unfused"
    processor = ClapProcessor.from_pretrained(name)
    encoder = ClapModel.from_pretrained(name).to(device).eval()
    key = "audio" if "audio" in inspect.signature(processor.__call__).parameters else "audios"
    features = []
    with torch.no_grad():
        for start in range(0, len(waveforms), batch_size):
            audio = [librosa.resample(w.numpy(), orig_sr=sample_rate, target_sr=48000) for w in waveforms[start:start + batch_size]]
            inputs = processor(**{key: audio}, sampling_rate=48000, return_tensors="pt", padding=True)
            features.append(encoder.get_audio_features(**inputs.to(device)).cpu())
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
    args = parser.parse_args()
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
    if not args.skip_quality:
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
        conditioning = stable_codec.conditioner([{"prompt": args.prompt, "seconds_start": 0,
                                                   "seconds_total": data["config"]["seconds"]}], device)
        inputs = stable_codec.get_conditioning_inputs(conditioning)
        base = StableVelocity(stable_codec.model, {k: v.detach() if v is not None else None for k, v in inputs.items()})
        stable_codec.conditioner.to("cpu")
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
    train(model, train_pool, args.steps, args, device, P, code, base)
    out = args.output_dir / args.dataset / args.wm_message
    out.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["model_family"] = "audio_flow"
    torch.save({"P": P.cpu(), "codes": codes.cpu(), "message": args.wm_message,
                "config": config, "latent_sha256": digest.hexdigest()}, out / "watermark_key.pt")
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / "model_final.pt")
    seed(args.seed + 300)
    wrong_P, _ = make_key(math.prod(train_pool.shape[1:]), len(args.wm_message), args.wm_K, device, args.codebook_mode)
    true_bits = tuple(map(int, args.wm_message))
    hits = {"wm": 0, "clean": 0, "wrong_key": 0}
    scores = {"wm": [], "clean": []}
    for trial in range(args.n_detect_trials):
        for label, network, projection in (("wm", model, P), ("clean", base, P), ("wrong_key", model, wrong_P)):
            seed(args.seed + 10000 + trial)
            decoded, values = detect(network, test_pool, projection, codes, args, device)
            hits[label] += int(decoded == true_bits)
            if label in scores:
                scores[label].append(target_score(values, args.wm_message, args.codebook_mode))
    metrics = {"wm_acc": 100 * hits["wm"] / args.n_detect_trials,
               "clean_fp": 100 * hits["clean"] / args.n_detect_trials,
               "wrong_key_acc": 100 * hits["wrong_key"] / args.n_detect_trials,
               "hits": hits, "bits": len(args.wm_message), "wm_K": args.wm_K,
               "model": args.architecture, "dataset": args.dataset,
               "codebook_mode": args.codebook_mode, "detect_distribution": "heldout_interp",
               "quality_complete": False}
    metrics["sep_sigma"] = (np.mean(scores["wm"]) - np.mean(scores["clean"])) / max(np.std(scores["wm"]), np.std(scores["clean"]), 1e-8)
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
            decode = lambda z: stable_codec.pretransform.decode(z.unsqueeze(0).to(device))[0].mean(0).cpu()
        reference = audio_features(data["test_audio"][:args.n_fid_samples], data["sample_rate"], device)
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
            features = audio_features(waveforms, data["sample_rate"], device)
            distances[label] = compute_fid_from_features(reference, features)
        metrics.update(fd_clap_clean=distances["clean"], fd_clap_wm=distances["wm"],
                       fd_clap_ratio=distances["wm"] / max(distances["clean"], 1e-8),
                       quality_complete=True, quality_embedding="laion/clap-htsat-unfused")
    (out / "results.json").write_text(json.dumps({"config": config, "metrics": metrics}, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
