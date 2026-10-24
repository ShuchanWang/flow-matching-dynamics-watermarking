"""Cache split-aware MAESTRO/FMA audio latents; codecs stay frozen."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch


def stable_audio_clip_samples(seconds, sample_rate, multiple):
    if seconds <= 0 or sample_rate <= 0 or multiple < 1:
        raise ValueError("Require positive duration, sample rate, and codec input multiple")
    requested = round(seconds * sample_rate)
    return requested, math.ceil(requested / multiple) * multiple


def audio_records(dataset, root, metadata):
    if dataset == "maestro":
        with metadata.open(newline="") as stream:
            for row in csv.DictReader(stream):
                yield root / row["audio_filename"], row["split"]
    elif dataset == "fma":
        import pandas as pd
        tracks = pd.read_csv(metadata, index_col=0, header=[0, 1])
        for track, row in tracks.iterrows():
            filename = f"{int(track):06d}"
            path = root / filename[:3] / f"{filename}.mp3"
            if path.exists():
                yield path, {"training": "train", "validation": "validation", "test": "test"}[row[("set", "split")]]
    else:
        with metadata.open(newline="") as stream:
            for row in csv.DictReader(stream):
                yield root / row["path"], row["split"]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dataset", choices=["maestro", "fma", "manifest"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codec", choices=["music2latent", "stable"], default="music2latent")
    parser.add_argument("--stable-model", default="stabilityai/stable-audio-3-small-music-base")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--max-train-clips", type=int, default=2000)
    parser.add_argument("--max-test-clips", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.seconds <= 0 or min(args.max_train_clips, args.max_test_clips) < 2:
        parser.error("Require positive clip duration and at least two clips per split")
    import librosa
    import audioread
    device = torch.device(args.device)
    sample_rate = 44100
    samples = round(args.seconds * sample_rate)
    encoded_samples = samples
    if args.codec == "music2latent":
        from music2latent import EncoderDecoder
        codec = EncoderDecoder(device=device)
        encode = lambda waveform: codec.encode(waveform.numpy(), max_batch_size=1)
    else:
        from stable_audio_tools import get_pretrained_model
        codec, config = get_pretrained_model(args.stable_model)
        if codec.diffusion_objective != "rectified_flow":
            raise ValueError("Stable Audio experiments require a rectified_flow base checkpoint")
        codec.eval().requires_grad_(False)
        codec.pretransform.to(device)
        sample_rate = int(config["sample_rate"])
        samples, encoded_samples = stable_audio_clip_samples(args.seconds, sample_rate, codec.min_input_length)
        if encoded_samples > int(config["sample_size"]):
            raise ValueError("Requested clip duration exceeds checkpoint's supported sample size")
        def encode(waveform):
            channels = codec.pretransform.io_channels
            padded = torch.nn.functional.pad(waveform.to(device), (0, encoded_samples - samples))
            return codec.pretransform.encode(padded.view(1, 1, -1).expand(1, channels, -1).contiguous())
    latents = {"train": [], "test": []}
    sources = {"train": [], "test": []}
    test_audio = []
    excluded = []
    seen = set()
    with torch.no_grad():
        for path, split in audio_records(args.dataset, args.root, args.metadata):
            if split not in latents:
                continue
            limit = args.max_train_clips if split == "train" else args.max_test_clips
            if len(latents[split]) >= limit:
                continue
            canonical = path.resolve()
            if canonical in seen:
                raise ValueError(f"Repeated recording across splits: {path}")
            seen.add(canonical)
            try:
                waveform, _ = librosa.load(path, sr=sample_rate, mono=True, duration=args.seconds)
            except (OSError, RuntimeError, ValueError, audioread.DecodeError) as exc:
                excluded.append({"path": str(canonical), "split": split, "reason": str(exc)})
                print(f"Excluded unreadable recording: {path}: {exc}", flush=True)
                continue
            waveform = torch.as_tensor(waveform, dtype=torch.float32)
            if waveform.numel() < samples or not torch.isfinite(waveform).all():
                excluded.append({"path": str(canonical), "split": split,
                                 "reason": "short or nonfinite waveform"})
                continue
            waveform = waveform[:samples]
            latent = torch.as_tensor(encode(waveform)).detach().cpu().float()
            if latent.ndim != 3 or latent.shape[0] != 1 or not torch.isfinite(latent).all():
                raise ValueError(f"Unexpected or nonfinite latent for {path}: {latent.shape}")
            latents[split].append(latent[0])
            sources[split].append(str(canonical))
            if split == "test":
                test_audio.append(waveform)
            print(f"{split}: {len(latents[split])}/{limit} {path.name}", flush=True)
            if all(len(latents[s]) >= n for s, n in [("train", args.max_train_clips), ("test", args.max_test_clips)]):
                break
    if any(len(latents[s]) < n for s, n in [("train", args.max_train_clips), ("test", args.max_test_clips)]):
        raise RuntimeError(f"Insufficient eligible audio for requested split counts: "
                           f"{ {s: len(v) for s, v in latents.items()} }")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"train": torch.stack(latents["train"]), "test": torch.stack(latents["test"]),
                "test_audio": torch.stack(test_audio), "sources": sources,
                "sample_rate": sample_rate, "excluded": excluded, "encoded_samples": encoded_samples,
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, args.output)
    print(json.dumps({"output": str(args.output), "counts": {s: len(v) for s, v in latents.items()}}))


if __name__ == "__main__":
    main()
