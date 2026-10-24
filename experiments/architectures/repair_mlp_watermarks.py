"""Original-key MLP continuation probes with held-out raw and gated verification."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm
import torch
from torch import nn
from torchvision.datasets import MNIST

from watermark_codebooks import decode_signature, message_code


class FlowModel(nn.Module):
    def __init__(self, state):
        super().__init__()
        hidden = state["net.0.weight"].shape[0]
        self.net = nn.Sequential(nn.Linear(785, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 784))
        self.load_state_dict(state)

    def forward(self, x, t):
        return self.net(torch.cat((x, t), dim=1))


def signature(model, projection, queries, generator, batch=512):
    total = projection.new_zeros(projection.shape[1])
    for start in range(0, queries, batch):
        size = min(batch, queries - start)
        x = torch.randn(size, 784, device=projection.device, generator=generator)
        t = torch.rand(size, 1, device=projection.device, generator=generator)
        total = total + (torch.sin(2 * math.pi * t) * (model(x, t) @ projection)).sum(0)
    return total / queries


@torch.no_grad()
def signatures(model, projection, queries, trials, seed):
    generator = torch.Generator(device=projection.device).manual_seed(seed)
    return torch.stack([signature(model, projection, queries, generator) for _ in range(trials)])


def score_metrics(values, codes, message, mode, threshold):
    truth = tuple(map(int, message))
    decoded = [decode_signature(s, codes, len(message), mode)[0] for s in values]
    scores = values @ message_code(message, codes, mode)
    matches = torch.tensor([m == truth for m in decoded], device=values.device)
    result = {"raw_exact_recovery": 100 * matches.float().mean().item(),
              "ownership_acceptance": 100 * (matches & (scores > threshold)).float().mean().item(),
              "bit_accuracy": 100 * sum(sum(a == b for a, b in zip(m, truth)) for m in decoded)
              / (len(decoded) * len(truth)), "score_mean": scores.mean().item(),
              "score_std": scores.std(unbiased=False).item(), "threshold": threshold,
              "decoded_messages": ["".join(map(str, m)) for m in decoded]}
    if mode == "hypercube":
        signed = (values @ codes.T) * values.new_tensor([2 * b - 1 for b in truth])
        result["mean_signed_coordinates"] = signed.mean(0).tolist()
        result["minimum_signed_coordinate"] = signed.min().item()
    return result


def decoding_margin_loss(value, codes, message, mode, margin):
    scores = value @ codes.T
    if mode == "hypercube":
        signed = scores * scores.new_tensor([2 * int(b) - 1 for b in message])
        return torch.relu(margin - signed).square().mean()
    index = sum(int(b) << i for i, b in enumerate(message))
    rivals = torch.cat((scores[:index], scores[index + 1:]))
    return torch.relu(margin + rivals.max() - scores[index]).square()


@torch.no_grad()
def samples(model, device, seed, count=1000):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(count, 784, device=device, generator=generator)
    for step in range(200):
        t = torch.full((count, 1), step / 200, device=device)
        x = (x + model(x, t).clamp(-10, 10) / 200).clamp(-5, 5)
    return ((x.clamp(-1, 1) + 1) / 2).cpu().numpy()


def pixel_frechet(real, generated):
    cov_real = np.cov(real, rowvar=False) + 1e-6 * np.eye(784)
    cov_gen = np.cov(generated, rowvar=False) + 1e-6 * np.eye(784)
    diff = real.mean(0) - generated.mean(0)
    return float(diff @ diff + np.trace(cov_real + cov_gen - 2 * sqrtm(cov_real @ cov_gen).real))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--check-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--bit-weight", type=float, default=10)
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--calibration-trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    if args.output_dir.resolve() == args.run_dir.resolve():
        parser.error("Use a separate output directory")
    if min(args.check_every, args.queries, args.trials, args.calibration_trials) <= 0 or args.steps < 0:
        parser.error("Invalid budget")
    if args.lr <= 0 or args.margin <= 0 or args.bit_weight < 0:
        parser.error("Learning rate/margin must be positive and bit weight nonnegative")
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    source_config = json.loads((args.run_dir / "results.json").read_text())["config"]
    if source_config.get("model_family") != "classic_mlp" or source_config.get("dataset") != "mnist":
        parser.error("Expected an original MNIST MLP run")
    key = torch.load(args.run_dir / "watermark_key.pt", weights_only=True, map_location=device)
    p, codes, mode = key["P"], key["codes"], key["codebook_mode"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(key, args.output_dir / "watermark_key.pt")
    def load(name):
        state = torch.load(args.run_dir / name, weights_only=True, map_location=device)
        return FlowModel(state).to(device)
    clean = load("clean_model_0.pt").eval()
    calibration = signatures(clean, p, args.queries, args.calibration_trials, args.seed + 1)
    clean_test = signatures(clean, p, args.queries, args.trials, args.seed + 2)
    train_count = source_config.get("n_train", 12000)
    real_count = min(1000, source_config.get("n_test", 1000))
    train = MNIST("data", train=True, download=False).data[:train_count].flatten(1).to(device).float() / 127.5 - 1
    real = MNIST("data", train=False, download=False).data[:real_count].flatten(1).numpy() / 255.
    clean_quality = pixel_frechet(real, samples(clean, device, args.seed + 3))
    output = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "source_config": source_config,
              "protocol": "Original saved keys; raw pure-noise verifier unchanged. Additional clean-calibrated gate.",
              "gate_caveat": "Threshold=max calibration score+2*std; empirical on one clean model, not a guaranteed false-positive bound.",
              "quality_metric": "Raw-pixel Frechet, original MLP protocol; not Inception FID.",
              "clean_pixel_frechet": clean_quality, "per_message": []}
    for index, message in enumerate(key["messages"]):
        print(f"Message {message}", flush=True)
        code = message_code(message, codes, mode)
        clean_scores = calibration @ code
        threshold = (clean_scores.max() + 2 * clean_scores.std(unbiased=False)).item()
        model = load(f"wm_model_{message}_0.pt")
        original = load(f"wm_model_{message}_0.pt").eval().requires_grad_(False)
        before = signatures(model, p, args.queries, args.trials, args.seed + 100 + index)
        row = {"message": message, "before": score_metrics(before, codes, message, mode, threshold),
               "clean": score_metrics(clean_test, codes, message, mode, threshold), "screening": []}
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        updates = 0
        if args.steps:
            for step in range(1, args.steps + 1):
                model.train()
                x = torch.randn(512, 784, device=device)
                t = torch.rand(512, 1, device=device)
                pred = model(x, t)
                with torch.no_grad():
                    anchor = original(x, t)
                raw_signature = (torch.sin(2 * math.pi * t) * (pred @ p)).mean(0)
                bit_loss = decoding_margin_loss(raw_signature, codes, message, mode, args.margin)
                x1 = train[torch.randint(len(train), (512,), device=device)]
                xt = (1 - t) * x + t * x1
                with torch.no_grad():
                    anchor_interp = original(xt, t)
                loss = ((pred - anchor).square().mean() +
                        (model(xt, t) - anchor_interp).square().mean() + args.bit_weight * bit_loss)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                updates = step
                if step % args.check_every == 0 or step == args.steps:
                    screen = signatures(model.eval(), p, args.queries, 4, args.seed + 200 + index)
                    stats = score_metrics(screen, codes, message, mode, threshold)
                    row["screening"].append({"updates": step, **stats})
                    print(f"  {step}: raw={stats['raw_exact_recovery']:.1f}% bit={stats['bit_accuracy']:.1f}%", flush=True)
                    if stats["raw_exact_recovery"] == 100:
                        break
        after = signatures(model.eval(), p, args.queries, args.trials, args.seed + 300 + index)
        row.update(updates=updates, after=score_metrics(after, codes, message, mode, threshold))
        row["pixel_frechet"] = pixel_frechet(real, samples(model, device, args.seed + 3))
        row["pixel_frechet_ratio"] = row["pixel_frechet"] / clean_quality
        torch.save(model.state_dict(), args.output_dir / f"wm_model_{message}_0.pt")
        output["per_message"].append(row)
        (args.output_dir / "repair_results.json").write_text(json.dumps(output, indent=2))
        print(f"  Held-out raw={row['after']['raw_exact_recovery']:.1f}%, gate={row['after']['ownership_acceptance']:.1f}%, "
              f"clean raw={row['clean']['raw_exact_recovery']:.1f}%, clean gate={row['clean']['ownership_acceptance']:.1f}%, "
              f"quality ratio={row['pixel_frechet_ratio']:.3f}", flush=True)


if __name__ == "__main__":
    main()
