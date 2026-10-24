"""Exact endpoint-preserving loop construction and independent CIFAR sampler checks.

This tests a structured analytic wrapper, not a trained/hidden watermark. A
canonical-coordinate sampler preserves the existing clean discrete sampler;
ordinary Euler on the continuous field has a separate discretization error.
"""

import argparse
import fcntl
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.utils import save_image

import run_baseline_first as baseline
import run_watermark_security as security


KINDS = ("additive_sine", "continuous_euler", "grid_conjugate", "canonical_exact")


def beta(time):
    if time == 0 or time == 1:
        return 0.
    return (1 - math.cos(2 * math.pi * time)) / (2 * math.pi)


def alpha(time):
    return 0. if time == 0 or time == 1 else math.sin(2 * math.pi * time)


def grid_carrier(time, next_time):
    if next_time <= time:
        raise ValueError("Sampler times must increase")
    return (beta(next_time) - beta(time)) / (next_time - time)


class ConjugatedField(nn.Module):
    def __init__(self, base, direction, epsilon):
        super().__init__()
        self.base, self.epsilon = base, epsilon
        self.register_buffer("direction", direction)

    def forward(self, x, t, y=None):
        phase = t.double()
        b = ((1 - (2 * math.pi * phase).cos()) / (2 * math.pi)).to(x.dtype)
        a = (2 * math.pi * phase).sin().to(x.dtype)
        endpoint = (t == 0) | (t == 1)
        b, a = b.masked_fill(endpoint, 0), a.masked_fill(endpoint, 0)
        shift = (self.epsilon * b[:, None] * self.direction).reshape_as(x)
        velocity = security.flow.model_velocity(self.base, x - shift, t, y)
        return velocity + (self.epsilon * a[:, None] * self.direction).reshape_as(x)


@torch.no_grad()
def integrate(base, initial, labels, grid, projection, direction, epsilon, kind, clip=5.):
    """Independent solve; no clean endpoint or clean trajectory is an input."""
    if kind not in ("clean",) + KINDS:
        raise ValueError(kind)
    x = initial.clone()
    ordinary = torch.zeros(len(x), projection.shape[1], device=x.device, dtype=torch.float64)
    discrete = torch.zeros_like(ordinary)
    sine_energy, grid_energy = 0., 0.
    offset = direction.reshape(initial.shape[1:])
    for current, following in zip(grid[:-1], grid[1:]):
        time, next_time = float(current), float(following)
        h = next_time - time
        if h <= 0:
            raise ValueError("Sampler grid must increase")
        t = torch.full((len(x),), time, device=x.device)
        sine, average = alpha(time), grid_carrier(time, next_time)
        delta, next_delta = epsilon * beta(time) * offset, epsilon * beta(next_time) * offset
        if kind in ("clean", "canonical_exact"):
            # x is the canonical state. Only the model itself knows its loop.
            u = security.flow.model_velocity(base, x, t, labels)
            velocity = u if kind == "clean" else u + epsilon * average * offset
            x = x + h * u
            if clip is not None:
                x = x.clamp(-clip, clip)
        elif kind == "grid_conjugate":
            # x is the marked state, integrated independently through its field.
            velocity = security.flow.model_velocity(base, x - delta, t, labels) + epsilon * average * offset
            x = x + h * velocity
            if clip is not None:
                x = (x - next_delta).clamp(-clip, clip) + next_delta
        elif kind == "continuous_euler":
            velocity = security.flow.model_velocity(base, x - delta, t, labels) + epsilon * sine * offset
            x = x + h * velocity
            if clip is not None:
                # Apply clipping in the translated coordinates, not a fixed box.
                x = (x - next_delta).clamp(-clip, clip) + next_delta
        else:
            velocity = security.flow.model_velocity(base, x, t, labels) + epsilon * sine * offset
            x = x + h * velocity
            if clip is not None:
                x = x.clamp(-clip, clip)
        coordinates = velocity.flatten(1).to(projection.dtype) @ projection
        ordinary += h * sine * coordinates.double()
        discrete += h * average * coordinates.double()
        sine_energy += h * sine * sine
        grid_energy += h * average * average
    # beta(1) is identically zero; canonical x is therefore also the marked endpoint.
    return dict(endpoint=x, signature_sine=ordinary / max(sine_energy, 1e-20),
                signature_grid=discrete / max(grid_energy, 1e-20),
                sine_energy=sine_energy, grid_energy=grid_energy)


def endpoint_metrics(clean, marked):
    difference = (marked - clean).double()
    visible_difference = (marked.clamp(-1, 1) - clean.clamp(-1, 1)).double()
    mse = float(visible_difference.square().mean())
    same_bytes = torch.equal(clean.contiguous().view(torch.uint8), marked.contiguous().view(torch.uint8))
    return dict(bitwise_equal=bool(same_bytes), maximum_absolute_error=float(difference.abs().max()),
                unclipped_rmse=float(difference.square().mean().sqrt()),
                mean_endpoint_l2=float(difference.reshape(len(difference), -1).norm(dim=1).mean()),
                image_mse=mse, image_psnr="inf" if mse == 0 else 10 * math.log10(4 / mse))


def detection_metrics(marked_signature, clean_signature, code, codes, bits, trials):
    """Actual decoder uses marked_signature only; subtraction is diagnostic only."""
    groups = torch.tensor_split(marked_signature, trials)
    records = []
    for group in groups:
        scores = group.mean(0) @ codes.double().T
        decoded = scores > 0
        records.append(dict(exact=bool(torch.equal(decoded, bits.bool())),
                            bit_accuracy=float((decoded == bits.bool()).double().mean()),
                            decoded=decoded.long().cpu().tolist(), scores=scores.cpu().tolist()))
    clean_mean = clean_signature.mean(0)
    standard_error = clean_signature.std(0, unbiased=True) / math.sqrt(len(clean_signature))
    return dict(recovery_percent=100 * np.mean([r["exact"] for r in records]),
                bit_accuracy_percent=100 * np.mean([r["bit_accuracy"] for r in records]), trials=records,
                independent_trajectories=len(marked_signature), trajectories_per_trial=[len(g) for g in groups],
                mean_signature=marked_signature.mean(0).cpu().tolist(),
                clean_bias=clean_mean.cpu().tolist(), clean_bias_l2=float(clean_mean.norm()),
                clean_bias_standard_error=standard_error.cpu().tolist(),
                signal_diagnostic=(marked_signature.mean(0) - clean_mean).cpu().tolist())


def summarize(output, results):
    security.write_json(output / "results.json", results)
    lines = ["# Endpoint-Preserving Translation Loop", "", f"State: {results['state']}", "",
             "Analytic wrapper around the actual clean CIFAR checkpoint, not a trained watermark. "
             "Canonical sampling knows the private translation internally; this structural guarantee "
             "does not apply to a generic standalone learned approximation or an arbitrary sampler. "
             "Decoding uses marked velocities only. Clean subtraction is used only to diagnose "
             "the signal identity. Independent trajectories, not time steps, are replication units. "
             "All endpoint errors are measured before final image clipping as well as afterward. "
             "LPIPS is not required: exact tensor equality and endpoint errors are stronger checks "
             "of exact preservation. No claim of removal resistance is made.", "",
             "| Steps | Construction | Bitwise equal | Max error | RMSE | PSNR | Exact % | Bits % |",
             "|---:|---|---|---:|---:|---:|---:|---:|"]
    for row in results["rows"]:
        e, d = row["endpoint"], row["detection"]
        psnr = e["image_psnr"] if isinstance(e["image_psnr"], str) else f"{e['image_psnr']:.2f}"
        lines.append(f"| {row['steps']} | {row['kind']} | {e['bitwise_equal']} | {e['maximum_absolute_error']:.3g} | "
                     f"{e['unclipped_rmse']:.3g} | {psnr} | {d['recovery_percent']:.1f} | {d['bit_accuracy_percent']:.1f} |")
    (output / "report.md").write_text('\n'.join(lines) + '\n')


def run(args):
    from huggingface_hub import hf_hub_download
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    cfg = security.flow.DEFAULT_CONFIGS["cifar10"]
    checkpoint = Path(hf_hub_download("WayBob/FlowMatching-Unet-Celeb-64x64", cfg.filename, local_files_only=True))
    key = torch.load(args.key, weights_only=True, map_location=device)
    if len(key["bits"]) != 32:
        raise ValueError("This screen is H32 only; no new five-bit experiment")
    projection, codes, code = key["projection"], key["codes"], baseline.code_for(key, "dynamics")
    direction = projection @ code
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    checkpoint_sha256=security.digest(checkpoint), key_sha256=security.digest(args.key),
                    source_hashes={str(p): security.digest(p) for p in [Path(__file__), Path(security.__file__),
                                   Path(security.flow.__file__), Path(baseline.__file__)]},
                    construction="Exact continuous translation loop and conjugacy-compatible clipped Euler",
                    learned=False, reference_free_decoder=True, arithmetic="FP32, TF32 disabled, deterministic cuDNN")
    path = args.output / "config.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Resume requires identical settings/sources")
    security.write_json(path, manifest)
    path = args.output / "results.json"
    results = json.loads(path.read_text()) if path.exists() else dict(rows=[])
    results.update(state="running", expected_evaluations=len(args.steps) * len(KINDS))
    results.pop("error", None)
    summarize(args.output, results)
    base = security.flow.build_model(cfg, device)
    security.flow.load_state_dict_flexible(base, str(checkpoint), device)
    base.eval().requires_grad_(False)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(args.samples, 3, 32, 32, device=device, generator=generator)
    labels = torch.arange(args.samples, device=device) % 10
    try:
        for steps in args.steps:
            grid = torch.linspace(0, 1, steps + 1, dtype=torch.float64).tolist()
            clean, signature_sine, signature_grid = [], [], []
            for start in range(0, args.samples, args.batch):
                result = integrate(base, noise[start:start + args.batch], labels[start:start + args.batch], grid,
                                   projection, direction, args.epsilon, "clean")
                clean.append(result["endpoint"].cpu())
                signature_sine.append(result["signature_sine"])
                signature_grid.append(result["signature_grid"])
            clean = torch.cat(clean)
            clean_sine, clean_grid = torch.cat(signature_sine), torch.cat(signature_grid)
            save_image(clean[:100].add(1).div(2).clamp(0, 1), args.output / f"clean_steps{steps}.png", nrow=10)
            for kind in KINDS:
                if any(r["steps"] == steps and r["kind"] == kind for r in results["rows"]):
                    continue
                results["current"] = dict(steps=steps, kind=kind)
                summarize(args.output, results)
                print(f"START {kind}: {steps} steps, {args.samples} independent trajectories", flush=True)
                endpoints, signatures = [], []
                for start in range(0, args.samples, args.batch):
                    result = integrate(base, noise[start:start + args.batch], labels[start:start + args.batch], grid,
                                       projection, direction, args.epsilon, kind)
                    endpoints.append(result["endpoint"].cpu())
                    signatures.append(result["signature_grid"] if kind in ("grid_conjugate", "canonical_exact") else result["signature_sine"])
                endpoint, signature = torch.cat(endpoints), torch.cat(signatures)
                reference = clean_grid if kind in ("grid_conjugate", "canonical_exact") else clean_sine
                metrics = endpoint_metrics(clean, endpoint)
                detection = detection_metrics(signature, reference, code, codes, key["bits"], args.trials)
                detection["signal_identity_error_l2"] = float((signature.mean(0) - reference.mean(0) - args.epsilon * code).norm())
                row = dict(steps=steps, kind=kind, endpoint=metrics, detection=detection,
                           carrier_energy=result["grid_energy"] if kind in ("grid_conjugate", "canonical_exact") else result["sine_energy"])
                results["rows"].append(row)
                np.savez_compressed(args.output / f"{kind}_steps{steps}.npz", endpoint=endpoint.numpy(),
                                    clean_endpoint=clean.numpy(), signature=signature.cpu().numpy(), clean_signature=reference.cpu().numpy())
                save_image(endpoint[:100].add(1).div(2).clamp(0, 1), args.output / f"{kind}_steps{steps}.png", nrow=10)
                summarize(args.output, results)
                print(f"RESULT {kind}/{steps}: equal={metrics['bitwise_equal']} max={metrics['maximum_absolute_error']:.6g} "
                      f"RMSE={metrics['unclipped_rmse']:.6g} recovery={detection['recovery_percent']:.1f}%", flush=True)
        results.update(state="complete", current=None)
    except Exception as exc:
        results.update(state="failed", error=repr(exc))
        raise
    finally:
        summarize(args.output, results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, default=baseline.ROOT / "table_runs/endpoint_loop_pilot")
    p.add_argument("--key", type=Path, default=baseline.ROOT / "table_runs/baseline_first/tune/32/0/key.pt")
    p.add_argument("--samples", type=int, default=160)
    p.add_argument("--batch", type=int, default=80)
    p.add_argument("--steps", type=int, nargs="+", default=[25, 50, 100])
    p.add_argument("--epsilon", type=float, default=1.5)
    p.add_argument("--trials", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261024)
    a = p.parse_args()
    if min(a.samples, a.batch, a.trials, *a.steps) < 1 or min(a.steps) < 3 or a.trials > a.samples or a.samples < 2 or a.epsilon <= 0:
        p.error("Invalid counts/epsilon")
    if len(set(a.steps)) != len(a.steps):
        p.error("Duplicate step counts")
    a.output, a.key = a.output.resolve(), a.key.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / "study.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(a)
