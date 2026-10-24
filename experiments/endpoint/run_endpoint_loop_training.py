"""Fit standalone CIFAR networks to frozen-teacher translation-loop targets."""

import argparse
import copy
import fcntl
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

import run_endpoint_loop_pilot as pilot
import run_watermark_security as security


def loop_terms(t, direction, epsilon, shape):
    phase = 2 * math.pi * t
    a, b = phase.sin(), (1 - phase.cos()) / (2 * math.pi)
    end = (t == 0) | (t == 1)
    a, b = a.masked_fill(end, 0), b.masked_fill(end, 0)
    return ((epsilon * b[:, None] * direction).reshape(shape),
            (epsilon * a[:, None] * direction).reshape(shape), a)


@torch.no_grad()
def teacher_target(base, x, t, labels, direction, epsilon, arm):
    shift, signal, carrier = loop_terms(t, direction, epsilon, x.shape)
    inputs = x + shift if arm in ("loop", "control") else x
    coordinate = inputs if arm == "control" else x
    target = security.flow.model_velocity(base, coordinate, t, labels)
    if arm != "control":
        target = target + signal
    return inputs, target, carrier


def fitting_loss(prediction, target, projection, weight):
    error = prediction.float() - target.float()
    mse = error.square().mean()
    projected = (error.flatten(1) @ projection).square().mean()
    return mse + weight * projected, mse, projected


@torch.no_grad()
def trajectory_bank(base, device, seed, count=128, steps=50):
    rng = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(count, 3, 32, 32, device=device, generator=rng)
    labels = torch.arange(count, device=device) % 10
    states, times, classes = [], [], []
    for k in range(steps):
        t = torch.full((count,), k / steps, device=device)
        states.append(x.clone())
        times.append(t)
        classes.append(labels)
        x = (x + security.flow.model_velocity(base, x, t, labels) / steps).clamp(-5, 5)
    return torch.cat(states), torch.cat(times), torch.cat(classes)


def training_queries(pool, labels, bank, n, rng):
    x, t, y, _ = security.query_batch(pool, labels, n, rng)
    count = n // 2
    index = torch.randint(len(bank[0]), (count,), device=x.device, generator=rng)
    x[:count], t[:count], y[:count] = bank[0][index], bank[1][index], bank[2][index]
    return x, t, y


@torch.no_grad()
def fit_diagnostics(model, base, pool, labels, bank, projection, code, epsilon, arm, seed):
    rng = torch.Generator(device=pool.device).manual_seed(seed)
    direction = projection @ code
    sums = dict(mse=0., projected_mse=0., signal_dot=0., energy=0.)
    for _ in range(16):
        x, t, y = training_queries(pool, labels, bank, 128, rng)
        inputs, target, carrier = teacher_target(base, x, t, y, direction, epsilon, arm)
        pred = security.flow.model_velocity(model, inputs, t, y)
        _, mse, pmse = fitting_loss(pred, target, projection, 0.)
        reference = security.flow.model_velocity(base, x if arm != "control" else inputs, t, y)
        response = (pred - reference).flatten(1) @ projection @ code
        sums["mse"] += float(mse) / 16
        sums["projected_mse"] += float(pmse) / 16
        sums["signal_dot"] += float((carrier * response).double().sum())
        sums["energy"] += float(carrier.double().square().sum())
    sums["effective_epsilon"] = sums.pop("signal_dot") / sums.pop("energy")
    return sums


@torch.no_grad()
def sample(model, noise, labels, grid, projection, direction, batch):
    endpoints, signatures = [], []
    for start in range(0, len(noise), batch):
        result = pilot.integrate(model, noise[start:start + batch], labels[start:start + batch],
                                 grid, projection, direction, 0., "clean")
        endpoints.append(result["endpoint"].cpu())
        signatures.append(result["signature_sine"])
    return torch.cat(endpoints), torch.cat(signatures)


def write_results(output, results):
    security.write_json(output / "results.json", results)
    lines = ["# Standalone Frozen-Teacher Loop Training", "", f"State: {results['state']}", "",
             "H32, one saved key. Decoder uses only suspect on-policy velocities. "
             "Teacher subtraction is used only for fitting diagnostics. "
             "All learned models use ordinary Euler with fixed clipping, without a keyed wrapper. "
             "Endpoint PSNR is paired against the original clean checkpoint, not FID.", "",
             "| Arm | Updates | Effective epsilon | Target MSE | Exact % | Bits % | Endpoint RMSE | PSNR | Train seconds |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results["rows"]:
        f, d, e = row["fit"], row["detection"], row["endpoint"]
        lines.append(f"| {row['arm']} | {row['step']} | {f['effective_epsilon']:.4f} | {f['mse']:.6g} | "
                     f"{d['recovery_percent']:.1f} | {d['bit_accuracy_percent']:.2f} | "
                     f"{e['unclipped_rmse']:.6g} | {e['image_psnr']} | {row['training_seconds']:.1f} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")


def run(args):
    from huggingface_hub import hf_hub_download
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    cfg = security.flow.DEFAULT_CONFIGS["cifar10"]
    checkpoint = Path(hf_hub_download("WayBob/FlowMatching-Unet-Celeb-64x64", cfg.filename, local_files_only=True))
    key = torch.load(args.key, map_location=device, weights_only=True)
    if len(key["bits"]) != 32:
        raise ValueError("H32 only")
    projection, codes = key["projection"], key["codes"]
    code = pilot.baseline.code_for(key, "dynamics")
    direction = projection @ code
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    checkpoint_sha256=security.digest(checkpoint), key_sha256=security.digest(args.key),
                    source_hashes={p.name: security.digest(p) for p in [Path(__file__), Path(pilot.__file__),
                                  Path(security.__file__), Path(security.flow.__file__)]},
                    objective="MSE to whole frozen-teacher target + weighted projected TARGET ERROR; no correlation reward",
                    train_states="Half data interpolants, half cached clean rollouts; disjoint evaluation bank/seeds",
                    optimizer="AdamW, constant lr, zero weight decay; no amplitude reward",
                    inference="Standalone learned model, ordinary Euler, fixed [-5,5] clipping",
                    trainable=args.parameterization)
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != manifest:
        raise ValueError("Output already belongs to another configuration")
    security.write_json(config_path, manifest)
    result_path = args.output / "results.json"
    results = json.loads(result_path.read_text()) if result_path.exists() else dict(rows=[])
    results.update(state="running")
    write_results(args.output, results)
    base = security.flow.build_model(cfg, device)
    security.flow.load_state_dict_flexible(base, str(checkpoint), device)
    base.eval().requires_grad_(False)
    dataset = CIFAR10(str(security.ROOT / "data"), train=True, download=False)
    pool = torch.from_numpy(dataset.data[:20000]).permute(0, 3, 1, 2).to(device).float() / 127.5 - 1
    labels = torch.tensor(dataset.targets[:20000], device=device)
    test = CIFAR10(str(security.ROOT / "data"), train=False, download=False)
    test_pool = torch.from_numpy(test.data).permute(0, 3, 1, 2).to(device).float() / 127.5 - 1
    test_labels = torch.tensor(test.targets, device=device)
    print("Caching clean training and held-out trajectories", flush=True)
    bank = trajectory_bank(base, device, args.seed + 1)
    heldout_bank = trajectory_bank(base, device, args.seed + 2, count=64)
    rng = torch.Generator(device=device).manual_seed(args.seed + 3)
    noise = torch.randn(args.samples, 3, 32, 32, generator=rng, device=device)
    sample_labels = torch.arange(args.samples, device=device) % 10
    grid = torch.linspace(0, 1, args.sampler_steps + 1, dtype=torch.float64).tolist()
    clean, clean_signature = sample(base, noise, sample_labels, grid, projection, direction, args.batch)
    save_image(clean[:64].add(1).div(2).clamp(0, 1), args.output / "clean.png", nrow=8)
    for arm in args.arms:
        folder = args.output / arm
        folder.mkdir(exist_ok=True)
        torch.manual_seed(args.seed)
        model = copy.deepcopy(base)
        if args.parameterization == "lora":
            security.flow.add_lora(model, 16, 16, "both")
            security.flow.freeze_non_lora(model, "time_embed,out.")
        else:
            model.requires_grad_(True)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.)
        rng = torch.Generator(device=device).manual_seed(args.seed + 4)
        completed, elapsed, history = 0, 0., []
        latest = folder / "latest.pt"
        if latest.exists():
            saved = torch.load(latest, map_location=device, weights_only=True)
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            completed, elapsed, history = saved["step"], saved["seconds"], saved["history"]
            rng.set_state(saved["rng"].cpu())
            torch.set_rng_state(saved["cpu_rng"].cpu())
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
            del saved
        print(f"TRAIN {arm} {args.parameterization}: {sum(p.numel() for p in parameters)} parameters, start={completed}", flush=True)
        for step in range(completed + 1, max(args.checkpoints) + 1):
            torch.cuda.synchronize()
            started = time.monotonic()
            # Dropout would change the deterministic teacher-fitting problem.
            model.eval()
            optimizer.zero_grad(set_to_none=True)
            x, t, y = training_queries(pool, labels, bank, args.batch, rng)
            inputs, target, _ = teacher_target(base, x, t, y, direction, args.epsilon, arm)
            pred = security.flow.model_velocity(model, inputs, t, y)
            loss, mse, pmse = fitting_loss(pred, target, projection, args.projected_weight)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed += time.monotonic() - started
            history.append(dict(step=step, loss=float(loss), mse=float(mse), projected_mse=float(pmse)))
            if step % 50 == 0:
                print(f"TRAIN {arm}/{step}: mse={float(mse):.6g} projected={float(pmse):.6g} train_seconds={elapsed:.1f}", flush=True)
            if step in args.checkpoints:
                snapshot = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                                seconds=elapsed, history=history, rng=rng.get_state(),
                                cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
                torch.save(snapshot, latest.with_suffix(".tmp"))
                latest.with_suffix(".tmp").replace(latest)
                del snapshot
                security.flow.save_adapter(model, folder / f"model_step{step}.pt")
                fit = fit_diagnostics(model, base, test_pool, test_labels, heldout_bank, projection, code,
                                      args.epsilon, arm, args.seed + 5)
                endpoint, signature = sample(model, noise, sample_labels, grid, projection, direction, args.batch)
                detection = pilot.detection_metrics(signature, clean_signature, code, codes, key["bits"], args.trials)
                metrics = pilot.endpoint_metrics(clean, endpoint)
                row = dict(arm=arm, step=step, fit=fit, endpoint=metrics, detection=detection,
                           training_seconds=elapsed, trainable_parameters=sum(p.numel() for p in parameters))
                results["rows"].append(row)
                write_results(args.output, results)
                np.savez_compressed(folder / f"evaluation_step{step}.npz", endpoint=endpoint.numpy(),
                                    clean_endpoint=clean.numpy(), signature=signature.cpu().numpy(),
                                    clean_signature=clean_signature.cpu().numpy())
                save_image(endpoint[:64].add(1).div(2).clamp(0, 1), folder / f"samples_step{step}.png", nrow=8)
                print(f"RESULT {arm}/{step}: eps={fit['effective_epsilon']:.4f}, exact={detection['recovery_percent']:.1f}, "
                      f"bits={detection['bit_accuracy_percent']:.2f}, rmse={metrics['unclipped_rmse']:.6g}, "
                      f"psnr={metrics['image_psnr']}", flush=True)
        del model, optimizer, parameters
        torch.cuda.empty_cache()
    results["state"] = "complete"
    write_results(args.output, results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("table_runs/endpoint_loop_training"))
    parser.add_argument("--key", type=Path, default=Path("table_runs/baseline_first/tune/32/0/key.pt"))
    parser.add_argument("--arms", nargs="+", choices=("control", "additive", "loop"), default=["control", "additive", "loop"])
    parser.add_argument("--parameterization", choices=("lora", "full"), default="lora")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[100, 300, 1000])
    parser.add_argument("--epsilon", type=float, default=1.5)
    parser.add_argument("--projected-weight", type=float, default=10.)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--sampler-steps", type=int, default=100)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--seed", type=int, default=73641)
    args = parser.parse_args()
    if args.samples < args.trials or min(args.checkpoints) < 1 or args.projected_weight < 0:
        parser.error("Invalid sample, checkpoint, or loss settings")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
