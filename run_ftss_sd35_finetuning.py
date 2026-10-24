"""Paired, unwatermarked SD3.5 fine-tuning and remaining-flow sensitivity pilot."""

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def interpolation(clean, noise, sigma):
    return (1 - sigma) * clean + sigma * noise, noise - clean


def report(root, results):
    rows = results["measurements"]
    lines = ["# Clean SD3.5 Fine-Tuning Sensitivity", "", f"State: {results['state']}", "",
             "Unwatermarked clean base, cat-conditioned CFG=1, 256px by default. "
             "LoRA fine-tuning on real CIFAR-10 cats. This is a low-resolution pilot, "
             "not native-resolution or full-parameter SD. Nearest-neighbor distances "
             "are in VAE latent space and are not a near-copy classification.", "",
             "| Seed | Images | Steps | Perturbation | FTSS minimum | Ratio to base | Train loss | Reference loss |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        for curve in row["curves"]:
            lines.append(f"| {row['seed']} | {row['size']} | {row['step']} | "
                         f"{curve['relative_epsilon']} | {curve['minimum']:.4f} | "
                         f"{curve['minimum_ratio_to_base']:.4f} | "
                         f"{row['flow_loss']['train']:.4f} | {row['flow_loss']['reference']:.4f} |")
    (root / "report.md").write_text("\n".join(lines) + "\n")
    if results["state"] != "complete":
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sizes = sorted({row["size"] for row in rows})
    figure, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 4), squeeze=False)
    for size, axis in zip(sizes, axes[0]):
        for step in sorted({row["step"] for row in rows}):
            selected = [row for row in rows if row["size"] == size and row["step"] == step]
            for epsilon in sorted({c["relative_epsilon"] for row in selected for c in row["curves"]}):
                curves = [c for row in selected for c in row["curves"] if c["relative_epsilon"] == epsilon]
                values = np.array([c["rms_curve"] for c in curves])
                x = curves[0]["generation_times"]
                line, = axis.plot(x, values.mean(0), label=f"{step} steps, epsilon={epsilon}")
                axis.fill_between(x, values.min(0), values.max(0), color=line.get_color(), alpha=.12)
        axis.axhline(1, color="black", linewidth=.6)
        axis.set(title=f"{size} training images", xlabel="Generation time", ylabel="RMS sensitivity")
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(root / "sensitivity_curves.png", dpi=180)
    figure.savefig(root / "sensitivity_curves.pdf")
    plt.close(figure)


@torch.no_grad()
def integrate(velocity, x, sigmas, start=0):
    for i in range(start, len(sigmas) - 1):
        x = x + (sigmas[i + 1] - sigmas[i]) * velocity(x, sigmas[i])
    return x


@torch.no_grad()
def sensitivity(velocity, noise, sigmas, indices, directions, epsilon, batch):
    def batched_velocity(x, sigma):
        return torch.cat([velocity(x[offset:offset + batch], sigma)
                          for offset in range(0, len(x), batch)])

    trajectory = [noise.clone()]
    for i in range(len(sigmas) - 1):
        x = trajectory[-1]
        trajectory.append(x + (sigmas[i + 1] - sigmas[i]) * batched_velocity(x, sigmas[i]))
    terminal = trajectory[-1]
    gains = []
    for j, index in enumerate(indices):
        state = trajectory[index]
        unit = F.normalize(directions[j].flatten(1), dim=1).reshape_as(state)
        radius = epsilon * state.flatten(1).norm(dim=1).clamp_min(1e-6)
        delta = unit * radius[:, None, None, None]
        # Identical batch shapes permit reuse of the nominal endpoint for every injection.
        perturbed = []
        for offset in range(0, len(state), batch):
            perturbed.append(integrate(velocity, (state + delta)[offset:offset + batch],
                                      sigmas, index))
        difference = torch.cat(perturbed) - terminal
        gains.append((difference.flatten(1).norm(dim=1) / radius).cpu())
    gains = torch.stack(gains).double()
    rms = gains.square().mean(dim=1).sqrt()
    if not torch.isfinite(gains).all():
        raise RuntimeError("Nonfinite SD sensitivity measurement")
    return {"relative_epsilon": epsilon, "generation_times": [1 - sigmas[i] for i in indices],
            "gains": gains.tolist(), "rms_curve": rms.tolist(),
            "mean_gain_curve": gains.mean(dim=1).tolist(), "minimum": rms.min().item()}, terminal


def parse_args():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, default=Path("table_runs/ftss_sd35_finetuning"))
    p.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    p.add_argument("--sizes", type=int, nargs="+", default=[10, 200])
    p.add_argument("--seeds", type=int, nargs="+", default=[24101, 24102, 24103])
    p.add_argument("--steps", type=int, nargs="+", default=[100, 300, 1000])
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accumulation", type=int, default=1)
    p.add_argument("--probes", type=int, default=8)
    p.add_argument("--times", type=int, default=8)
    p.add_argument("--ode-steps", type=int, default=40)
    p.add_argument("--epsilons", type=float, nargs="+", default=[0.02, 0.1])
    a = p.parse_args()
    if (min(a.sizes) < 2 or max(a.sizes) > 2000 or min(a.steps) < 1
            or a.resolution < 256 or a.resolution % 16 or min(a.epsilons) <= 0
            or min(a.batch, a.accumulation, a.probes, a.rank) < 1
            or a.times < 2 or a.ode_steps < a.times or a.lr <= 0):
        p.error("Invalid study settings")
    a.steps = sorted(set(a.steps))
    return a


def main():
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from diffusers import StableDiffusion3Pipeline
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from torchvision.datasets import CIFAR10
    import psutil

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "config.json").exists():
        raise FileExistsError("Use a fresh output directory to preserve existing measurements")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  prompt="a photo of a cat", watermark=False,
                  data="CIFAR-10 training cats, disjoint fine-tuning/reference pools",
                  data_caveat="Prior inclusion in SD pretraining is unknown",
                  protocol="Uniform sigma Euler, CFG=1; t_generation=1-sigma; FP32 states",
                  scope="Low-resolution LoRA pilot; not full-parameter or native-resolution SD",
                  outcome="Paired FTSS curves, held-out flow loss and latent nearest neighbors")
    write_json(args.output / "config.json", config)
    results = {"state": "running", "measurements": []}
    write_json(args.output / "results.json", results)
    begin = time.monotonic()
    try:
        print("Loading local clean SD3.5", flush=True)
        pipe = StableDiffusion3Pipeline.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, local_files_only=True,
            low_cpu_mem_usage=True, device_map="balanced",
            max_memory={0: 8 * 1024**3, "cpu": int(psutil.virtual_memory().available * .7)})
        with torch.no_grad():
            prompt, _, pooled, _ = pipe.encode_prompt(
                prompt=config["prompt"], prompt_2=config["prompt"], prompt_3=config["prompt"],
                device="cuda", do_classifier_free_guidance=False)
        pipe.remove_all_hooks()
        pipe.text_encoder = pipe.text_encoder_2 = pipe.text_encoder_3 = None
        pipe.reset_device_map()
        gc.collect()
        torch.cuda.empty_cache()
        data = CIFAR10("data", train=True, download=False)
        cat_indices = np.flatnonzero(np.asarray(data.targets) == 3)
        count = 2 * max(args.sizes)
        indices = np.random.default_rng(24000).permutation(cat_indices)[:count]
        write_json(args.output / "data_indices.json", indices.tolist())
        vae = pipe.vae.to("cuda").eval()
        latents = []
        with torch.no_grad():
            for offset in range(0, count, args.batch):
                image = torch.from_numpy(data.data[indices[offset:offset + args.batch]]).to("cuda")
                image = image.permute(0, 3, 1, 2).float() / 127.5 - 1
                image = F.interpolate(image, (args.resolution, args.resolution),
                                      mode="bilinear", align_corners=False)
                z = vae.encode(image.to(vae.dtype)).latent_dist.mode()
                latents.append(((z - vae.config.shift_factor) * vae.config.scaling_factor).cpu())
        latent_pool = torch.cat(latents).float()
        torch.save(latent_pool, args.output / "latent_pool.pt")
        vae.to("cpu")
        transformer = pipe.transformer.to("cuda")
        transformer.enable_gradient_checkpointing()
        targets = [name for name, layer in transformer.named_modules()
                   if isinstance(layer, torch.nn.Linear) and
                   (any(key in name for key in ("to_q", "to_k", "to_v", "to_out.0"))
                    or "ff" in name)]
        torch.manual_seed(24000)
        transformer = get_peft_model(transformer, LoraConfig(
            r=args.rank, lora_alpha=args.rank, target_modules=targets,
            lora_dropout=0, bias="none"))
        initial = {name: p.detach().clone() for name, p in transformer.named_parameters()
                   if p.requires_grad}
        config["trainable_parameters"] = sum(p.numel() for p in initial.values())
        write_json(args.output / "config.json", config)
        transformer.print_trainable_parameters()
        prompt, pooled = prompt.cuda(), pooled.cuda()

        def velocity(x, sigma):
            return transformer(hidden_states=x.to(torch.bfloat16),
                               timestep=torch.full((len(x),), float(sigma) * 1000,
                                                   device="cuda", dtype=torch.float32),
                               encoder_hidden_states=prompt.expand(len(x), -1, -1),
                               pooled_projections=pooled.expand(len(x), -1),
                               return_dict=False)[0].float()

        sigmas = np.linspace(1., 0., args.ode_steps + 1).tolist()
        injection_indices = np.linspace(0, args.ode_steps - 1, args.times, dtype=int).tolist()
        for seed in args.seeds:
            generator = torch.Generator("cuda").manual_seed(seed)
            shape = tuple(latent_pool.shape[1:])
            noise = torch.randn(args.probes, *shape, device="cuda", generator=generator)
            directions = torch.randn(args.times, *noise.shape, device="cuda", generator=generator)
            baseline = {}
            for size in args.sizes:
                torch.manual_seed(seed)
                order = np.random.default_rng(seed).permutation(max(args.sizes))
                pool = latent_pool[order[:size]].cuda()
                reference = latent_pool[max(args.sizes):max(args.sizes) + size].cuda()
                write_json(args.output / f"indices_seed{seed}_n{size}.json", order[:size].tolist())
                with torch.no_grad():
                    for name, parameter in transformer.named_parameters():
                        if name in initial:
                            parameter.copy_(initial[name])
                optimizer = torch.optim.AdamW([p for p in transformer.parameters() if p.requires_grad],
                                              lr=args.lr, weight_decay=.01)
                train_rng = torch.Generator("cuda").manual_seed(seed + 100)

                @torch.no_grad()
                def measure(step):
                    transformer.eval()
                    curves = []
                    for epsilon in args.epsilons:
                        if step == 0 and epsilon in baseline:
                            curve, terminal = copy.deepcopy(baseline[epsilon])
                        else:
                            curve, terminal = sensitivity(velocity, noise, sigmas,
                                                          injection_indices, directions,
                                                          epsilon, args.batch)
                        if step == 0:
                            baseline[epsilon] = (copy.deepcopy(curve), terminal.clone())
                        curve["minimum_ratio_to_base"] = curve["minimum"] / baseline[epsilon][0]["minimum"]
                        curves.append(curve)
                    losses = {}
                    for label, images in (("train", pool), ("reference", reference)):
                        eval_rng = torch.Generator("cuda").manual_seed(seed + 200)
                        total = 0.
                        for offset in range(0, size, args.batch):
                            clean = images[offset:offset + args.batch]
                            eps = torch.randn(clean.shape, device="cuda", generator=eval_rng)
                            sigma = torch.rand(len(clean), 1, 1, 1, device="cuda", generator=eval_rng)
                            x, target = interpolation(clean, eps, sigma)
                            # Per-example timesteps, unlike scalar ODE evaluation.
                            pred = transformer(hidden_states=x.to(torch.bfloat16),
                                timestep=sigma.flatten() * 1000,
                                encoder_hidden_states=prompt.expand(len(x), -1, -1),
                                pooled_projections=pooled.expand(len(x), -1), return_dict=False)[0].float()
                            total += (pred - target).square().flatten(1).mean(1).sum().item()
                        losses[label] = total / size
                    distances = [torch.cdist(terminal.flatten(1), images.flatten(1))
                                 .min(dim=1).values / np.sqrt(terminal[0].numel())
                                 for images in (pool, reference)]
                    row = dict(seed=seed, size=size, step=step, curves=curves, flow_loss=losses,
                               latent_train_rmse=distances[0].tolist(),
                               latent_reference_rmse=distances[1].tolist(),
                               elapsed_seconds=time.monotonic() - begin)
                    torch.save(terminal.cpu(), args.output / f"seed{seed}_n{size}_step{step}_samples.pt")
                    results["measurements"].append(row)
                    write_json(args.output / "results.json", results)
                    report(args.output, results)
                    print(f"MEASURE seed={seed} n={size} step={step} "
                          f"ratios={[round(c['minimum_ratio_to_base'], 4) for c in curves]} "
                          f"loss={losses}", flush=True)
                    transformer.train()

                measure(0)
                train_begin = time.monotonic()
                for step in range(1, max(args.steps) + 1):
                    optimizer.zero_grad(set_to_none=True)
                    loss_value = 0.
                    for _ in range(args.accumulation):
                        clean = pool[torch.randint(size, (args.batch,), device="cuda", generator=train_rng)]
                        eps = torch.randn(clean.shape, device="cuda", generator=train_rng)
                        sigma = torch.rand(args.batch, 1, 1, 1, device="cuda", generator=train_rng)
                        x, target = interpolation(clean, eps, sigma)
                        pred = transformer(hidden_states=x.to(torch.bfloat16),
                            timestep=sigma.flatten() * 1000,
                            encoder_hidden_states=prompt.expand(len(x), -1, -1),
                            pooled_projections=pooled.expand(len(x), -1), return_dict=False)[0].float()
                        loss = (pred - target).square().mean() / args.accumulation
                        loss.backward()
                        loss_value += loss.item()
                    torch.nn.utils.clip_grad_norm_([p for p in transformer.parameters() if p.requires_grad], 1.)
                    optimizer.step()
                    if step % 25 == 0 or step == 1:
                        print(f"TRAIN seed={seed} n={size} {step}/{max(args.steps)} "
                              f"loss={loss_value:.5f} elapsed={time.monotonic()-train_begin:.0f}s", flush=True)
                    if step in args.steps:
                        torch.save({"adapter": get_peft_model_state_dict(transformer),
                                    "optimizer": optimizer.state_dict(), "step": step,
                                    "rng": train_rng.get_state()},
                                   args.output / f"seed{seed}_n{size}_step{step}.pt")
                        measure(step)
                del optimizer
        results["state"] = "complete"
    except Exception as exc:
        results.update(state="failed", error=repr(exc))
        raise
    finally:
        write_json(args.output / "results.json", results)
        report(args.output, results)


if __name__ == "__main__":
    main()
