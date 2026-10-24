"""SD3.5 bit diagnostics and optional repair probes; base subtraction is not the main verifier."""

import argparse
import gc
import json
import math
from pathlib import Path
import random
from contextlib import nullcontext

import numpy as np

import torch
import torch.nn.functional as F

from watermark_codebooks import decode_signature, message_code, hypercube_margin_loss


@torch.no_grad()
def evaluate_quality(model, velocity, config, count, batch_size=8,
                     standard_pipeline=None, prompt=None, pooled=None, image_path=None):
    """Match the original SD runner's small-sample base-reference Inception protocol."""
    from diffusers import AutoencoderKL
    from scipy.linalg import sqrtm
    from torchvision.models import inception_v3, Inception_V3_Weights
    if standard_pipeline is None:
        vae = AutoencoderKL.from_pretrained(config["model"], subfolder="vae", torch_dtype=torch.bfloat16).cuda().eval()
    else:
        standard_pipeline.vae.to("cuda")
        standard_pipeline.set_progress_bar_config(disable=True)
        vae = standard_pipeline.vae
    def images(marked, offset):
        output = []
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            if standard_pipeline is not None:
                generators = [torch.Generator("cuda").manual_seed(offset + i) for i in range(start, end)]
                with nullcontext() if marked else model.disable_adapter():
                    imgs = standard_pipeline(prompt_embeds=prompt.expand(end - start, -1, -1),
                                             pooled_prompt_embeds=pooled.expand(end - start, -1),
                                             height=config["latent_size"] * standard_pipeline.vae_scale_factor,
                                             width=config["latent_size"] * standard_pipeline.vae_scale_factor,
                                             num_inference_steps=config["n_eval_steps"], guidance_scale=1.0,
                                             generator=generators, output_type="pt").images
                output.append(imgs.float().cpu())
                continue
            x = torch.stack([torch.randn(16, config["latent_size"], config["latent_size"], device="cuda",
                                        dtype=torch.bfloat16,
                                        generator=torch.Generator("cuda").manual_seed(offset + i))
                             for i in range(start, min(start + batch_size, count))])
            times = torch.linspace(0, 1, config["n_eval_steps"] + 1, device="cuda")
            with nullcontext() if marked else model.disable_adapter():
                for i in range(config["n_eval_steps"]):
                    t = torch.full((len(x),), times[i].item(), device="cuda", dtype=torch.bfloat16)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        v = velocity(x, t)
                    x = x + v.to(torch.bfloat16) * (times[i + 1].item() - times[i].item())
            decoded = vae.decode(x / vae.config.scaling_factor).sample
            output.append(((decoded.clamp(-1, 1) + 1) / 2).float().cpu())
        return torch.cat(output)
    print(f"Quality evaluation: {count} samples per distribution", flush=True)
    reference, clean, marked = images(False, 100000), images(False, 50000), images(True, 50000)
    if image_path is not None:
        from torchvision.utils import save_image
        save_image(torch.cat((clean[:6], marked[:6])), image_path, nrow=6)
    del vae
    if standard_pipeline is not None:
        standard_pipeline.vae.to("cpu")
    torch.cuda.empty_cache()
    inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                             transform_input=False).cuda().eval()
    inception.fc = torch.nn.Identity()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    def features(imgs):
        rows = []
        for start in range(0, len(imgs), 16):
            batch = F.interpolate(imgs[start:start + 16].cuda(), size=(299, 299), mode="bilinear", align_corners=False)
            rows.append(inception((batch - mean) / std).cpu().numpy())
        return np.concatenate(rows)
    ref, base, wm = features(reference), features(clean), features(marked)
    del inception
    torch.cuda.empty_cache()
    def fid(a, b):
        covariance_a = np.cov(a, rowvar=False) + 1e-6 * np.eye(a.shape[1])
        covariance_b = np.cov(b, rowvar=False) + 1e-6 * np.eye(b.shape[1])
        diff = a.mean(0) - b.mean(0)
        return float(diff @ diff + np.trace(covariance_a + covariance_b -
                                            2 * sqrtm(covariance_a @ covariance_b).real))
    clean_fid, wm_fid = fid(ref, base), fid(ref, wm)
    return {"fid_real_clean": clean_fid, "fid_real_wm": wm_fid,
            "fid_ratio": wm_fid / max(clean_fid, 1e-6), "n_samples": count,
            "sampling_protocol": "diffusers_sd3" if standard_pipeline is not None else "legacy_forward_euler",
            "feature_extractor": "torchvision_inception_v3_imagenet_prelogits",
            "guidance_scale": 1.0 if standard_pipeline is not None else None,
            "n_sampling_steps": config["n_eval_steps"],
            "reference": "base_samples", "warning": "Small-sample estimate, not population FID"}


def summarize(signatures, codes, message, mode):
    truth = tuple(map(int, message))
    decoded = [decode_signature(s, codes, len(message), mode)[0] for s in signatures]
    result = {"exact_recovery": 100 * sum(bits == truth for bits in decoded) / len(decoded),
              "bit_accuracy": 100 * sum(sum(a == b for a, b in zip(bits, truth)) for bits in decoded)
              / (len(decoded) * len(truth)),
              "decoded_messages": ["".join(map(str, bits)) for bits in decoded]}
    if mode == "hypercube":
        coordinates = signatures @ codes.T
        signed = coordinates * coordinates.new_tensor([2 * b - 1 for b in truth])
        result.update(mean_signed_coordinates=signed.mean(0).tolist(),
                      std_signed_coordinates=signed.std(0, unbiased=False).tolist(),
                      wrong_bit_frequencies=[sum(bits[i] != truth[i] for bits in decoded) / len(decoded)
                                             for i in range(len(truth))],
                      minimum_signed_coordinate_per_trial=signed.min(1).values.tolist())
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, help="Alternative adapter trained with this run's saved key")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=256)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--loading-gpu-budget-gib", type=int,
                        help="Optional GPU placement budget while loading the pretrained pipeline")
    parser.add_argument("--quality-samples", type=int, default=0,
                        help="Optional base-reference image FID, using the original generation settings")
    parser.add_argument("--quality-only", action="store_true",
                        help="Evaluate sample quality only, without training or repeating detection")
    parser.add_argument("--quality-batch-size", type=int, default=8)
    parser.add_argument("--quality-protocol", choices=("legacy", "standard"), default="legacy",
                        help="Keep legacy measurements separate from the official SD3 pipeline")
    parser.add_argument("--repair-steps", type=int, default=0)
    parser.add_argument("--repair-lr", type=float, default=5e-5)
    parser.add_argument("--repair-epsilon", type=float,
                        help="Optional compensated-target strength; defaults to the original epsilon")
    parser.add_argument("--repair-proj-weight", type=float, default=10)
    parser.add_argument("--repair-bit-weight", type=float, default=1)
    parser.add_argument("--repair-grad-accum", type=int, default=4)
    parser.add_argument("--repair-batch-size", type=int, default=1)
    parser.add_argument("--repair-clean-calibration", type=Path,
                        help="Prior diagnostic JSON with the same clean model, projection and query pool")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if min(args.queries, args.trials, args.batch_size) < 1:
        parser.error("Query/trial/batch counts must be positive")
    if args.repair_steps < 0 or args.repair_lr <= 0 or min(args.repair_grad_accum, args.repair_batch_size) < 1:
        parser.error("Invalid repair budget, learning rate, or accumulation count")
    if args.repair_epsilon is not None and args.repair_epsilon <= 0:
        parser.error("Repair epsilon must be positive")
    if min(args.repair_proj_weight, args.repair_bit_weight) < 0:
        parser.error("Repair loss weights must be nonnegative")
    if args.repair_clean_calibration and not args.repair_steps:
        parser.error("Clean compensation requires a repair budget")
    if args.quality_samples < 0 or args.quality_samples == 1:
        parser.error("Quality samples must be zero or at least two")
    if args.quality_only and (args.quality_samples < 2 or args.repair_steps):
        parser.error("Quality-only requires at least two samples and no repair updates")
    if args.quality_batch_size < 1:
        parser.error("Quality batch size must be positive")
    if args.loading_gpu_budget_gib is not None and args.loading_gpu_budget_gib < 1:
        parser.error("Loading GPU budget must be positive")
    config = json.loads((args.run_dir / "sweep_results.json").read_text())["config"]
    key = torch.load(args.run_dir / "codebook.pt", map_location="cpu", weights_only=False)
    message = key.get("message", config["wm_message"])
    if message != config["wm_message"]:
        raise ValueError("Saved key and run message disagree")
    mode = key.get("codebook_mode", config.get("codebook_mode", "auto"))
    if args.repair_steps and mode != "hypercube":
        parser.error("The repair probe is only defined for hypercube adapters")
    expected = message_code(message, key["codes"], mode)
    torch.testing.assert_close(key["WM_CODE"], expected)
    pool = None if args.quality_only else torch.load(
        args.run_dir / f"latent_pool_{config['latent_size']}_{config['n_train_latents']}.pt",
        map_location="cpu", weights_only=False)
    from diffusers import StableDiffusion3Pipeline
    from peft import PeftModel
    print("Loading SD3.5 with balanced placement for prompt encoding", flush=True)
    placement = {}
    if args.loading_gpu_budget_gib is not None:
        import psutil
        placement["max_memory"] = {0: args.loading_gpu_budget_gib * 1024 ** 3,
                                   "cpu": int(psutil.virtual_memory().available * 0.8)}
    pipe = StableDiffusion3Pipeline.from_pretrained(config["model"], torch_dtype=torch.bfloat16,
                                                   low_cpu_mem_usage=True, device_map="balanced", **placement)
    with torch.no_grad():
        prompt, _, pooled, _ = pipe.encode_prompt(prompt=config["prompt"], prompt_2=config["prompt"],
                                                 prompt_3=config["prompt"], device="cuda",
                                                 num_images_per_prompt=1, do_classifier_free_guidance=False)
    backbone = pipe.transformer
    quality_pipe = None
    if args.quality_samples and args.quality_protocol == "standard":
        # Free text encoders before reset_device_map would copy their weights to CPU.
        pipe.remove_all_hooks()
        pipe.text_encoder = pipe.text_encoder_2 = pipe.text_encoder_3 = None
        pipe.tokenizer = pipe.tokenizer_2 = pipe.tokenizer_3 = None
        pipe.reset_device_map()
        quality_pipe = pipe
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    adapter_path = args.adapter_path or args.run_dir / "final"
    model = PeftModel.from_pretrained(backbone, adapter_path,
                                     is_trainable=args.repair_steps > 0).to("cuda").eval()
    if quality_pipe is not None:
        quality_pipe.transformer = model
    P, codes = key["P"].cuda().float(), key["codes"].cuda().float()
    prompt, pooled = prompt.cuda().to(torch.bfloat16), pooled.cuda().to(torch.bfloat16)
    def velocity(x, t):
        return model(hidden_states=x, timestep=t * 1000,
                     encoder_hidden_states=prompt.expand(len(x), -1, -1),
                     pooled_projections=pooled.expand(len(x), -1), return_dict=False)[0].float()
    if args.repair_steps:
        print(f"Repair probe: {args.repair_steps} updates, unchanged adapter rank", flush=True)
        model.enable_gradient_checkpointing()
        model.train()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=args.repair_lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.repair_steps)
        generator = torch.Generator("cuda").manual_seed(args.seed + 100000)
        rng = random.Random(args.seed + 100000)
        code = expected.cuda()
        epsilon = config["wm_eps"] if args.repair_epsilon is None else args.repair_epsilon
        coefficient = epsilon * code
        if args.repair_clean_calibration:
            calibration = json.loads(args.repair_clean_calibration.read_text())
            calibration_source = Path(calibration["source"])
            calibration_config = json.loads((calibration_source / "sweep_results.json").read_text())["config"]
            for name in ("model", "prompt", "latent_size", "n_train_latents"):
                if calibration_config[name] != config[name]:
                    raise ValueError(f"Clean calibration differs in {name}")
            calibration_key = torch.load(calibration_source / "codebook.pt", map_location="cpu", weights_only=False)
            calibration_pool = torch.load(calibration_source / f"latent_pool_{config['latent_size']}_{config['n_train_latents']}.pt",
                                          map_location="cpu", weights_only=False)
            if not torch.equal(calibration_key["P"], key["P"]) or not torch.equal(calibration_pool, pool):
                raise ValueError("Clean calibration must use the same projection and query pool")
            # E[sin^2(2*pi*t)]=1/2: compensate a fixed clean signature in the residual target.
            clean_bias = torch.tensor(calibration["clean_signatures"], device="cuda").mean(0)
            if clean_bias.shape != coefficient.shape or not torch.isfinite(clean_bias).all():
                raise ValueError("Invalid clean calibration signature")
            coefficient = coefficient - 2 * clean_bias
        for step in range(args.repair_steps):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(args.repair_grad_accum):
                index = torch.randint(len(pool), (args.repair_batch_size,), device="cuda", generator=generator).cpu()
                x1 = pool[index].cuda().to(torch.bfloat16)
                x0 = torch.randn(x1.shape, generator=generator, device="cuda", dtype=torch.bfloat16)
                t = torch.tensor([rng.random() for _ in range(args.repair_batch_size)], device="cuda", dtype=torch.bfloat16)
                view = t.view(-1, 1, 1, 1)
                x = ((1 - view) * x0 + view * x1).to(torch.bfloat16)
                carrier = torch.sin(2 * math.pi * t.float()).unsqueeze(1)
                with torch.no_grad(), model.disable_adapter(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    clean = velocity(x, t)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    prediction = velocity(x, t)
                residual = (prediction - clean).flatten(1)
                target = carrier * coefficient
                projected = residual @ P
                correlation = (carrier * (projected * code).sum(1, keepdim=True)).mean()
                reward = -torch.tanh(config["wm_tanh_scale"] * correlation / (0.5 * config["wm_eps"] + 1e-8))
                loss = (F.mse_loss(residual, target @ P.T) + config["wm_lambda"] * reward
                        + args.repair_proj_weight * F.mse_loss(projected, target)
                        + args.repair_bit_weight * hypercube_margin_loss(prediction.flatten(1) @ P, codes,
                                                                        message, carrier, epsilon))
                (loss / args.repair_grad_accum).backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            if step % 10 == 0 or step == args.repair_steps - 1:
                print(f"repair={step + 1}/{args.repair_steps} loss={float(loss.detach()):.6f}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.output.with_suffix(""))
        (args.output.with_suffix("") / "repair_metadata.json").write_text(json.dumps({
            "source_run": str(args.run_dir), "input_adapter": str(adapter_path), "message": message,
            "seed": args.seed,
            "repair": {name: str(value) if isinstance(value, Path) else value
                       for name, value in vars(args).items() if name.startswith("repair_")}
        }, indent=2) + "\n")
        del optimizer, scheduler
        model.eval()
    if args.quality_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        quality = evaluate_quality(model, velocity, config, args.quality_samples,
                                   args.quality_batch_size, quality_pipe, prompt, pooled)
        quality["batch_size"] = args.quality_batch_size
        result = {"source": str(args.run_dir), "input_adapter": str(adapter_path),
                  "message": message, "codebook_mode": mode, "quality_evaluated": True,
                  "quality_only": True, "training": False, "quality": quality}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        return
    wm_signatures, clean_signatures = [], []
    with torch.no_grad():
        for trial in range(args.trials):
            generator = torch.Generator("cuda").manual_seed(args.seed + trial)
            rng = random.Random(args.seed + trial)
            wm_sum = torch.zeros(P.shape[1], device="cuda")
            clean_sum = torch.zeros_like(wm_sum)
            for start in range(0, args.queries, args.batch_size):
                n = min(args.batch_size, args.queries - start)
                index = torch.randint(len(pool), (n,), generator=generator, device="cuda").cpu()
                x1 = pool[index].cuda().to(torch.bfloat16)
                x0 = torch.randn(x1.shape, generator=generator, device="cuda", dtype=torch.bfloat16)
                t = torch.tensor([rng.random() for _ in range(n)], device="cuda", dtype=torch.bfloat16)
                view = t.view(-1, 1, 1, 1)
                x = ((1 - view) * x0 + view * x1).to(torch.bfloat16)
                carrier = torch.sin(2 * math.pi * t.float()).unsqueeze(1)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    marked = velocity(x, t)
                    with model.disable_adapter():
                        clean = velocity(x, t)
                # Project in FP32, matching the original detector's numerical contract.
                with torch.amp.autocast("cuda", enabled=False):
                    wm_sum += (carrier * (marked.flatten(1) @ P)).sum(0)
                    clean_sum += (carrier * (clean.flatten(1) @ P)).sum(0)
            wm_signatures.append(wm_sum / args.queries)
            clean_signatures.append(clean_sum / args.queries)
            decoded = decode_signature(wm_signatures[-1], codes, len(message), mode)[0]
            bit_hits = sum(a == int(b) for a, b in zip(decoded, message))
            print(f"trial={trial + 1}/{args.trials} queries={args.queries} bits={bit_hits}/{len(message)}", flush=True)
    wm, clean = torch.stack(wm_signatures), torch.stack(clean_signatures)
    result = {"source": str(args.run_dir), "input_adapter": str(adapter_path),
              "message": message, "codebook_mode": mode,
              "queries": args.queries, "trials": args.trials, "seed": args.seed,
              "query_batch_size": args.batch_size,
              "diagnostic_only": True, "wm": summarize(wm, codes, message, mode),
              "repair": {name: str(value) if isinstance(value, Path) else value
                         for name, value in vars(args).items() if name.startswith("repair_")},
              "quality_evaluated": False,
              "clean": summarize(clean, codes, message, mode),
              "paired_base_subtracted": summarize(wm - clean, codes, message, mode),
              "wm_signatures": wm.cpu().tolist(), "clean_signatures": clean.cpu().tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.quality_samples:
        result["quality"] = evaluate_quality(model, velocity, config, args.quality_samples, args.quality_batch_size,
                                             quality_pipe, prompt, pooled,
                                             args.output.with_suffix(".png") if quality_pipe is not None else None)
        result["quality"]["batch_size"] = args.quality_batch_size
        result["quality_evaluated"] = True
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({label: {k: result[label][k] for k in ("exact_recovery", "bit_accuracy")}
                      for label in ("wm", "clean", "paired_base_subtracted")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
