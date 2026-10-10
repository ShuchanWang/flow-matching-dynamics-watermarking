"""SD3.5 bit diagnostics and optional repair probes; base subtraction is not the main verifier."""

import argparse
import gc
import json
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from watermark_codebooks import decode_signature, message_code, hypercube_margin_loss


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
    parser.add_argument("--repair-steps", type=int, default=0)
    parser.add_argument("--repair-lr", type=float, default=5e-5)
    parser.add_argument("--repair-proj-weight", type=float, default=10)
    parser.add_argument("--repair-bit-weight", type=float, default=1)
    parser.add_argument("--repair-grad-accum", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if min(args.queries, args.trials, args.batch_size) < 1:
        parser.error("Query/trial/batch counts must be positive")
    if args.repair_steps < 0 or args.repair_lr <= 0 or args.repair_grad_accum < 1:
        parser.error("Invalid repair budget, learning rate, or accumulation count")
    if min(args.repair_proj_weight, args.repair_bit_weight) < 0:
        parser.error("Repair loss weights must be nonnegative")
    config = json.loads((args.run_dir / "sweep_results.json").read_text())["config"]
    key = torch.load(args.run_dir / "codebook.pt", map_location="cpu", weights_only=False)
    message = key["message"]
    if message != config["wm_message"]:
        raise ValueError("Saved key and run message disagree")
    mode = key.get("codebook_mode", config.get("codebook_mode", "auto"))
    if args.repair_steps and mode != "hypercube":
        parser.error("The repair probe is only defined for hypercube adapters")
    expected = message_code(message, key["codes"], mode)
    torch.testing.assert_close(key["WM_CODE"], expected)
    pool = torch.load(args.run_dir / f"latent_pool_{config['latent_size']}_{config['n_train_latents']}.pt",
                      map_location="cpu", weights_only=False)
    from diffusers import StableDiffusion3Pipeline
    from peft import PeftModel
    print("Loading SD3.5 with balanced placement for prompt encoding", flush=True)
    pipe = StableDiffusion3Pipeline.from_pretrained(config["model"], torch_dtype=torch.bfloat16,
                                                   low_cpu_mem_usage=True, device_map="balanced")
    with torch.no_grad():
        prompt, _, pooled, _ = pipe.encode_prompt(prompt=config["prompt"], prompt_2=config["prompt"],
                                                 prompt_3=config["prompt"], device="cuda",
                                                 num_images_per_prompt=1, do_classifier_free_guidance=False)
    backbone = pipe.transformer
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    adapter_path = args.adapter_path or args.run_dir / "final"
    model = PeftModel.from_pretrained(backbone, adapter_path,
                                     is_trainable=args.repair_steps > 0).to("cuda").eval()
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
        for step in range(args.repair_steps):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(args.repair_grad_accum):
                index = torch.randint(len(pool), (1,), device="cuda", generator=generator).cpu()
                x1 = pool[index].cuda().to(torch.bfloat16)
                x0 = torch.randn(x1.shape, generator=generator, device="cuda", dtype=torch.bfloat16)
                t = torch.tensor([rng.random()], device="cuda", dtype=torch.bfloat16)
                view = t.view(-1, 1, 1, 1)
                x = ((1 - view) * x0 + view * x1).to(torch.bfloat16)
                carrier = torch.sin(2 * math.pi * t.float()).unsqueeze(1)
                with torch.no_grad(), model.disable_adapter(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    clean = velocity(x, t)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    prediction = velocity(x, t)
                residual = (prediction - clean).flatten(1)
                target = config["wm_eps"] * carrier * code
                projected = residual @ P
                correlation = (carrier * (projected * code).sum(1, keepdim=True)).mean()
                reward = -torch.tanh(config["wm_tanh_scale"] * correlation / (0.5 * config["wm_eps"] + 1e-8))
                loss = (F.mse_loss(residual, target @ P.T) + config["wm_lambda"] * reward
                        + args.repair_proj_weight * F.mse_loss(projected, target)
                        + args.repair_bit_weight * hypercube_margin_loss(prediction.flatten(1) @ P, codes,
                                                                        message, carrier, config["wm_eps"]))
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
            "repair": {name: value for name, value in vars(args).items() if name.startswith("repair_")}
        }, indent=2) + "\n")
        del optimizer, scheduler
        model.eval()
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
            print(f"trial={trial + 1}/{args.trials} queries={args.queries}", flush=True)
    wm, clean = torch.stack(wm_signatures), torch.stack(clean_signatures)
    result = {"source": str(args.run_dir), "input_adapter": str(adapter_path),
              "message": message, "codebook_mode": mode,
              "queries": args.queries, "trials": args.trials, "seed": args.seed,
              "diagnostic_only": True, "wm": summarize(wm, codes, message, mode),
              "repair": {name: value for name, value in vars(args).items() if name.startswith("repair_")},
              "quality_evaluated": False,
              "clean": summarize(clean, codes, message, mode),
              "paired_base_subtracted": summarize(wm - clean, codes, message, mode),
              "wm_signatures": wm.cpu().tolist(), "clean_signatures": clean.cpu().tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({label: {k: result[label][k] for k in ("exact_recovery", "bit_accuracy")}
                      for label in ("wm", "clean", "paired_base_subtracted")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
