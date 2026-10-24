"""Generate genuine paired SD3.5 samples and a publication comparison figure."""

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compose(root, manifest, figure_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, axes = plt.subplots(2, len(manifest["seeds"]),
                             figsize=(3.1 * len(manifest["seeds"]) + .7, 6.6), squeeze=False)
    for column, seed in enumerate(manifest["seeds"]):
        for row, label in enumerate(("clean", "watermarked")):
            with Image.open(root / f"{label}_{seed}.png") as sample:
                axes[row, column].imshow(sample)
            axes[row, column].set_axis_off()
        if "subjects" in manifest:
            axes[0, column].set_title(manifest["subjects"][column], fontsize=12, pad=6)
    fig.subplots_adjust(left=0.07, right=1, top=1, bottom=0, wspace=0.015, hspace=0.025)
    fig.text(0.025, 0.75, "Clean", rotation=90, ha="center", va="center", fontsize=14)
    fig.text(0.025, 0.25, "Watermarked", rotation=90, ha="center", va="center", fontsize=14)
    destination = Path("paper/figures")
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(destination / f"{figure_name}.{extension}", dpi=300,
                    facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=Path("table_runs/sd35_sample_figure"))
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--varied-subjects", action="store_true")
    parser.add_argument("--figure-name", default="sd35_clean_watermarked")
    parser.add_argument("--seeds", type=int, nargs="+", help="Predefined paired noise seeds")
    args = parser.parse_args()
    if args.size < 256 or args.size % 16 or args.steps < 1:
        parser.error("Size must be >=256 and divisible by 16; steps must be positive")
    if args.seeds is not None and (len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0):
        parser.error("Seeds must be distinct nonnegative integers")
    if args.varied_subjects and args.seeds is not None and len(args.seeds) != 3:
        parser.error("Varied-subject comparison requires three seeds")
    adapter = Path("table_runs/sd35_diagnostics/hypercube32_compensated_eps3_01100011")
    source = Path("table_runs/main_architectures/sd35/hypercube32/01100011100000010111101001111110")
    config = json.loads((source / "sweep_results.json").read_text())["config"]
    manifest = {"model": config["model"], "adapter": str(adapter),
                "adapter_sha256": digest(adapter / "adapter_model.safetensors"),
                "message": config["wm_message"], "prompt": config["prompt"],
                "seeds": [50000, 50001, 50002], "height": args.size, "width": args.size,
                "steps": args.steps, "guidance_scale": args.guidance,
                "sampling_protocol": "diffusers_sd3", "training": False,
                "selection": "All predefined seeds; no quality-based selection within this batch",
                "caveat": "Qualitative generation comparison; recovery was tested at the original latent resolution"}
    if args.varied_subjects:
        manifest.update(subjects=["Landscape", "Architecture", "Still Life"],
                        seeds=[50010, 50011, 50012],
                        prompts=[
                            "A professional landscape photograph of a turquoise alpine lake beneath snow-covered mountains, pine forest, clear morning light, realistic detail.",
                            "An architectural photograph of a modern white concrete museum with tall glass windows and a reflecting pool, clean lines, soft daylight.",
                            "A studio still life photograph of three ripe peaches and a clear glass vase with white flowers on a blue linen tablecloth, natural window light, fine texture."],
                        caveat="Qualitative generation only; recovery has not been tested at this resolution or under these new prompts")
    if args.seeds is not None:
        manifest["seeds"] = args.seeds
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Existing figure samples have different provenance")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    missing = [(seed, label) for seed in manifest["seeds"] for label in ("clean", "watermarked")
               if not (root / f"{label}_{seed}.png").exists()]
    if missing:
        import psutil
        import torch
        from diffusers import StableDiffusion3Pipeline
        from peft import PeftModel

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to generate SD3.5 samples")
        print("Loading cached SD3.5 and encoding the original prompt", flush=True)
        pipe = StableDiffusion3Pipeline.from_pretrained(
            config["model"], torch_dtype=torch.bfloat16, local_files_only=True,
            low_cpu_mem_usage=True, device_map="balanced",
            max_memory={0: 8 * 1024**3, "cpu": int(psutil.virtual_memory().available * 0.75)})
        with torch.inference_mode():
            encoded = []
            for text in manifest.get("prompts", [config["prompt"]] * len(manifest["seeds"])):
                encoded.append(pipe.encode_prompt(
                    prompt=text, prompt_2=text, prompt_3=text, negative_prompt="",
                    device="cuda", do_classifier_free_guidance=args.guidance > 1))
        pipe.remove_all_hooks()
        pipe.text_encoder = pipe.text_encoder_2 = pipe.text_encoder_3 = None
        pipe.tokenizer = pipe.tokenizer_2 = pipe.tokenizer_3 = None
        pipe.reset_device_map()
        gc.collect()
        torch.cuda.empty_cache()
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, adapter,
                                                   is_trainable=False).to("cuda").eval()
        pipe.vae.to("cuda")
        pipe.vae.enable_tiling()
        for seed, label in missing:
            prompt, negative, pooled, negative_pooled = encoded[manifest["seeds"].index(seed)]
            embeddings = {"prompt_embeds": prompt.cuda(), "pooled_prompt_embeds": pooled.cuda()}
            if args.guidance > 1:
                embeddings.update(negative_prompt_embeds=negative.cuda(),
                                  negative_pooled_prompt_embeds=negative_pooled.cuda())
            print(f"Generating {label}: seed {seed}, {args.size}px, {args.steps} steps", flush=True)
            with torch.inference_mode(), (pipe.transformer.disable_adapter() if label == "clean" else nullcontext()):
                sample = pipe(**embeddings, height=args.size, width=args.size,
                              num_inference_steps=args.steps, guidance_scale=args.guidance,
                              generator=torch.Generator("cuda").manual_seed(seed)).images[0]
            sample.save(root / f"{label}_{seed}.png")
        del pipe
        torch.cuda.empty_cache()
    compose(root, manifest, args.figure_name)
    print(f"Saved paper/figures/{args.figure_name}.pdf and .png", flush=True)


if __name__ == "__main__":
    main()
