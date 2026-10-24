"""Complete uniform-claim FPR measurements using actual clean-model decodes."""
from repo_layout import ROOT, source_path

import gc
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

from evaluate_fresh_mlp_fpr import binomial_summary
from refresh_paper_performance import groups

ROOT = Path("table_runs/paper_fpr")
SEED = 202610150
CLAIMS = 1000000


def cohort_id(name, dataset, code):
    return "_".join((name, dataset, code)).replace(" ", "_").replace(".", "")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def publish(identity, source, records, new_queries):
    name, dataset, code = identity
    bits = 32 if code == "H32" else 5
    messages = [r["decoded"] for r in records]
    if len(messages) != 20 or any(len(m) != bits or set(m) - {"0", "1"} for m in messages):
        raise ValueError("Expected 20 complete clean-model decoded messages")
    claim_seed = SEED + int(hashlib.sha256(cohort_id(*identity).encode()).hexdigest()[:8], 16)
    rng = random.Random(claim_seed)
    outputs = [int(m, 2) for m in messages]
    hits = sum(rng.getrandbits(bits) == outputs[i % 20] for i in range(CLAIMS))
    report = {"generator": name, "dataset": dataset, "code": code, "source": source,
              "claim_seed": claim_seed, "claim_distribution": "independent uniform full-codebook messages",
              "clean_query_trials": 20, "new_velocity_queries": new_queries,
              "scope": "Claim sampling conditional on recorded clean decodes under one saved key; not one million model-query trials",
              "records": records, "metric": binomial_summary(hits, CLAIMS), "complete": True}
    save(ROOT / f"{cohort_id(*identity)}.json", report)
    print(f"Saved FPR: {name}/{dataset}/{code}: {hits}/{CLAIMS}", flush=True)


def archived(identity, source):
    path = Path(source)
    data = json.loads(path.read_text())
    if "clean" in data and "decoded_messages" in data["clean"]:
        return [{"trial": i, "query_seed": data["seed"] + i, "decoded": m}
                for i, m in enumerate(data["clean"]["decoded_messages"])], str(path)
    config = data["config"]
    folder = {"Audio MLP": "mlp", "Audio Transformer": "transformer", "Stable Audio": "stable"}[identity[0]]
    audit = Path("table_runs/audio_ownership") / folder / f"{config['wm_message']}_{config['codebook_mode']}.json"
    records = json.loads(audit.read_text())["trials"]
    return [{"trial": i, "query_seed": r["seed"], "decoded": "".join(map(str, r["clean"]["decoded"]))}
            for i, r in enumerate(records)], str(audit)


def unet(entries):
    import torch
    import watermark_hf_flow_unet as hf
    from huggingface_hub import hf_hub_download
    from watermark_codebooks import decode_signature

    device = torch.device("cuda")
    config = json.loads(Path(entries[0][1]).read_text())["config"]
    args = SimpleNamespace(**config)
    args.n_fid_samples, args.num_workers = 0, 0
    cfg = hf.DEFAULT_CONFIGS[args.dataset]
    checkpoint = hf_hub_download(args.repo_id, args.checkpoint or cfg.filename, local_files_only=True)
    model = hf.build_model(cfg, device).eval().requires_grad_(False)
    hf.load_state_dict_flexible(model, checkpoint, device)
    if args.data_source == "base_samples":
        if not Path(args.base_sample_pool).exists():
            raise ValueError("Expected original cached sample pool")
        loader = hf.build_base_sample_loader(model, cfg, args, device)
    else:
        loader, _ = hf.load_real_data(cfg, args, device)
    keys = [torch.load(Path(source).parent / "watermark_key.pt", map_location=device, weights_only=False)
            for _, source in entries]
    configs = [json.loads(Path(source).read_text())["config"] for _, source in entries]
    rows = [[] for _ in entries]
    with torch.inference_mode():
        for trial in range(20):
            hf.seed_everything(SEED + trial)
            total = torch.zeros(math.prod(cfg.dim), device=device)
            for start in range(0, args.n_queries, args.batch_size):
                size = min(args.batch_size, args.n_queries - start)
                x, t, y = hf.sample_detection_batch(model, cfg, args, device, size, loader)
                value = hf.model_velocity(model, x, t, y).flatten(1).float()
                total += (torch.sin(2 * math.pi * t)[:, None] * value).sum(0)
            total /= args.n_queries
            if not torch.isfinite(total).all():
                raise ValueError("Nonfinite clean UNet signature")
            for row, key, config in zip(rows, keys, configs):
                decoded, _ = decode_signature(total @ key["P"], key["codes"], len(config["wm_message"]),
                                              config.get("codebook_mode", "auto"))
                row.append({"trial": trial, "query_seed": SEED + trial, "decoded": "".join(map(str, decoded))})
            print(f"UNet {args.dataset}: clean queries {trial + 1}/20", flush=True)
    for (identity, source), records in zip(entries, rows):
        publish(identity, source, records, 20 * args.n_queries)
    del model, keys, loader
    gc.collect()
    torch.cuda.empty_cache()


def sd(entries):
    import torch
    import psutil
    from diffusers import StableDiffusion3Pipeline
    from watermark_codebooks import decode_signature

    root = Path(entries[0][1]).parent
    configs = [json.loads(Path(source).read_text())["config"] for _, source in entries]
    config = configs[0]
    keys = [torch.load(Path(source).parent / "codebook.pt", map_location="cuda", weights_only=False)
            for _, source in entries]
    pool = torch.load(root / f"latent_pool_{config['latent_size']}_{config['n_train_latents']}.pt",
                      map_location="cpu", weights_only=False)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        config["model"], torch_dtype=torch.bfloat16, local_files_only=True, device_map="balanced",
        max_memory={0: 12 * 1024**3, "cpu": int(psutil.virtual_memory().available * .75)})
    with torch.inference_mode():
        prompt, _, pooled, _ = pipe.encode_prompt(prompt=config["prompt"],
                                                prompt_2=config["prompt"], prompt_3=config["prompt"], device="cuda",
                                                do_classifier_free_guidance=False)
    model = pipe.transformer
    pipe.remove_all_hooks()
    pipe.text_encoder = pipe.text_encoder_2 = pipe.text_encoder_3 = None
    pipe.reset_device_map()
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    model.to("cuda").eval().requires_grad_(False)
    rows = [[] for _ in entries]
    queries = config["n_detect_queries"]
    with torch.inference_mode():
        for trial in range(20):
            generator = torch.Generator("cuda").manual_seed(SEED + trial)
            rng = random.Random(SEED + trial)
            total = torch.zeros(math.prod(pool.shape[1:]), device="cuda")
            for start in range(0, queries, 16):
                size = min(16, queries - start)
                index = torch.randint(len(pool), (size,), device="cuda", generator=generator).cpu()
                x1 = pool[index].cuda().to(torch.bfloat16)
                x0 = torch.randn(x1.shape, device="cuda", dtype=torch.bfloat16, generator=generator)
                t = torch.tensor([rng.random() for _ in range(size)], device="cuda", dtype=torch.bfloat16)
                x = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
                v = model(hidden_states=x, timestep=t * 1000,
                          encoder_hidden_states=prompt.expand(size, -1, -1),
                          pooled_projections=pooled.expand(size, -1), return_dict=False)[0].float()
                total += (torch.sin(2 * math.pi * t.float())[:, None] * v.flatten(1)).sum(0)
            total /= queries
            if not torch.isfinite(total).all():
                raise ValueError("Nonfinite clean SD signature")
            for row, key, config in zip(rows, keys, configs):
                decoded, _ = decode_signature(total @ key["P"], key["codes"], len(config["wm_message"]),
                                              config.get("codebook_mode", "auto"))
                row.append({"trial": trial, "query_seed": SEED + trial, "decoded": "".join(map(str, decoded))})
            print(f"SD3.5: clean queries {trial + 1}/20", flush=True)
    for (identity, source), records in zip(entries, rows):
        publish(identity, source, records, 20 * queries)
    del model, keys
    gc.collect()
    torch.cuda.empty_cache()


def main():
    import torch
    torch.set_num_threads(4)
    entries = [((name, dataset, code), rows[0]["source"]) for name, dataset, code, rows in groups()
               if name != "Image MLP" and not (ROOT / f"{cohort_id(name, dataset, code)}.json").exists()]
    for identity, source in entries:
        if identity[0] in ("Audio MLP", "Audio Transformer", "Stable Audio") or identity[0] == "SD3.5 DiT" and identity[2] == "H32":
            records, archive = archived(identity, source)
            publish(identity, archive, records, 0)
    for dataset in ("MNIST", "CIFAR-10", "CelebA-64"):
        selected = [(identity, source) for identity, source in entries if identity[:2] == ("RF-UNet", dataset)]
        if selected:
            unet(selected)
    selected = [(identity, source) for identity, source in entries if identity[0] == "SD3.5 DiT" and identity[2] != "H32"]
    if selected:
        sd(selected)
    subprocess.run([sys.executable, str(source_path('refresh_paper_performance.py')), "--require-complete"], check=True)
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd="paper", check=True)


if __name__ == "__main__":
    main()
