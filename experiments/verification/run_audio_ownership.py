"""Calibrate ownership gates for final audio checkpoints, without training."""

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from watermark_audio_flow import AudioFlow, StableVelocity, detect, load_adapter_checkpoint, seed, stable_conditioning, training_batch
from watermark_codebooks import decode_signature, make_key, target_score


ROOT = Path("table_runs/audio_ownership")
PROTOCOL = {"version": 1, "queries": 4096, "calibration_trials": 40,
            "test_trials": 20, "calibration_seed": 800042, "test_seed": 900042,
            "calibration_split": "train", "test_split": "test",
            "gate": "exact target AND score > max(clean calibration scores) + 2*population_std"}


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def final_sources(architecture):
    if architecture != "stable":
        return sorted(Path(f"table_runs/main_architectures/audio/{architecture}").glob("*/fma/*/results.json"))
    paths = list(Path("table_runs/stable_audio_validation/orthogonal5").glob("*/*/results.json"))
    paths += list(Path("table_runs/stable_audio_margin_pilot/hypercube32").glob("*/*/results.json"))
    paths += list(Path("table_runs/stable_audio_margin_suite/hypercube32").glob("*/*/results.json"))
    return sorted(paths)


def summarize(calibration, records):
    scores = np.asarray(calibration, dtype=float)
    if scores.shape != (40,) or not np.isfinite(scores).all() or len(records) != 20:
        raise ValueError("Expected 40 finite calibration scores and 20 test records")
    threshold = float(scores.max() + 2 * scores.std())
    def accepts(record):
        return bool(record["exact"] and record["target_score"] > threshold)
    result = {"threshold": threshold}
    for label, metric in (("clean", "ownership_fpr_pct"), ("wrong_key", "wrong_key_acceptance_pct"),
                          ("wm", "ownership_tpr_pct")):
        result[metric] = 100 * sum(accepts(r[label]) for r in records) / len(records)
    result["ownership_fnr_pct"] = 100 - result["ownership_tpr_pct"]
    result["recovery_failure_pct"] = 100 * sum(not r["wm"]["exact"] for r in records) / len(records)
    result["ber_pct"] = 100 * sum(r["wm"]["bit_errors"] for r in records) / sum(len(r["wm"]["decoded"]) for r in records)
    positives = np.asarray([r["wm"]["target_score"] for r in records])[:, None]
    for label in ("clean", "wrong_key"):
        negatives = np.asarray([r[label]["target_score"] for r in records])[None, :]
        result[f"score_auc_vs_{label}"] = float(((positives > negatives) + 0.5 * (positives == negatives)).mean())
    return result


def record(decoded, values, config):
    truth = tuple(map(int, config["wm_message"]))
    return {"decoded": list(decoded), "exact": decoded == truth,
            "bit_errors": sum(a != b for a, b in zip(decoded, truth)),
            "target_score": target_score(values, config["wm_message"], config["codebook_mode"])}


def load_clean(config, data, device):
    pool = data["train"].float()
    if config["architecture"] == "stable":
        from stable_audio_tools import get_pretrained_model
        codec, _ = get_pretrained_model(data["config"]["stable_model"])
        if codec.diffusion_objective != "rectified_flow":
            raise ValueError("Expected RF Stable Audio")
        codec.to(device).eval().requires_grad_(False)
        with torch.no_grad():
            inputs = stable_conditioning(codec, pool.shape[1:], config["prompt"], data["config"]["seconds"], device)
        codec.conditioner.to("cpu")
        for conditioner in codec.conditioner.conditioners.values():
            encoder = getattr(conditioner, "model", None)
            if isinstance(encoder, torch.nn.Module):
                encoder.to("cpu")
        codec.pretransform.to("cpu")
        return StableVelocity(codec.model, inputs).eval(), pool, data["test"].float()
    cache_config = {k: config[k] for k in ("architecture", "width", "depth", "base_steps", "lr", "batch_size", "seed")}
    cache_config["latent_sha256"] = hashlib.sha256(Path(config["latents"]).read_bytes()).hexdigest()
    cache_id = hashlib.sha256(json.dumps(cache_config, sort_keys=True).encode()).hexdigest()[:16]
    path = Path(config["base_dir"]) / f"{config['dataset']}_{config['architecture']}_{cache_id}.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["config"] != cache_config:
        raise ValueError("Clean checkpoint mismatch")
    model = AudioFlow(pool.shape[1:], config["architecture"], config["width"], config["depth"]).to(device)
    model.load_state_dict(saved["model"], strict=True)
    mean, std = pool.mean((0, 2), keepdim=True), pool.std((0, 2), keepdim=True).clamp_min(1e-5)
    return model.eval().requires_grad_(False), (pool - mean) / std, (data["test"].float() - mean) / std


@torch.no_grad()
def run_group(architecture, status):
    sources = final_sources(architecture)
    if len(sources) != 10:
        raise ValueError(f"Expected ten final runs for {architecture}, got {len(sources)}")
    entries = []
    for path in sources:
        config = json.loads(path.read_text())["config"]
        key = torch.load(path.parent / "watermark_key.pt", map_location="cpu", weights_only=False)
        entries.append((path, config, key))
    folder = ROOT / architecture
    folder.mkdir(exist_ok=True)
    manifest = {"protocol": PROTOCOL, "sources": [str(p) for p in sources],
                "key_hashes": [hashlib.sha256((p.parent / "watermark_key.pt").read_bytes()).hexdigest() for p in sources]}
    manifest_path = folder / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing evaluation has a different protocol or source keys")
    write_json(manifest_path, manifest)
    if all((folder / f"{c['wm_message']}_{c['codebook_mode']}.json").exists() for _, c, _ in entries):
        return
    device = torch.device("cuda")
    data = torch.load(entries[0][1]["latents"], map_location="cpu", weights_only=False)
    digest = hashlib.sha256(Path(entries[0][1]["latents"]).read_bytes()).hexdigest()
    if set(data["sources"]["train"]) & set(data["sources"]["test"]):
        raise ValueError("Train/test recording overlap")
    for _, config, key in entries:
        if key["latent_sha256"] != digest or config["latents"] != entries[0][1]["latents"]:
            raise ValueError("Inconsistent latent pool")
    base, train_pool, test_pool = load_clean(entries[0][1], data, device)
    # Share clean forward passes across all original projections, without sharing thresholds.
    projections = torch.cat([key["P"] for _, _, key in entries], dim=1).to(device)
    calibration_path = folder / "calibration.json"
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else []
    batch = 4 if architecture == "stable" else 128
    for trial in range(len(calibration), 40):
        seed(PROTOCOL["calibration_seed"] + trial)
        signature = torch.zeros(projections.shape[1], device=device)
        for start in range(0, 4096, batch):
            x, t, _ = training_batch(train_pool, min(batch, 4096 - start), device)
            signature += (torch.sin(2 * math.pi * t)[:, None] * (base(x, t).flatten(1) @ projections)).sum(0)
        chunks = (signature / 4096).split([key["P"].shape[1] for _, _, key in entries])
        calibration.append([target_score(value @ key["codes"].to(device).T, config["wm_message"], config["codebook_mode"])
                            for value, (_, config, key) in zip(chunks, entries)])
        write_json(calibration_path, calibration)
        status.update(stage="calibration", architecture=architecture, completed_trials=trial + 1)
        write_json(ROOT / "summary.json", status)
        print(f"{architecture}: calibration {trial + 1}/40", flush=True)
    thresholds = [float(np.max(np.asarray(calibration)[:, i]) + 2 * np.std(np.asarray(calibration)[:, i])) for i in range(10)]
    write_json(folder / "frozen_thresholds.json", {"protocol": PROTOCOL, "thresholds": thresholds})
    for index, (path, config, key) in enumerate(entries):
        destination = folder / f"{config['wm_message']}_{config['codebook_mode']}.json"
        if destination.exists():
            continue
        saved_records = path.parent / "recovery_progress.json"
        progress = json.loads(saved_records.read_text()) if saved_records.exists() else {}
        reused = architecture == "stable" and len(progress.get("trials", [])) == 20
        if reused:
            records = progress["trials"]
            if progress["completed_trials"] != 20 or config["n_queries"] != 4096:
                raise ValueError("Incomplete saved evaluation")
            for row in records:
                if PROTOCOL["calibration_seed"] <= row["seed"] < PROTOCOL["calibration_seed"] + 40:
                    raise ValueError("Calibration/test seed overlap")
        else:
            if architecture == "stable":
                from peft import LoraConfig, inject_adapter_in_model
                model = copy.deepcopy(base)
                inject_adapter_in_model(LoraConfig(r=16, lora_alpha=16, target_modules="all-linear"), model.backbone)
                adapter = config.get("evaluate_adapter") or str(path.parent / f"adapter_step_{config['embedding_updates']}.pt")
                load_adapter_checkpoint(model, adapter)
            else:
                model = AudioFlow(train_pool.shape[1:], architecture, config["width"], config["depth"]).to(device)
                model.load_state_dict(torch.load(path.parent / "model_final.pt", map_location=device, weights_only=False), strict=True)
            model.eval().requires_grad_(False)
            args = SimpleNamespace(**config)
            args.eval_batch_size = batch
            P, codes = key["P"].to(device), key["codes"].to(device)
            seed(config["seed"] + 300)
            wrong_P, _ = make_key(math.prod(train_pool.shape[1:]), len(config["wm_message"]), config["wm_K"], device, config["codebook_mode"])
            progress_path = destination.with_suffix(".progress.json")
            records = json.loads(progress_path.read_text()) if progress_path.exists() else []
            for trial in range(len(records), 20):
                row = {"trial": trial, "seed": PROTOCOL["test_seed"] + trial}
                for label, network, projection in (("wm", model, P), ("clean", base, P), ("wrong_key", model, wrong_P)):
                    seed(row["seed"])
                    decoded, values = detect(network, test_pool, projection, codes, args, device)
                    row[label] = record(decoded, values, config)
                records.append(row)
                write_json(progress_path, records)
                print(f"{architecture} {config['wm_message']}: test {trial + 1}/20", flush=True)
            del model
        scores = np.asarray(calibration)[:, index].tolist()
        result = {"source": str(path), "protocol": PROTOCOL, "architecture": architecture,
                  "message": config["wm_message"], "codebook_mode": config["codebook_mode"],
                  "reused_saved_test_trials": reused, "calibration_scores": scores,
                  "trials": records, "metrics": summarize(scores, records),
                  "caveat": "Empirical repeated-query rates for one clean counterpart and one wrong key per message; not a population guarantee. Saved Stable Audio trials evaluated selected final checkpoints."}
        write_json(destination, result)
        print(json.dumps({"message": config["wm_message"], **result["metrics"]}), flush=True)
    del base
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--architectures", nargs="+", choices=["mlp", "transformer", "stable"], default=["mlp", "transformer", "stable"])
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "study.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {"state": "running", "pid": os.getpid(), "protocol": PROTOCOL, "architectures": args.architectures}
        write_json(ROOT / "summary.json", status)
        try:
            for architecture in args.architectures:
                run_group(architecture, status)
            status.update(state="complete")
        except Exception as exc:
            status.update(state="failed", error=str(exc))
            raise
        finally:
            status["results"] = [json.loads(path.read_text()) | {"trials": None, "calibration_scores": None}
                                 for architecture in args.architectures for path in sorted((ROOT / architecture).glob("*_*.json"))
                                 if not path.name.endswith(".progress.json") and path.name != "frozen_thresholds.json"]
            write_json(ROOT / "summary.json", status)


if __name__ == "__main__":
    main()
