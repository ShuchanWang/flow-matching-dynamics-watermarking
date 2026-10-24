"""Cosine-regression removal control for the saved sine-carrier security study."""

import argparse
import copy
import fcntl
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torchvision.datasets import CIFAR10

import run_watermark_security as security


class CosineRemoved(nn.Module):
    def __init__(self, model, direction):
        super().__init__()
        self.model = model
        self.register_buffer("direction", direction)

    def forward(self, x, t, y=None):
        velocity = security.flow.model_velocity(self.model, x, t, y)
        return velocity - torch.cos(2 * math.pi * t)[:, None, None, None] * self.direction


@torch.no_grad()
def estimate_cosine(model, pool, labels, queries, batch, seed):
    generator = torch.Generator(device=pool.device).manual_seed(seed)
    total = torch.zeros(pool.shape[1:], device=pool.device, dtype=torch.float64)
    energy = torch.zeros((), device=pool.device, dtype=torch.float64)
    model.eval()
    for start in range(0, queries, batch):
        x, t, y, _ = security.query_batch(pool, labels, min(batch, queries-start), generator)
        carrier = torch.cos(2 * math.pi * t)
        velocity = security.flow.model_velocity(model, x, t, y)
        total += (carrier[:, None, None, None] * velocity).double().sum(0)
        energy += carrier.double().square().sum()
    return (total / energy).float()


def summarize(output, results):
    security.write_json(output / "results.json", results)
    groups = []
    lines = ["# Mismatched-Carrier Removal", "", f"State: {results['state']}", "",
             "The marked models use sine. The attacker regresses onto cosine and subtracts "
             "that cosine component at strength one, without receiving the owner key, "
             "message or clean-model responses. Owner verification remains unchanged. "
             "This is a fixed wrong-carrier control, not an adaptive unknown-carrier search. "
             "All query seeds, counts, generated-image seeds and FID references match the "
             "original carrier-removal study.", "",
             "| Code | Messages | Exact recovery (%) | Bit accuracy (%) | FID / marked teacher |",
             "|---|---:|---:|---:|---:|"]
    for scheme in sorted({row["scheme"] for row in results["rows"]}):
        selected = [row["metrics"] for row in results["rows"] if row["scheme"] == scheme]
        group = dict(scheme=scheme, messages=len(selected))
        for key in ("recovery_percent", "bit_accuracy_percent", "fid_ratio_teacher"):
            values = [row[key] for row in selected]
            group[key + "_mean"] = float(np.mean(values))
            group[key + "_std"] = float(np.std(values))
        groups.append(group)
        values = [f"{group[key+'_mean']:.3f} +/- {group[key+'_std']:.3f}" for key in
                  ("recovery_percent", "bit_accuracy_percent", "fid_ratio_teacher")]
        lines.append(f"| {scheme} | {len(selected)} | " + " | ".join(values) + " |")
    security.write_json(output / "summary.json", dict(state=results["state"], groups=groups))
    (output / "report.md").write_text("\n".join(lines) + "\n")


def run(source, output, limit=0):
    from huggingface_hub import hf_hub_download
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    original = json.loads((source / "config.json").read_text())
    original_results = json.loads((source / "results.json").read_text())
    if original_results["state"] != "complete":
        raise ValueError("The reference study must be complete")
    args = SimpleNamespace(**original["arguments"])
    plan = original["plan"]
    if limit:
        plan = plan[:limit]
    cfg = security.flow.DEFAULT_CONFIGS["cifar10"]
    checkpoint = Path(hf_hub_download("WayBob/FlowMatching-Unet-Celeb-64x64",
                                     cfg.filename, local_files_only=True))
    config = dict(reference_config_sha256=security.digest(source / "config.json"),
                  reference_quality_sha256=security.digest(source / "quality_reference.pt"),
                  runner_sha256=security.digest(Path(__file__)),
                  security_runner_sha256=security.digest(Path(security.__file__)),
                  base_sha256=security.digest(checkpoint), plan=plan, arguments=vars(args),
                  attacker_carrier="cos(2*pi*t)", subtraction_strength=1,
                  verification="Unmodified original sine decoder", limit=limit)
    if config["base_sha256"] != original["base_sha256"]:
        raise ValueError("Clean checkpoint changed")
    if security.digest(Path(security.flow.__file__)) != original["flow_source_sha256"]:
        raise ValueError("Reference model implementation changed")
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Resume requires identical sources and settings")
    security.write_json(config_path, config)
    cached = torch.load(source / "quality_reference.pt", weights_only=True, map_location="cpu")
    dataset = CIFAR10(str(security.ROOT / "data"), train=True, download=False)
    pool = torch.from_numpy(dataset.data[:20000]).permute(0, 3, 1, 2).to(device).float()/127.5-1
    labels = torch.tensor(dataset.targets[:20000], device=device)
    base = security.flow.build_model(cfg, device)
    security.flow.load_state_dict_flexible(base, str(checkpoint), device)
    base.eval().requires_grad_(False)
    path = output / "results.json"
    results = json.loads(path.read_text()) if path.exists() else dict(rows=[])
    results["state"] = "running"
    try:
        for job in plan:
            c, scheme = job["config"], job["scheme"]
            message = c["wm_message"]
            if any(row["scheme"] == scheme and row["message"] == message for row in results["rows"]):
                continue
            run_dir = Path(job["run_dir"])
            for name, expected in job["hashes"].items():
                if security.digest(run_dir / name) != expected:
                    raise ValueError(f"Saved source changed: {run_dir / name}")
            print(f"START {scheme}/{message}: cosine regression and removal", flush=True)
            teacher = copy.deepcopy(base)
            security.flow.add_lora(teacher, c["lora_rank"], c["lora_alpha"], c["lora_targets"])
            security.flow.load_adapter(teacher, run_dir / "adapter_final.pt", device)
            teacher.eval().requires_grad_(False)
            directory = output / scheme / message
            directory.mkdir(parents=True, exist_ok=True)
            estimate_path = directory / "cosine_estimate.pt"
            if estimate_path.exists():
                direction = torch.load(estimate_path, map_location=device, weights_only=True)
            else:
                direction = estimate_cosine(teacher, pool, labels, args.attack_queries,
                                            args.eval_batch, args.seed + 50000)
                torch.save(direction.cpu(), estimate_path)
            # Owner-only information is loaded after the attack has been constructed.
            attacked = CosineRemoved(teacher, direction)
            key = torch.load(run_dir / "watermark_key.pt", map_location="cpu", weights_only=False)
            marked = json.loads((source / scheme / message / "marked.json").read_text())
            metrics = security.evaluate(attacked, key, c.get("codebook_mode", "auto"),
                                        pool, labels, cached["real_features"], args,
                                        directory / "subtract_cosine", cached["clean_fid"], marked["fid"])
            results["rows"].append(dict(scheme=scheme, message=message, metrics=metrics))
            summarize(output, results)
            del teacher, attacked, direction
            torch.cuda.empty_cache()
        results["state"] = "complete"
    except Exception as exc:
        results.update(state="failed", error=repr(exc))
        raise
    finally:
        summarize(output, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, default=Path("table_runs/watermark_security"))
    parser.add_argument("--output", type=Path, default=Path("table_runs/carrier_mismatch"))
    parser.add_argument("--limit", type=int, default=0, help="Debug only: limit saved teachers")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "study.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.source.resolve(), args.output.resolve(), args.limit)
