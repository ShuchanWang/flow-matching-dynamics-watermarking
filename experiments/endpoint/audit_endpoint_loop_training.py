"""Independent-query decoding audit for saved standalone loop-training models."""

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torchvision.datasets import CIFAR10

import run_baseline_first as baseline
import run_watermark_security as security


def run(args):
    from huggingface_hub import hf_hub_download
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    cfg = security.flow.DEFAULT_CONFIGS["cifar10"]
    checkpoint = Path(hf_hub_download("WayBob/FlowMatching-Unet-Celeb-64x64", cfg.filename, local_files_only=True))
    checkpoint_hash = security.digest(checkpoint)
    base = security.flow.build_model(cfg, device)
    security.flow.load_state_dict_flexible(base, str(checkpoint), device)
    base.eval().requires_grad_(False)
    dataset = CIFAR10(str(security.ROOT / "data"), train=False, download=False)
    pool = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).to(device).float() / 127.5 - 1
    labels = torch.tensor(dataset.targets, device=device)
    settings = SimpleNamespace(eval_seed=args.seed, trials=args.trials, queries=args.queries,
                               eval_batch=args.batch)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(arguments={k: [str(p) for p in v] if isinstance(v, list) else str(v)
                               if isinstance(v, Path) else v for k, v in vars(args).items()},
                    checkpoint_sha256=checkpoint_hash,
                    source_hashes={p.name: security.digest(p) for p in [Path(__file__), Path(baseline.__file__),
                                                                      Path(security.__file__), Path(security.flow.__file__)]},
                    decoder="Existing sine-demodulation decoder, independent CIFAR test interpolants, no clean subtraction")
    security.write_json(args.output / "config.json", manifest)
    results = dict(state="running", rows=[])
    security.write_json(args.output / "results.json", results)
    for directory in args.runs:
        config_path = directory / "config.json"
        config = json.loads(config_path.read_text())
        if config["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Teacher checkpoint mismatch")
        key_path = Path(config["arguments"]["key"])
        if security.digest(key_path) != config["key_sha256"]:
            raise ValueError("Saved key changed")
        key = torch.load(key_path, map_location=device, weights_only=True)
        for arm in ("additive", "loop"):
            model_path = directory / arm / f"model_step{args.step}.pt"
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            model = copy.deepcopy(base)
            if config["trainable"] == "lora":
                security.flow.add_lora(model, 16, 16, "both")
            security.flow.load_adapter(model, model_path, device)
            model.eval().requires_grad_(False)
            with torch.no_grad():
                detection = baseline.verify(model, "dynamics", key, pool, labels, settings)
            results["rows"].append(dict(run=str(directory), arm=arm, step=args.step,
                                        model_sha256=security.digest(model_path),
                                        training_config_sha256=security.digest(config_path), detection=detection))
            security.write_json(args.output / "results.json", results)
            print(f"RESULT {directory.name}/{arm}: exact={detection['recovery_percent']:.1f}%, "
                  f"bits={detection['bit_accuracy_percent']:.3f}%, {args.trials} x {args.queries} queries", flush=True)
            del model
    results["state"] = "complete"
    security.write_json(args.output / "results.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", default=[Path("table_runs/endpoint_loop_training"),
                                                               Path("table_runs/endpoint_loop_training_balanced_full")])
    parser.add_argument("--output", type=Path, default=Path("table_runs/endpoint_loop_detection_audit"))
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=94731)
    run(parser.parse_args())
