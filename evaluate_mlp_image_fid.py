"""Measure final MNIST MLP checkpoints with the image study's Inception FID."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch
from torchvision.datasets import MNIST

from repair_mlp_watermarks import FlowModel, samples
from watermark_hf_flow_unet import compute_fid_from_features, inception_features


ROOT = Path("table_runs/mlp_inception_quality")


def save(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--samples", type=int, default=500)
    args = parser.parse_args()
    if args.samples < 2 or args.samples > 10000:
        parser.error("Expected 2--10000 MNIST test samples")
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "study.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = torch.device("cuda")
        status = {"state": "running", "pid": os.getpid(), "samples": args.samples,
                  "feature_extractor": "torchvision_inception_v3_imagenet_prelogits",
                  "reference": "MNIST test images", "sampling_steps": 200,
                  "training": False, "per_setting": []}
        save(ROOT / "summary.json", status)
        try:
            images = MNIST("data", train=False, download=False).data[:args.samples, None].float() / 127.5 - 1
            reference = inception_features(images, device)
            for scheme in ("orthogonal5", "hypercube5", "hypercube32"):
                directory = Path("table_runs/repairs/mlp") / scheme
                report = json.loads((directory / "repair_results.json").read_text())
                clean_path = Path(report["config"]["run_dir"]) / "clean_model_0.pt"
                seed_value = report["config"]["seed"] + 3
                keys = report["per_message"]
                if len(keys) != 5:
                    raise ValueError("Expected five final marked checkpoints")
                manifest = {"scheme": scheme, "samples": args.samples, "seed": seed_value,
                            "source": str(directory), "clean_sha256": digest(clean_path),
                            "marked_sha256": {r["message"]: digest(directory / f"wm_model_{r['message']}_0.pt") for r in keys}}
                output = ROOT / f"{scheme}.json"
                if output.exists():
                    result = json.loads(output.read_text())
                    if result["manifest"] != manifest:
                        raise ValueError("Existing quality result has a different source/protocol")
                else:
                    result = {"manifest": manifest, "per_message": []}
                if "clean_fid" not in result:
                    clean = FlowModel(torch.load(clean_path, map_location="cpu", weights_only=True)).to(device).eval()
                    generated = torch.from_numpy(samples(clean, device, seed_value, args.samples)).reshape(-1, 1, 28, 28) * 2 - 1
                    clean_features = inception_features(generated, device)
                    result["clean_fid"] = compute_fid_from_features(reference, clean_features)
                    del clean, generated, clean_features
                    save(output, result)
                complete = {r["message"] for r in result["per_message"]}
                for row in keys:
                    message = row["message"]
                    if message in complete:
                        continue
                    path = directory / f"wm_model_{message}_0.pt"
                    model = FlowModel(torch.load(path, map_location="cpu", weights_only=True)).to(device).eval()
                    generated = torch.from_numpy(samples(model, device, seed_value, args.samples)).reshape(-1, 1, 28, 28) * 2 - 1
                    features = inception_features(generated, device)
                    fid = compute_fid_from_features(reference, features)
                    result["per_message"].append({"message": message, "fid": fid,
                                                  "fid_ratio": fid / result["clean_fid"]})
                    save(output, result)
                    print(f"{scheme} {message}: FID={fid:.6f}, ratio={fid / result['clean_fid']:.6f}", flush=True)
                    del model, generated, features
                    torch.cuda.empty_cache()
                ratios = [r["fid_ratio"] for r in result["per_message"]]
                result.update(complete=True, ratio_mean=statistics.mean(ratios), ratio_std=statistics.pstdev(ratios))
                save(output, result)
                status["per_setting"].append({"scheme": scheme, "ratio_mean": result["ratio_mean"],
                                              "ratio_std": result["ratio_std"], "source": str(output)})
                save(ROOT / "summary.json", status)
            status["state"] = "complete"
        except Exception as exc:
            status.update(state="failed", error=str(exc))
            raise
        finally:
            save(ROOT / "summary.json", status)


if __name__ == "__main__":
    main()
