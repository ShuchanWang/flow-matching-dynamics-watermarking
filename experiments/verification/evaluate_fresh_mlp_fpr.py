"""Fresh clean-model query trials for random-claim and fixed-target acceptance."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time


def binomial_summary(matches, trials):
    if trials < 1 or not 0 <= matches <= trials:
        raise ValueError("Require positive trial count and 0 <= matches <= trials")
    rate = matches / trials
    z = 1.959963984540054
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials**2)) / denominator
    return {"matches": matches, "trials": trials, "fpr_pct": 100 * rate,
            "wilson_95_interval_pct": [100 * max(0, center - radius), 100 * min(1, center + radius)],
            "zero_match_one_sided_95_upper_pct": -100 * math.expm1(math.log(.05) / trials) if matches == 0 else None}


def weighted_velocity_mean(model, queries, generator, device, batch):
    import torch

    total = torch.zeros(784, device=device)
    for offset in range(0, queries, batch):
        size = min(batch, queries - offset)
        x = torch.randn(size, 784, generator=generator, device=device)
        t = torch.rand(size, 1, generator=generator, device=device)
        total += (torch.sin(2 * math.pi * t) * model(x, t)).sum(0)
    # Projection is linear, so one velocity mean can serve all tested keys.
    return total / queries


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=202610140)
    parser.add_argument("--output", type=Path, default=Path("table_runs/fresh_mlp_fpr"))
    args = parser.parse_args()
    if min(args.trials, args.queries, args.batch) < 1:
        parser.error("Budgets must be positive")
    import torch
    from repair_mlp_watermarks import FlowModel
    from watermark_codebooks import decode_signature, make_key

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schemes = (("orthogonal5", 5, "orthogonal"), ("hypercube5", 5, "hypercube"),
               ("hypercube32", 32, "hypercube"))
    sources = [Path("table_runs/main_architectures/mlp") / scheme for scheme, _, _ in schemes]
    hashes = [hashlib.sha256((p / "clean_model_0.pt").read_bytes()).hexdigest() for p in sources]
    if len(set(hashes)) != 1:
        raise ValueError("Sharing clean queries requires identical clean checkpoints")
    keys = [torch.load(p / "watermark_key.pt", map_location=device, weights_only=True) for p in sources]
    model = FlowModel(torch.load(sources[0] / "clean_model_0.pt", map_location=device,
                                 weights_only=True)).to(device).eval().requires_grad_(False)
    config = {"trials": args.trials, "queries_per_trial": args.queries, "batch": args.batch,
              "seed": args.seed, "clean_sha256": hashes[0],
              "original_key_sha256": [hashlib.sha256((p / "watermark_key.pt").read_bytes()).hexdigest() for p in sources],
              "distribution": "pure_noise_original_mlp_protocol", "model": "MNIST image MLP",
              "training": False, "shared_clean_queries": True,
              "random_decision": "Exact equality to an independent full-codebook claim under a fresh random key",
              "fixed_decision": "Exact equality to each original predeclared target under its original key"}
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "results.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"config": config, "records": []}
    if report["config"] != config:
        previous = dict(report["config"])
        previous_trials = previous.pop("trials")
        current = dict(config)
        current.pop("trials")
        if previous != current or args.trials < previous_trials:
            raise ValueError("Existing fresh-query results have a different protocol")
        report["config"] = config
        report.pop("complete", None)
        report.pop("summary", None)
        write(report_path, report)
    start = time.monotonic()
    with torch.inference_mode():
        for trial in range(len(report["records"]), args.trials):
            query_seed = args.seed + trial
            generator = torch.Generator(device=device).manual_seed(query_seed)
            total = weighted_velocity_mean(model, args.queries, generator, device, args.batch)
            row = {"trial": trial, "query_seed": query_seed, "results": []}
            for index, ((scheme, bits, mode), original) in enumerate(zip(schemes, keys)):
                key_seed = args.seed + 100000 + 3 * trial + index
                claim_seed = args.seed + 200000 + 3 * trial + index
                claim = format(random.Random(claim_seed).getrandbits(bits), f"0{bits}b")
                with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == "cuda" else []):
                    torch.manual_seed(key_seed)
                    projection, codes = make_key(784, bits, original["P"].shape[1], device, mode)
                decoded, _ = decode_signature(total @ projection, codes, bits, mode)
                fixed, _ = decode_signature(total @ original["P"], original["codes"], bits, mode)
                decoded, fixed = "".join(map(str, decoded)), "".join(map(str, fixed))
                row["results"].append({"scheme": scheme, "key_seed": key_seed,
                                       "claim_seed": claim_seed, "claim": claim,
                                       "decoded": decoded, "match": decoded == claim,
                                       "original_key_decoded": fixed})
            report["records"].append(row)
            if (trial + 1) % 50 == 0 or trial + 1 == args.trials:
                write(report_path, report)
                print(f"Fresh clean query trials: {trial + 1}/{args.trials}; elapsed {time.monotonic() - start:.1f}s", flush=True)
    report["summary"] = []
    for index, ((scheme, bits, _), key) in enumerate(zip(schemes, keys)):
        values = [r["results"][index] for r in report["records"]]
        fixed = {message: binomial_summary(sum(r["original_key_decoded"] == message for r in values), args.trials)
                 for message in key["messages"]}
        report["summary"].append({"scheme": scheme, "bits": bits,
                                  "random_key_random_claim": binomial_summary(sum(r["match"] for r in values), args.trials),
                                  "original_key_fixed_targets": fixed})
    report.update(complete=True, new_velocity_queries=args.trials * args.queries,
                  caveat="One clean MLP checkpoint, not population FPR across architectures; randomized claims do not validate fixed-target ownership")
    write(report_path, report)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
