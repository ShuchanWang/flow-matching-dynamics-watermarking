"""Monte Carlo check of uniform-random-claim acceptance using saved clean decodes."""

import argparse
import json
import math
from pathlib import Path
import random


def sample_matches(decoded_messages, bits, trials, seed):
    if not decoded_messages or trials < 1 or bits < 1:
        raise ValueError("Positive trial/bit counts and saved clean decodes are required")
    if any(len(m) != bits or set(m) - {"0", "1"} for m in decoded_messages):
        raise ValueError("Saved clean messages do not match the payload")
    rng = random.Random(seed)
    outputs = [int(m, 2) for m in decoded_messages]
    matches = sum(rng.getrandbits(bits) == outputs[i % len(outputs)] for i in range(trials))
    # Wilson interval for the binomial match rate; exact one-sided bound at zero.
    rate = matches / trials
    z = 1.959963984540054
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials**2)) / denominator
    return {"matches": matches, "random_claims": trials,
            "sampled_fpr_pct": 100 * matches / trials,
            "wilson_95_interval_pct": [100 * max(0, center - radius), 100 * min(1, center + radius)],
            "zero_match_one_sided_95_upper_pct": -100 * math.expm1(math.log(0.05) / trials) if matches == 0 else None,
            "expected_fpr_probability": 2.0 ** -bits,
            "expected_fpr_pct": 100 * 2.0 ** -bits,
            "unique_saved_clean_query_trials": len(outputs), "new_velocity_queries": 0}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--trials", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=730042)
    parser.add_argument("--output", type=Path, default=Path("table_runs/random_claim_fpr/results.json"))
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("Trials must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for offset, (scheme, bits) in enumerate((("orthogonal5", 5), ("hypercube5", 5), ("hypercube32", 32))):
        path = Path("table_runs/repairs/mlp") / scheme / "repair_results.json"
        data = json.loads(path.read_text())
        messages = data["per_message"][0]["clean"]["decoded_messages"]
        if any(r["clean"]["decoded_messages"] != messages for r in data["per_message"]):
            raise ValueError("Expected the same clean query trials across target messages")
        results.append({"scheme": scheme, "bits": bits, "source": str(path),
                        "seed": args.seed + offset, **sample_matches(messages, bits, args.trials, args.seed + offset)})
    report = {"definition": "Accept iff decoded clean message equals an independent uniformly sampled full-codebook claim",
              "validation_only": True,
              "scope": "Monte Carlo over claim randomness conditional on saved MLP clean decodes; not new model-query trials",
              "zero_count_caveat": "Zero sampled 32-bit matches does not empirically resolve the probability 2^-32",
              "results": results}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
