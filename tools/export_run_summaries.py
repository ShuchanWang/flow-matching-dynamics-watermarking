"""Export compact completed-study evidence without changing full local artifacts."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STUDIES = {
    "summary_stage1": "core", "summary_classic": "core", "summary_main_audio": "core",
    "main_architectures": "core", "audio_ownership": "core", "stable_audio_validation": "core",
    "payload_extension": "payload", "consistent_image_quality": "quality", "mlp_inception_quality": "quality",
    "paper_fpr": "verification", "fresh_mlp_fpr": "verification", "random_claim_fpr": "verification",
    "watermark_security": "security", "carrier_mismatch": "security",
    "embedding_strategy_robustness": "robustness",
    "secret_carrier_pilot": "pilots", "tail_gate_pilot": "pilots",
    "baseline_first/tune": "baselines", "baseline_first/confirm": "baselines",
    "baseline_screen32": "baselines",
    "endpoint_loop_pilot": "pilots",
    "endpoint_loop_training": "pilots",
    "endpoint_loop_training_balanced_full": "pilots",
    "endpoint_loop_detection_audit": "pilots",
    "endpoint_loop_distillation": "pilots",
}
FILES = {"config.json", "results.json", "summary.json", "aggregate.json", "per_run.json",
         "status.json", "manifest.json", "analysis.json", "report.md", "summary.csv", "aggregate.csv"}
DROP = {"scores", "normalized_scores", "decoded", "projection", "weight_projection", "trigger_noise",
        "coordinates", "signed_coordinates", "maximum_query_projected_mean", "maximum_query_projected_second_moment",
        "claim", "original_key_decoded",
        "private_seed", "access_token", "token", "password", "authorization", "api_key", "secret"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def clean_string(value, root):
    value = value.replace(str(root) + "/", "")
    if value.startswith("/"):
        return "<external>/" + Path(value).name
    return value


def compact(value, root):
    if isinstance(value, dict):
        return {clean_string(k, root): compact(v, root) for k, v in value.items() if k.lower() not in DROP}
    if isinstance(value, list):
        return [compact(v, root) for v in value]
    if isinstance(value, str):
        return clean_string(value, root)
    return value


def completed(directory):
    for name in ("results.json", "status.json", "summary.json"):
        path = directory / name
        if path.exists():
            value = json.loads(path.read_text())
            if isinstance(value, dict):
                state = value.get("state", value.get("status"))
                if isinstance(state, str):
                    return state in ("complete", "completed", "done"), state
    return True, "unspecified"


def export(root, output):
    entries, skipped = [], []
    for study, category in STUDIES.items():
        directory = root / "table_runs" / study
        if not directory.is_dir():
            continue
        ready, state = completed(directory)
        if not ready:
            skipped.append(dict(study=study, state=state))
            continue
        candidates = sorted(p for p in directory.iterdir() if p.is_file() and p.name in FILES)
        if study == "paper_fpr":
            candidates = sorted(directory.glob("*.json"))
        for source in candidates:
            raw = source.read_bytes()
            artifacts = []
            if source.suffix == ".json":
                value = compact(json.loads(raw), root)
                if study == "fresh_mlp_fpr" and source.name == "results.json" and "records" in value:
                    records = value.pop("records")
                    stream = io.StringIO()
                    writer = csv.DictWriter(stream, fieldnames=["trial", "query_seed", "scheme", "key_seed", "claim_seed", "match"], lineterminator="\n")
                    writer.writeheader()
                    count = 0
                    for record in records:
                        for outcome in record["results"]:
                            writer.writerow(dict(trial=record["trial"], query_seed=record["query_seed"], **outcome))
                            count += 1
                    artifacts.append(("trial_outcomes.csv", stream.getvalue().encode()))
                    value.update(trial_outcomes_file="trial_outcomes.csv", trial_outcome_count=count)
                data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
            elif source.suffix == ".csv":
                stream = io.StringIO()
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerows(csv.reader(io.StringIO(raw.decode().replace(str(root) + "/", ""))))
                data = stream.getvalue().encode()
            else:
                data = raw.decode().replace(str(root) + "/", "").encode()
            artifacts.append((source.name, data))
            for name, data in artifacts:
                destination = output / category / study / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                temporary.write_bytes(data)
                temporary.replace(destination)
                entries.append(dict(source=str(source.relative_to(root)), snapshot=str(destination.relative_to(output)),
                                    source_sha256=digest(raw), snapshot_sha256=digest(data), bytes=len(data), state=state))
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(files=entries, skipped=skipped,
                    note="Compact evidence, not a full artifact archive. Unspecified state is not a completion certificate.")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    result = export(ROOT, args.output.resolve())
    print(f"Exported {len(result['files'])} compact files; skipped {len(result['skipped'])} unfinished studies.")
