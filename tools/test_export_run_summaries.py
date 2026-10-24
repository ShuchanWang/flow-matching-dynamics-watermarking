import json
from pathlib import Path
import tempfile
import unittest

from tools.export_run_summaries import compact, export


class ExportTests(unittest.TestCase):
    def test_compaction_retains_evidence_not_credentials_or_scores(self):
        root = Path("/example/repo")
        value = {"seed": 41, "token": "private", "source": "/example/repo/run.py",
                 "trials": [{"exact": False, "bit_accuracy": .5, "scores": [1, 2], "decoded": [0, 1]}],
                 "paths": {"/example/repo/core.py": "hash"}}
        result = compact(value, root)
        self.assertEqual(result["seed"], 41)
        self.assertNotIn("token", result)
        self.assertEqual(result["trials"], [{"exact": False, "bit_accuracy": .5}])
        self.assertEqual(result["source"], "run.py")
        self.assertEqual(result["paths"], {"core.py": "hash"})

    def test_export_skips_running_and_never_copies_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            complete = root / "table_runs/tail_gate_pilot"
            running = root / "table_runs/baseline_first/tune"
            complete.mkdir(parents=True)
            running.mkdir(parents=True)
            source = complete / "results.json"
            source.write_text(json.dumps(dict(state="complete", rows=[dict(exact=False)])))
            before = source.read_bytes()
            (complete / "private_key.pt").write_bytes(b"do not export")
            (running / "results.json").write_text(json.dumps(dict(state="running")))
            result = export(root, root / "results")
            self.assertEqual(len(result["files"]), 1)
            self.assertEqual(result["skipped"], [dict(study="baseline_first/tune", state="running")])
            self.assertFalse(list((root / "results").rglob("*.pt")))
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(len(result["files"][0]["source_sha256"]), 64)

    def test_fpr_trials_export_as_rows_without_losing_outcomes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study = root / "table_runs/fresh_mlp_fpr"
            study.mkdir(parents=True)
            record = dict(trial=0, query_seed=1, results=[dict(scheme="hypercube32", key_seed=2,
                          claim_seed=3, claim="001", match=False, original_key_decoded="100")])
            (study / "results.json").write_text(json.dumps(dict(records=[record], complete=True)))
            result = export(root, root / "results")
            path = root / "results/verification/fresh_mlp_fpr"
            metadata = json.loads((path / "results.json").read_text())
            self.assertNotIn("records", metadata)
            self.assertEqual(metadata["trial_outcome_count"], 1)
            self.assertIn("hypercube32,2,3,False", (path / "trial_outcomes.csv").read_text())
            self.assertEqual(len(result["files"]), 2)


if __name__ == "__main__":
    unittest.main()
