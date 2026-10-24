import unittest
from pathlib import Path

from run_audio_ownership import summarize
from audit_saved_metrics import extract
from finalize_audio_ownership import render


class OwnershipMetricsTest(unittest.TestCase):
    def records(self):
        return [{"wm": {"decoded": [1, 0], "exact": True, "target_score": 2., "bit_errors": 0},
                 "clean": {"exact": True, "target_score": 0.},
                 "wrong_key": {"exact": False, "target_score": 3.}}
                for _ in range(20)]

    def test_gate_requires_both_score_and_message(self):
        result = summarize([0.] * 40, self.records())
        self.assertEqual(result["ownership_fpr_pct"], 0.)
        self.assertEqual(result["wrong_key_acceptance_pct"], 0.)
        self.assertEqual(result["ownership_fnr_pct"], 0.)
        self.assertEqual(result["score_auc_vs_clean"], 1.)
        self.assertEqual(result["score_auc_vs_wrong_key"], 0.)

    def test_fnr_is_not_recovery_failure(self):
        result = summarize([2.] * 40, self.records())
        self.assertEqual(result["ownership_fnr_pct"], 100.)
        self.assertEqual(result["recovery_failure_pct"], 0.)

    def test_calibration_uses_population_std(self):
        result = summarize([0., 2.] * 20, self.records())
        self.assertEqual(result["threshold"], 4.)

    def test_bit_errors_and_score_ties(self):
        records = self.records()
        records[0]["wm"].update(decoded=[0, 0], exact=False, bit_errors=1)
        for row in records:
            row["clean"]["target_score"] = 2.
        result = summarize([0.] * 40, records)
        self.assertEqual(result["ber_pct"], 2.5)
        self.assertEqual(result["ownership_fnr_pct"], 5.)
        self.assertEqual(result["score_auc_vs_clean"], .5)

    def test_incomplete_calibration_rejected(self):
        with self.assertRaises(ValueError):
            summarize([0.] * 39, self.records())

    def test_audit_imports_calibrated_rates(self):
        from run_audio_ownership import PROTOCOL
        payload = {"calibration_scores": [0.] * 40, "protocol": PROTOCOL,
                   "metrics": summarize([0.] * 40, self.records()),
                   "architecture": "mlp", "message": "10", "codebook_mode": "hypercube", "caveat": "test"}
        row = extract(Path("test.json"), payload)[0]
        self.assertEqual(row["phase"], "ownership_audit")
        self.assertEqual(row["ownership_fnr_pct"], 0.)
        self.assertEqual(row["roc_auc"], 1.)

    def test_paper_refuses_missing_groups(self):
        with self.assertRaises(ValueError):
            render([])


if __name__ == "__main__":
    unittest.main()
