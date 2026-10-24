import unittest
import numpy as np

from run_ftss_diagnostic_validation import build_pools, exposure_weights, fit_rule, confusion, metrics_summary, replication_metrics


class DiagnosticValidationTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.repeat(np.arange(10), 1200)

    def test_identical_support_and_disjoint_audit_pools(self):
        pools = build_pools(self.labels, 42)
        self.assertEqual(pools["uniform"], pools["repeat"])
        for p in pools.values():
            self.assertEqual(len(set(p["train"])), 400)
            self.assertEqual(set(p["audit"]),set(p["train"]))
            self.assertTrue(set(p["audit"]).issubset(p["train"]))
            self.assertFalse(set(p["train"]) & set(p["reference"]+p["calibration"]))
            self.assertFalse(set(p["reference"]) & set(p["calibration"]))
        self.assertEqual(set(self.labels[pools["specialize"]["train"]]), {1})
        np.testing.assert_array_equal(np.bincount(self.labels[pools["uniform"]["train"]]), np.full(10,40))

    def test_repetition_changes_exposure_not_training_support(self):
        p = build_pools(self.labels, 42)["repeat"]
        w = exposure_weights(p)
        target = np.isin(p["train"], p["targets"])
        self.assertTrue(np.all(w > 0))
        self.assertAlmostEqual(w.sum(), 1.)
        self.assertAlmostEqual(w[target].sum(), .8)
        self.assertTrue(set(p["targets"]).issubset(p["audit"]))

    def test_insufficient_narrow_data_is_not_filled_with_duplicates(self):
        with self.assertRaises(ValueError):
            build_pools(np.repeat(np.arange(10),900),42)

    def test_calibration_rule_uses_only_contraction_thresholds(self):
        rule = fit_rule([.2,.4,.9,1.1],[True,True,False,False])
        self.assertGreaterEqual(rule["threshold"], .4)
        self.assertLess(rule["threshold"], .9)
        self.assertEqual(rule["calibration_balanced_accuracy"], 1.)
        with self.assertRaises(ValueError):
            fit_rule([.2,.4],[False,False])

    def test_empty_positive_class_does_not_report_perfect_sensitivity(self):
        m = confusion([False,False],[False,True])
        self.assertIsNone(m["sensitivity"])
        self.assertEqual(m["false_alarm_rate"], .5)

    def test_primary_counts_exclude_repeated_intermediate_checkpoints(self):
        rows = [dict(model="unet", seed=42, arm="specialize",step=s, ftss_ratio=.2,
                     metrics=dict(copy_excess=0.)) for s in (0,1000,3000)]
        result = metrics_summary(rows,.5,.01,3000)["unet"]["specialize"]
        self.assertEqual(result["checkpoints"],1)
        self.assertEqual(result["fp"],1)

    def test_common_patterns_matching_both_pools_are_not_training_specific(self):
        m,_ = replication_metrics([0.,10.,1.,10.],[10.,0.,10.,1.],.05)
        self.assertEqual(m["copy_rate"],.5)
        self.assertEqual(m["reference_copy_rate"],.5)
        self.assertEqual(m["copy_excess"],0.)


if __name__ == "__main__":
    unittest.main()
