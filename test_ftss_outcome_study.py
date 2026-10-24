import unittest
import numpy as np
import torch

from run_ftss_outcome_study import make_split, correlation, loss_metrics, summarize


class OutcomeStudyTests(unittest.TestCase):
    def test_nested_balanced_disjoint_splits(self):
        labels = np.repeat(np.arange(10), 900)
        small = make_split(labels, 50, 10)
        large = make_split(labels, 1000, 10)
        self.assertTrue(set(small["tuned"]).issubset(large["tuned"]))
        self.assertTrue(set(small["reference"]).issubset(large["reference"]))
        self.assertEqual(small["calibration"], large["calibration"])
        self.assertFalse(set(large["tuned"]) & set(large["reference"]))
        self.assertFalse(set(large["tuned"]) & set(large["calibration"]))
        self.assertFalse(set(large["reference"]) & set(large["calibration"]))
        np.testing.assert_array_equal(np.bincount(labels[small["tuned"]]), np.full(10, 5))

    def test_paired_loss_has_zero_gap_for_identical_pools(self):
        class Zero(torch.nn.Module):
            def forward(self, x, t):
                return torch.zeros_like(x)
        images = torch.randn(12, 4)
        m, values = loss_metrics(Zero(), images, {"tuned": list(range(12)), "reference": list(range(12))},
                                torch.device("cpu"), 9)
        self.assertEqual(m["loss_gap"], 0.)
        self.assertAlmostEqual(m["membership_auc"], .5)
        np.testing.assert_array_equal(*values)

    def test_baseline_relative_contraction_and_outcomes(self):
        keys = ("training_preference", "training_anchor_advantage", "loss_gap", "membership_auc",
                "coverage", "precision", "style_diversity_ratio", "effective_digit_modes")
        before = dict.fromkeys(keys, .5)
        before["min"] = 2.
        after = dict.fromkeys(keys, .75)
        after["min"] = 1.
        rows = [dict(base=0, seed=1, size=50, step=step, metrics=m) for step, m in ((0, before), (3000, after))]
        changes = summarize(rows)["changes"]
        self.assertEqual(changes[0]["ftss_ratio"], .5)
        self.assertAlmostEqual(changes[0]["contraction"], np.log(2))
        self.assertEqual(changes[0]["training_preference_change"], .25)

    def test_degenerate_association_is_not_nan(self):
        self.assertIsNone(correlation([1, 1, 1], [2, 3, 4]))
        self.assertIsNone(correlation([], []))
        self.assertAlmostEqual(correlation([1, 2, 3], [3, 2, 1]), -1.)


if __name__ == "__main__":
    unittest.main()
