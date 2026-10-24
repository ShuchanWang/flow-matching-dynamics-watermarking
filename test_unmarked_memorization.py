import unittest

import numpy as np
import torch

from run_unmarked_memorization import balanced_audit, balanced_split, copy_metrics, membership_auc
from analyze_unmarked_memorization import adjusted_correlation, correlation, fixed_error_copy_rates


class MemorizationMetricsTest(unittest.TestCase):
    def test_fixed_error_copies_require_tolerance_and_training_preference(self):
        rates = fixed_error_copy_rates([0.14, 0.28, 0.14], [1., 1., 0.1], 4)
        self.assertEqual(rates["visible_replica_rmse_005"], 1 / 3)
        self.assertEqual(rates["visible_replica_rmse_010"], 2 / 3)

    def test_constant_measurement_has_no_correlation(self):
        self.assertIsNone(correlation([1, 1, 1], [1, 2, 3]))

    def test_adjustment_removes_shared_step_effect(self):
        rows = [dict(run=run, size=10, step=step, contraction=step + run,
                     outcome_change=2 * step - run)
                for run in range(3) for step in (300, 1000, 3000)]
        self.assertIsNone(adjusted_correlation(rows, "outcome"))

    def test_adjustment_retains_within_trajectory_association(self):
        rows = [dict(run=run, size=10, step=step,
                     contraction=(run + 1) * step,
                     outcome_change=2 * (run + 1) * step + 3 * step + run)
                for run in range(3) for step in (1, 2, 3)]
        self.assertAlmostEqual(adjusted_correlation(rows, "outcome"), 1.)

    def test_balanced_splits_are_disjoint_even_at_largest_size(self):
        labels = np.repeat(np.arange(10), 1000)
        split = balanced_split(labels, 4000, 42)
        self.assertEqual(len(split["tuned"]), 4000)
        self.assertEqual(len(split["calibration"]), 800)
        combined = sum(split.values(), [])
        self.assertEqual(len(set(combined)), len(combined))
        for name in split:
            counts = np.bincount(labels[split[name]])
            self.assertTrue((counts == counts[0]).all())

    def test_loss_auc_direction_and_ties(self):
        self.assertEqual(membership_auc([0., 0.], [1., 1.]), 1.)
        self.assertEqual(membership_auc([1., 1.], [0., 0.]), 0.)
        self.assertEqual(membership_auc([1., 1.], [1., 1.]), 0.5)

    def test_visible_copy_and_raw_copy_are_distinct(self):
        tuned = torch.tensor([[1., 0.], [-1., 0.]])
        reference = torch.tensor([[0., 1.], [0., -1.]])
        calibration = torch.tensor([[0.6, 0.6], [-0.6, -0.6]])
        metrics, audit = copy_metrics(torch.tensor([[2., 0.]]), tuned, reference, calibration)
        self.assertEqual(metrics["raw_copy_rate_1pct"], 0.)
        self.assertEqual(metrics["visible_copy_rate_1pct"], 1.)
        self.assertEqual(metrics["visible_copy_coverage_1pct"], 0.5)
        self.assertEqual(audit["visible_train_nearest_indices"].tolist(), [0])

    def test_audit_subsampling_preserves_digits(self):
        pool = torch.arange(10).repeat_interleave(400).float()[:, None]
        audit = balanced_audit(pool)
        self.assertEqual(len(audit), 200)
        self.assertEqual(torch.bincount(audit.flatten().long()).tolist(), [20] * 10)


if __name__ == "__main__":
    unittest.main()
