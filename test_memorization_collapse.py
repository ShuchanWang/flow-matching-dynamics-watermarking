import unittest

import numpy as np

from analyze_memorization_collapse import dispersion


class CollapseTest(unittest.TestCase):
    def test_scale_contraction_and_rank_loss_are_different(self):
        x = np.array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])
        before = dispersion(x, [0] * 4, 1)
        after = dispersion(0.1 * x, [0] * 4, 1)
        self.assertAlmostEqual(after["output_variance"] / before["output_variance"], 0.01)
        self.assertAlmostEqual(after["participation_rank"], before["participation_rank"])

    def test_multiple_point_masses_collapse_locally_not_globally(self):
        x = np.array([[-1., 0.], [-1., 0.], [1., 0.], [1., 0.]])
        metrics = dispersion(x, [0, 0, 1, 1], 2)
        self.assertEqual(metrics["output_variance"], 1.)
        self.assertEqual(metrics["within_prototype_variance"], 0.)
        self.assertEqual(metrics["effective_occupied_prototypes"], 2.)
        self.assertEqual(metrics["within_prototype_pair_mse"], 0.)

    def test_single_point_global_collapse(self):
        metrics = dispersion(np.ones((4, 2)), [0] * 4, 2)
        self.assertEqual(metrics["output_variance"], 0.)
        self.assertEqual(metrics["participation_rank"], 0.)
        self.assertEqual(metrics["largest_prototype_mass"], 1.)

    def test_singletons_do_not_create_pairwise_evidence(self):
        metrics = dispersion(np.eye(4), range(4), 4)
        self.assertEqual(metrics["within_prototype_variance"], 0.)
        self.assertIsNone(metrics["within_prototype_pair_mse"])
        self.assertEqual(metrics["within_prototype_ordered_pairs"], 0)

    def test_pairwise_dispersion(self):
        metrics = dispersion([[0., 0.], [2., 0.], [10., 0.]], [0, 0, 1], 2)
        self.assertEqual(metrics["within_prototype_pair_mse"], 4.)
        self.assertAlmostEqual(metrics["within_prototype_variance"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
