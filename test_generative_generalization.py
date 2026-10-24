import unittest

import numpy as np

from evaluate_generative_generalization import balanced_part, prdc, semantic_diversity, training_preference


class ExternalOutcomesTest(unittest.TestCase):
    def test_heldout_parts_are_disjoint_and_balanced(self):
        first = balanced_part(list(range(800)), 0, 200)
        second = balanced_part(list(range(800)), 20, 200)
        self.assertEqual(len(set(first) & set(second)), 0)
        self.assertEqual(np.bincount(first // 80).tolist(), [20] * 10)

    def test_training_preference_direction(self):
        result = training_preference(np.array([[0.], [0.1]]), np.array([[0.]]), np.array([[1.]]))
        self.assertEqual(result["training_preference"], 1.)
        self.assertGreater(result["training_anchor_advantage"], 0.)

    def test_single_category_collapse(self):
        result = semantic_diversity(np.zeros((4, 2)), np.zeros(4, dtype=int),
                                    np.array([[0., 0.], [1., 1.]]), np.zeros(2, dtype=int))
        self.assertEqual(result["style_diversity_ratio"], 0.)
        self.assertEqual(result["effective_digit_modes"], 1.)
        self.assertEqual(result["represented_digits"], 1)

    def test_authors_coverage_detects_point_collapse(self):
        real = np.arange(20, dtype=float)[:, None]
        fake = np.repeat(real[:1], 20, axis=0)
        collapsed = prdc(real, fake, 3)
        matched = prdc(real, real + 0.001, 3)
        self.assertLess(collapsed["coverage"], matched["coverage"])
        self.assertLess(collapsed["recall"], matched["recall"])


if __name__ == "__main__":
    unittest.main()
