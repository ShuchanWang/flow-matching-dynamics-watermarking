import unittest

import torch

from evaluate_fresh_mlp_fpr import binomial_summary, weighted_velocity_mean
from repair_mlp_watermarks import signature


class FreshQueryTest(unittest.TestCase):
    def test_shared_mean_matches_original_projected_verifier(self):
        torch.manual_seed(42)
        linear = torch.nn.Linear(785, 784).eval()
        def model(x, t):
            return linear(torch.cat((x, t), dim=1))
        projection = torch.linalg.qr(torch.randn(784, 32))[0]
        with torch.inference_mode():
            expected = signature(model, projection, 65, torch.Generator().manual_seed(8), batch=16)
            mean = weighted_velocity_mean(model, 65, torch.Generator().manual_seed(8), "cpu", 16)
        torch.testing.assert_close(mean @ projection, expected, rtol=1e-5, atol=1e-6)

    def test_zero_matches_have_positive_upper_bound(self):
        result = binomial_summary(0, 10000)
        self.assertEqual(result["fpr_pct"], 0)
        self.assertAlmostEqual(result["zero_match_one_sided_95_upper_pct"], .0299528360)

    def test_invalid_counts_rejected(self):
        for matches, trials in ((1, 0), (-1, 10), (11, 10)):
            with self.assertRaises(ValueError):
                binomial_summary(matches, trials)


if __name__ == "__main__":
    unittest.main()
