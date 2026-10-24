import unittest

import torch

from run_ftss_sd35_finetuning import integrate, interpolation, sensitivity


class SDFlowTests(unittest.TestCase):
    def test_noise_to_data_sign(self):
        clean = torch.ones(2, 1, 2, 2) * 3
        noise = torch.ones_like(clean) * 7
        x, target = interpolation(clean, noise, 1.)
        self.assertTrue(torch.equal(x, noise))
        actual = integrate(lambda x, sigma: target, x, [1., .5, 0.])
        torch.testing.assert_close(actual, clean)

    def test_identity_remaining_map(self):
        x = torch.randn(3, 1, 2, 2)
        directions = torch.randn(2, *x.shape)
        curve, endpoint = sensitivity(lambda x, sigma: torch.zeros_like(x), x,
                                      [1., .5, 0.], [0, 1], directions, .1, 2)
        torch.testing.assert_close(endpoint, x)
        torch.testing.assert_close(torch.tensor(curve["rms_curve"]), torch.ones(2))

    def test_known_contraction(self):
        x = torch.randn(3, 1, 2, 2)
        curve, _ = sensitivity(lambda x, sigma: x, x, [1., .5, 0.], [0, 1],
                               torch.randn(2, *x.shape), .1, 2)
        torch.testing.assert_close(torch.tensor(curve["rms_curve"]), torch.tensor([.25, .5]))

    def test_rms_not_mean_norm(self):
        x = torch.ones(4, 1, 2, 2)
        scale = torch.tensor([3., 1., 1., 1.]).reshape(1, 1, 2, 2)
        directions = torch.eye(4).reshape(1, 4, 1, 2, 2)
        curve, _ = sensitivity(lambda x, sigma: (1 - scale) * x, x,
                               [1., 0.], [0], directions, .1, 2)
        self.assertAlmostEqual(curve["rms_curve"][0], 3 ** .5, places=5)
        self.assertAlmostEqual(curve["mean_gain_curve"][0], 1.5, places=5)


if __name__ == "__main__":
    unittest.main()
