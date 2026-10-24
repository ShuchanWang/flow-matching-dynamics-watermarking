import math
import unittest

import torch
from torch import nn

from run_endpoint_loop_pilot import (alpha, beta, grid_carrier, ConjugatedField, integrate,
                                     endpoint_metrics, detection_metrics)


class Linear(nn.Module):
    def forward(self, x, t, y=None):
        return .7 * x


class Zero(nn.Module):
    def forward(self, x, t, y=None):
        return torch.zeros_like(x)


class EndpointLoopTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(120)

    def test_loop_closes_and_grid_forcing_telescopes(self):
        self.assertEqual(beta(0), 0.)
        self.assertEqual(beta(1), 0.)
        grid = [0., .01, .1, .6, .8, 1.]
        integral = sum((b - a) * grid_carrier(a, b) for a, b in zip(grid[:-1], grid[1:]))
        self.assertAlmostEqual(integral, 0., places=15)

    def test_continuous_field_matches_translated_trajectory(self):
        initial = torch.tensor([.2, -.3], dtype=torch.float64).reshape(2, 1, 1, 1)
        times = torch.tensor([.25, .75], dtype=torch.float64)
        x = initial * (.7 * times).exp()[:, None, None, None]
        shifted = x + torch.tensor([2 * beta(float(t)) for t in times])[:, None, None, None]
        marked = ConjugatedField(Linear(), torch.ones(1, dtype=torch.float64), 2.)
        actual = marked(shifted, times)
        expected = .7 * x + torch.tensor([2 * alpha(float(t)) for t in times])[:, None, None, None]
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)

    def test_canonical_sampler_is_independently_bitwise_equal(self):
        x = torch.randn(8, 1, 1, 1)
        p, q = torch.ones(1, 1), torch.ones(1)
        grid = torch.linspace(0, 1, 51).tolist()
        clean = integrate(Linear(), x, None, grid, p, q, 2., "clean")
        marked = integrate(Linear(), x, None, grid, p, q, 2., "canonical_exact")
        self.assertTrue(endpoint_metrics(clean["endpoint"], marked["endpoint"])["bitwise_equal"])

    def test_grid_conjugacy_survives_clipping_and_nonuniform_grid(self):
        x = torch.tensor([4.9, -4.9], dtype=torch.float64).reshape(2, 1, 1, 1)
        p, q = torch.ones(1, 1, dtype=torch.float64), torch.ones(1, dtype=torch.float64)
        grid = [0., .01, .1, .6, .8, 1.]
        clean = integrate(Linear(), x, None, grid, p, q, 20., "clean")
        marked = integrate(Linear(), x, None, grid, p, q, 20., "grid_conjugate")
        torch.testing.assert_close(clean["endpoint"], marked["endpoint"], atol=1e-12, rtol=1e-12)

    def test_continuous_euler_error_decreases_with_step_size(self):
        x = torch.full((2, 1, 1, 1), .2)
        p, q = torch.ones(1, 1), torch.ones(1)
        errors = []
        for steps in (25, 50):
            grid = torch.linspace(0, 1, steps + 1, dtype=torch.float64).tolist()
            clean = integrate(Linear(), x, None, grid, p, q, 2., "clean", clip=None)
            marked = integrate(Linear(), x, None, grid, p, q, 2., "continuous_euler", clip=None)
            errors.append(float((marked["endpoint"] - clean["endpoint"]).abs().max()))
        self.assertGreater(errors[0], 1e-5)
        self.assertLess(errors[1], errors[0])

    def test_uncompensated_additive_carrier_changes_endpoints(self):
        x = torch.full((2, 1, 1, 1), .2)
        p, q = torch.ones(1, 1), torch.ones(1)
        grid = torch.linspace(0, 1, 101, dtype=torch.float64).tolist()
        clean = integrate(Linear(), x, None, grid, p, q, 2., "clean", clip=None)
        additive = integrate(Linear(), x, None, grid, p, q, 2., "additive_sine", clip=None)
        self.assertGreater(float((clean["endpoint"] - additive["endpoint"]).abs().max()), .01)

    def test_decoder_uses_no_clean_subtraction(self):
        code = torch.tensor([1., -1.]) / math.sqrt(2)
        signature = (1.5 * code.double()).expand(8, -1)
        result = detection_metrics(signature, torch.full_like(signature, 100.), code, torch.eye(2),
                                   torch.tensor([1, 0]), 2)
        self.assertEqual(result["recovery_percent"], 100.)
        self.assertEqual(result["independent_trajectories"], 8)

    def test_exact_endpoints_do_not_imply_unbiased_detection(self):
        class Counterexample(nn.Module):
            def forward(self, x, t, y=None):
                return -torch.sin(2 * math.pi * t)[:, None, None, None].expand_as(x)
        field = ConjugatedField(Counterexample(), torch.ones(1), 1.)
        x = torch.randn(8, 1, 1, 1)
        times = torch.rand(8)
        self.assertLess(float(field(x, times).abs().max()), 1e-6)

    def test_bitwise_check_distinguishes_signed_zero(self):
        metrics = endpoint_metrics(torch.tensor([0.]), torch.tensor([-0.]))
        self.assertFalse(metrics["bitwise_equal"])
        self.assertEqual(metrics["image_mse"], 0.)


if __name__ == "__main__":
    unittest.main()
