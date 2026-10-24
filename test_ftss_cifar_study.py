import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ftss_batched import estimate_ftss_batched
from run_ftss_cifar_study import FixedClassFlow, conditional_neighbors, generate, flow_losses, summarize, OUTCOMES, Measurement
from run_ftss_outcome_study import make_split


class ConditionalLinear(torch.nn.Module):
    def forward(self, x, t, y):
        return -(0.1+0.02*y).reshape(-1, *([1]*(x.ndim-1))) * x


class CifarFTSSTests(unittest.TestCase):
    def test_fixed_class_branch_order_and_chunking(self):
        labels = torch.arange(10)
        model = ConditionalLinear()
        wrapper = FixedClassFlow(model, (1, 2, 2), labels, batch=7)
        x = torch.ones(30, 4)
        expected = -(0.1+0.02*labels.repeat(3))[:, None].expand(-1, 4)
        torch.testing.assert_close(wrapper(x, torch.zeros(30, 1)), expected)
        with self.assertRaises(ValueError):
            wrapper(torch.ones(11, 4), torch.zeros(11, 1))

    def test_conditioned_ftss_matches_linear_solution(self):
        labels = torch.arange(10)
        model = FixedClassFlow(ConditionalLinear(), (1, 2, 2), labels, batch=7)
        torch.manual_seed(42)
        result = estimate_ftss_batched(model, torch.device("cpu"), 10, 20, 4)
        rates = .1+.02*labels.numpy()
        expected = [np.sqrt(np.mean((1-rates/20)**(2*(20-int(t*20))))) for t in result["test_times"]]
        np.testing.assert_allclose(result["mean_curve"], expected, atol=3e-6)

    def test_nearest_neighbors_do_not_cross_conditions(self):
        d, i = conditional_neighbors(np.array([[0.], [10.]]), np.array([0, 1]),
                                     np.array([[10.], [0.]]), np.array([0, 1]))
        np.testing.assert_allclose(d, [10, 10])
        np.testing.assert_array_equal(i, [0, 1])

    def test_generation_is_class_balanced_and_reproducible(self):
        a, labels = generate(ConditionalLinear(), torch.device("cpu"), 20, 4, 7, 42)
        b, _ = generate(ConditionalLinear(), torch.device("cpu"), 20, 4, 20, 42)
        torch.testing.assert_close(a, b)
        np.testing.assert_array_equal(np.bincount(labels), np.full(10, 2))
        self.assertLessEqual(float(a.abs().max()), 1.)

    def test_identical_loss_pools_have_chance_membership(self):
        images = torch.randn(10, 3, 32, 32)
        labels = torch.arange(10)
        split = dict(tuned=list(range(10)), reference=list(range(10)))
        m, _ = flow_losses(ConditionalLinear(), images, labels, split, torch.device("cpu"), 42, repeats=2, batch=7)
        self.assertEqual(m["loss_gap"], 0.)
        self.assertAlmostEqual(m["membership_auc"], .5)

    def test_summary_preserves_baseline_and_stratifies(self):
        before = dict.fromkeys(OUTCOMES, .5)
        before["min"] = 1.
        after = dict.fromkeys(OUTCOMES, .75)
        after["min"] = .5
        rows = [dict(seed=0, size=50, step=step, metrics=m) for step, m in ((0, before), (1000, after))]
        s = summarize(rows)
        self.assertEqual(s["changes"][0]["ftss_ratio"], .5)
        self.assertEqual(s["changes"][0]["training_preference_change"], .25)
        self.assertIn("50", s["associations"]["loss_gap"]["by_size"])

    def test_real_quality_control_is_disjoint_and_balanced(self):
        labels = torch.arange(10).repeat_interleave(900)
        with tempfile.TemporaryDirectory() as directory:
            evaluator = SimpleNamespace(args=SimpleNamespace(output=Path(directory), samples=20),
                                        labels=labels, bank=np.random.default_rng(42).normal(size=(9000, 8)).astype(np.float32))
            Measurement.controls(evaluator, 0)
            result = json.loads((Path(directory) / "quality_controls_seed0.json").read_text())
            split = make_split(labels.numpy(), 1000, 71000)
            excluded = set(split["tuned"]+split["reference"]+split["calibration"])
            self.assertFalse(set(result["comparison_indices"]) & excluded)
            np.testing.assert_array_equal(np.bincount(labels[result["comparison_indices"]].numpy()), np.full(10, 2))
            self.assertTrue(np.isfinite(result["fid"]))


if __name__ == "__main__":
    unittest.main()
