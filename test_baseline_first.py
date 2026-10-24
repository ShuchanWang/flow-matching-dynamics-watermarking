import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from run_baseline_first import (LAYER, make_key, code_for, trigger_image, trigger_bits,
                                projected_loss, weight_logits, kid, select_trials, binomial_interval)


class WeightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 4, 3, 3) * .01)

    def named_parameters(self, *args, **kwargs):
        return iter([(LAYER, self.weight)])


class BaselineFirstTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.key = make_key(32, 150, (8, 4, 3, 3), torch.device("cpu"))

    def test_private_keys_differ_public_control_does_not(self):
        other = make_key(32, 151, (8, 4, 3, 3), torch.device("cpu"))
        self.assertFalse(torch.equal(self.key["projection"], other["projection"]))
        self.assertFalse(torch.equal(self.key["weight_projection"], other["weight_projection"]))
        torch.testing.assert_close(self.key["public_projection"], other["public_projection"])

    def test_matched_payload_and_orthogonal_code(self):
        key = make_key(5, 150, (8, 4, 3, 3), torch.device("cpu"))
        code = code_for(key, "dynamics")
        decoded = int((code @ key["codes"].T).argmax())
        self.assertEqual([(decoded >> i) & 1 for i in range(5)], key["bits"].tolist())
        self.assertEqual(len(code_for(key, "naive")), 5)
        self.assertAlmostEqual(float(code_for(self.key, "dynamics").norm()), 1., places=6)

    def test_trigger_tile_roundtrip(self):
        for payload in (5, 32):
            key = make_key(payload, 411, (8, 4, 3, 3), torch.device("cpu"))
            image = trigger_image(key)
            self.assertEqual(image.shape, (1, 3, 32, 32))
            torch.testing.assert_close(trigger_bits(image, key)[0], key["bits"].bool())

    def test_uchida_regularizer_changes_the_selected_weights(self):
        model = WeightModel()
        before = model.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=.02)
        for _ in range(50):
            loss = F.binary_cross_entropy_with_logits(weight_logits(model, self.key), self.key["bits"].float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertFalse(torch.equal(before, model.weight))
        self.assertTrue(torch.equal(weight_logits(model, self.key) > 0, self.key["bits"].bool()))

    def test_zero_strength_control_has_no_correlation_reward(self):
        prediction = torch.zeros(4, 3, 32, 32, requires_grad=True)
        zero = torch.zeros_like(prediction)
        t = torch.tensor([.1, .3, .6, .9])
        loss = projected_loss(prediction, zero, zero, t, self.key, "dynamics", 0.)
        loss.backward()
        self.assertEqual(float(loss), 0.)
        self.assertEqual(float(prediction.grad.norm()), 0.)

    def test_signal_objectives_have_finite_gradients(self):
        for method in ("dynamics", "naive"):
            prediction = torch.randn(4, 3, 32, 32, requires_grad=True)
            loss = projected_loss(prediction, torch.zeros_like(prediction), torch.zeros_like(prediction),
                                  torch.rand(4), self.key, method, 1.5)
            loss.backward()
            self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_kid_matches_unbiased_mmd(self):
        args = SimpleNamespace(eval_seed=1, kid_subset=6, kid_subsets=1)
        a, b = torch.randn(6, 3).double(), torch.randn(6, 3).double()
        xx, yy, xy = (a @ a.T / 3 + 1).pow(3), (b @ b.T / 3 + 1).pow(3), (a @ b.T / 3 + 1).pow(3)
        expected = (xx.sum() - xx.trace() + yy.sum() - yy.trace()) / 30 - 2 * xy.mean()
        self.assertAlmostEqual(kid(a, b, args)["mean"], float(expected), places=10)
        self.assertAlmostEqual(kid(a, a, args, paired=True)["mean"], 0., places=10)

    def test_selection_obeys_recovery_and_quality_gate(self):
        rows = [dict(payload=32, method="weight", trial=i, strength=float(i + 1),
                     metrics=dict(recovery_percent=recovery, fid_ratio_control=ratio))
                for i, (recovery, ratio) in enumerate([(100, 1.06), (95, 1.04), (100, 1.03)])]
        selected = select_trials(rows)[0]
        self.assertEqual(selected["strength"], 3.)
        self.assertTrue(selected["qualified"])
        rows[1]["metrics"]["recovery_percent"] = 0
        rows[2]["metrics"]["recovery_percent"] = 0
        selected = select_trials(rows)[0]
        self.assertFalse(selected["qualified"])
        self.assertEqual(selected["strength"], 1.)

    def test_exact_intervals_keep_key_count_distinct(self):
        self.assertEqual(binomial_interval(0, 5)[0], 0.)
        self.assertEqual(binomial_interval(5, 5)[1], 1.)
        self.assertLess(binomial_interval(5, 5)[0], .5)


if __name__ == "__main__":
    unittest.main()
