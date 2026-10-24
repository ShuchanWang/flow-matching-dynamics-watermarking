import inspect
import math
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from run_tail_gate_pilot import TailGate, GatedField, probe, probe_diagnostics, coherent_amplitude, embedding_loss


class SimpleVelocity(nn.Module):
    def forward(self, x, t, y=None):
        return torch.sin(2 * math.pi * t)[:, None, None, None] * torch.tanh(x)


class TailGateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(118)
        self.direction = torch.randn(12)
        self.direction /= self.direction.norm()
        self.gate = TailGate(self.direction, .3, .7, 3., 4.)
        self.x, self.t = torch.randn(100, 1, 3, 4), torch.rand(100)

    def test_normalization_matches_interpolation_moments(self):
        mean, scale = self.gate.moments(torch.tensor([0., 1.]))
        torch.testing.assert_close(mean, torch.tensor([0., .3]))
        torch.testing.assert_close(scale, torch.tensor([1., math.sqrt(.7)]))

    def test_verifier_shift_hits_requested_coordinate(self):
        position = torch.linspace(3.5, 4.5, len(self.x))
        trigger = self.gate.trigger(self.x, self.t, position)
        torch.testing.assert_close(self.gate.coordinate(trigger, self.t), position)
        self.assertGreater(float(self.gate(trigger, self.t).min()), .88)
        delta = (trigger - self.x).flatten(1)
        orthogonal = delta - (delta @ self.direction)[:, None] * self.direction
        self.assertLess(float(orthogonal.abs().max()), 1e-6)

    def test_gate_is_smooth_and_finite(self):
        x = self.x.requires_grad_(True)
        gradient = torch.autograd.grad(self.gate(x, self.t).sum(), x)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.norm()), 0)

    def test_higher_threshold_reduces_ordinary_activation(self):
        high = TailGate(self.direction, .3, .7, 5., 4.)
        self.assertTrue((high(self.x, self.t) < self.gate(self.x, self.t)).all())

    def test_injected_field_has_expected_signal(self):
        direction = torch.randn(12)
        base = SimpleVelocity()
        marked = GatedField(base, self.gate, direction, 3.)
        observed = marked(self.x, self.t) - base(self.x, self.t)
        expected = (3 * (torch.sin(2 * math.pi * self.t) * self.gate(self.x, self.t))[:, None] * direction).reshape_as(self.x)
        torch.testing.assert_close(observed, expected, atol=1e-7, rtol=1e-5)

    def test_training_penalizes_always_on_shortcut(self):
        projection = torch.linalg.qr(torch.randn(12, 4)).Q
        code = torch.tensor([1., -1., 1., -1.]) / 2
        trigger = self.gate.trigger(self.x, self.t, torch.full_like(self.t, 4.))
        zeros = torch.zeros_like(self.x)
        sine = torch.sin(2 * math.pi * self.t)
        on = ((3 * sine * self.gate(self.x, self.t))[:, None] * code @ projection.T).reshape_as(self.x)
        off = ((3 * sine * self.gate(trigger, self.t))[:, None] * code @ projection.T).reshape_as(self.x)
        shortcut = ((3 * sine)[:, None] * code @ projection.T).reshape_as(self.x)
        perfect = embedding_loss(on, off, zeros, zeros, zeros, self.x, trigger, self.t,
                                 self.gate, projection, code, 3., 10.)
        bad = embedding_loss(shortcut, off, zeros, zeros, zeros, self.x, trigger, self.t,
                             self.gate, projection, code, 3., 10.)
        self.assertLess(float(perfect), 1e-10)
        self.assertGreater(float(bad), .1)

    def test_coherence_removes_affine_time_nuisance(self):
        class Affine(nn.Module):
            def forward(self, x, t, y=None):
                return x * (1 + t[:, None, None, None])
        self.assertLess(float(coherent_amplitude(Affine(), self.x, None).abs().max()), 1e-5)

    def test_probe_has_no_owner_argument_and_supports_gradients(self):
        self.assertEqual(list(inspect.signature(probe).parameters), ["model", "args", "device", "gradient"])
        args = SimpleNamespace(seed=1, probe_candidates=6, probe_batch=3, probe_steps=2)
        for gradient in (False, True):
            candidates, diagnostics = probe(SimpleVelocity(), args, torch.device("cpu"), gradient)
            self.assertEqual(candidates.shape, (6, 3, 32, 32))
            self.assertFalse(candidates.requires_grad)
            self.assertTrue(torch.isfinite(candidates).all())
            self.assertEqual(diagnostics["forward_evaluations"], 6 * 4 * (3 if gradient else 1))
            self.assertEqual(diagnostics["backward_evaluations"], 12 if gradient else 0)

    def test_diagnostic_does_not_change_probe_selection(self):
        gate = TailGate(torch.randn(3072), 0., 1., 3., 4.)
        x = torch.randn(6, 3, 32, 32)
        before = x.clone()
        result = probe_diagnostics(x, gate)
        torch.testing.assert_close(x, before)
        self.assertTrue(0 <= result["selected_activation_rate"] <= 1)


if __name__ == "__main__":
    unittest.main()
