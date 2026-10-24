import unittest

import torch
from torch import nn

from run_endpoint_loop_training import fitting_loss, loop_terms, teacher_target


class Linear(nn.Module):
    def forward(self, x, t, y=None):
        return 2 * x


class LoopTrainingTests(unittest.TestCase):
    def test_target_has_correct_coordinate_correction(self):
        x = torch.ones(2, 1, 1, 1)
        t = torch.tensor([.25, .75])
        q = torch.ones(1)
        inputs, target, _ = teacher_target(Linear(), x, t, None, q, 1.5, "loop")
        shift, signal, _ = loop_terms(t, q, 1.5, x.shape)
        torch.testing.assert_close(inputs, x + shift)
        torch.testing.assert_close(target, 2 * x + signal)
        self.assertFalse(torch.allclose(target, 2 * inputs + signal))

    def test_control_is_teacher_at_same_shifted_input(self):
        x = torch.ones(2, 1, 1, 1)
        inputs, target, _ = teacher_target(Linear(), x, torch.tensor([.25, .75]), None, torch.ones(1), 1.5, "control")
        torch.testing.assert_close(target, 2 * inputs)

    def test_weighted_error_has_same_optimum_not_amplitude_reward(self):
        target = torch.tensor([[1., -1.], [2., -2.]])
        for weight in (0., 10., 100.):
            pred = target.clone().requires_grad_()
            loss, _, _ = fitting_loss(pred, target, torch.eye(2), weight)
            loss.backward()
            self.assertEqual(float(loss), 0.)
            torch.testing.assert_close(pred.grad, torch.zeros_like(pred))
            self.assertGreater(float(fitting_loss(2 * target, target, torch.eye(2), weight)[0]), 0.)

    def test_loop_closes(self):
        shift, signal, _ = loop_terms(torch.tensor([0., 1.]), torch.ones(1), 2., (2, 1, 1, 1))
        self.assertEqual(float(shift.abs().max()), 0.)
        self.assertEqual(float(signal.abs().max()), 0.)


if __name__ == "__main__":
    unittest.main()
