"""Mechanism checks for the small secret-carrier pilot."""

import math
import unittest

import torch
import numpy as np
from scipy.linalg import sqrtm

from run_secret_carrier_pilot import BlindRegressor, OwnerCarrier, time_features, pilot_fid


class SecretCarrierTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(411)
        self.x = torch.randn(2048, 1, 4, 4)
        self.t = torch.rand(len(self.x))
        self.p = torch.linalg.qr(torch.randn(16, 4)).Q
        self.codes = torch.eye(4)

    def owner(self, kind):
        return OwnerCarrier(kind, self.p, self.codes, "1010")

    def test_secret_time_code_is_still_rank_one(self):
        owner = self.owner("multitone")
        code = torch.tensor([1., -1., 1., -1.]) / 2
        signal = owner.signal(self.x, self.t, code, 1.5).flatten(1)
        singular = torch.linalg.svdvals(signal)
        self.assertLess(float(singular[1] / singular[0]), 1e-6)

    def test_joint_spectrum_recovers_unknown_multitone(self):
        owner = self.owner("multitone")
        code = torch.tensor([1., -1., 1., -1.]) / 2
        signal = owner.signal(self.x, self.t, code, 1.5)
        fit = BlindRegressor("time", 16, features=8).fit(self.x, self.t, signal, ridge=1e-8)
        test_x, test_t = torch.randn(200, 1, 4, 4), torch.rand(200)
        torch.testing.assert_close(fit(test_x, test_t), owner.signal(test_x, test_t, code, 1.5), atol=1e-5, rtol=1e-5)

    def test_input_carrier_varies_direction_and_centers_calibration(self):
        owner = self.owner("input")
        owner.calibrate(self.x)
        gates = owner.gates(self.x, torch.full_like(self.t, .25))
        torch.testing.assert_close(gates.mean(0), torch.zeros(4), atol=1e-6, rtol=0)
        torch.testing.assert_close(gates.square().mean(0), torch.ones(4), atol=1e-6, rtol=0)
        signal = owner.signal(self.x, self.t, torch.ones(4) / 2, 1.5).flatten(1)
        singular = torch.linalg.svdvals(signal)
        self.assertGreater(float(singular[3] / singular[0]), .05)

    def test_owner_can_demodulate_all_three_carriers(self):
        code = torch.tensor([1., -1., 1., -1.]) / 2
        for kind in ("sine", "multitone", "input"):
            owner = self.owner(kind)
            owner.calibrate(self.x)
            scores = owner.scores(owner.signal(self.x, self.t, code, 1.5), self.x, self.t).mean(0)
            self.assertEqual((scores > 0).tolist(), [True, False, True, False])

    def test_time_bin_svd_fits_without_owner_information(self):
        regressor = BlindRegressor("bins", 16, features=8, rank=1)
        levels = torch.linspace(-1, 1, 16)
        direction = torch.randn(16)
        bins = (self.t * 16).long()
        signal = (levels[bins, None] * direction).reshape_as(self.x)
        regressor.fit(self.x, self.t, signal, ridge=1e-8)
        fitted = regressor(self.x, self.t).flatten(1)
        centered = signal.flatten(1) - signal.flatten(1).mean(0)
        torch.testing.assert_close(fitted, centered, atol=2e-5, rtol=2e-5)

    def test_input_attack_generalizes_for_its_feature_family(self):
        regressor = BlindRegressor("input_ridge", 16, features=8)
        weights = torch.randn(regressor.design(self.x, self.t).shape[1], 16)
        weights[0].zero_()
        velocity = (regressor.design(self.x, self.t) @ weights).reshape_as(self.x)
        regressor.fit(self.x, self.t, velocity, ridge=1e-8)
        x, t = torch.randn(100, 1, 4, 4), torch.rand(100)
        expected = (regressor.design(x, t) @ weights).reshape_as(x)
        torch.testing.assert_close(regressor(x, t), expected, atol=2e-5, rtol=2e-5)

    def test_time_basis_matches_sine_decoder(self):
        torch.testing.assert_close(time_features(self.t, 8)[:, 0], torch.sin(2 * math.pi * self.t))

    def test_sample_space_fid_matches_full_covariance(self):
        a, b = torch.randn(40, 8).double(), torch.randn(50, 8).double()
        ca, cb = np.cov(a.numpy(), rowvar=False), np.cov(b.numpy(), rowvar=False)
        expected = float((a.mean(0) - b.mean(0)).square().sum()) + np.trace(ca + cb - 2 * sqrtm(ca @ cb))
        self.assertAlmostEqual(pilot_fid(a, b), float(expected), places=10)
        self.assertLess(pilot_fid(a, a), 1e-10)


if __name__ == "__main__":
    unittest.main()
