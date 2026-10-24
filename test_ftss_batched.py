from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn

from ftss_batched import estimate_ftss_batched
from run_matched_ftss_mlp import execution_only_update, original_definitions


class BatchedFTSSTest(unittest.TestCase):
    def test_execution_migration_cannot_change_training_protocol(self):
        old = {"runner_sha256": "old", "scratch_steps": 20000, "runs": 2}
        new = dict(old, runner_sha256="new", ftss_backend="batched", batched_source_sha256="hash")
        self.assertTrue(execution_only_update(old, new))
        self.assertFalse(execution_only_update(old, dict(new, scratch_steps=10000)))
        self.assertFalse(execution_only_update(old, dict(new, runs=1)))

    def test_matches_literal_estimator_and_rng(self):
        repo = Path(__file__).resolve().parent.parent / "ftss-memorization"
        if not repo.exists():
            self.skipTest("Sibling original evaluator unavailable")
        original = original_definitions(repo, torch.device("cpu"), None)["estimate_ftss"]
        original.__globals__["D"] = 4
        class Velocity(nn.Module):
            def forward(self, x, t):
                return -0.7 * x + torch.sin(t)
        torch.manual_seed(1234)
        literal = original(Velocity(), torch.device("cpu"), n_samples=3, n_steps=20)
        state = torch.get_rng_state()
        torch.manual_seed(1234)
        batched = estimate_ftss_batched(Velocity(), torch.device("cpu"), 3, 20, 4)
        torch.testing.assert_close(torch.get_rng_state(), state)
        np.testing.assert_allclose(batched["test_times"], literal["test_times"], atol=0, rtol=0)
        np.testing.assert_allclose(batched["mean_curve"], literal["mean_curve"], atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
