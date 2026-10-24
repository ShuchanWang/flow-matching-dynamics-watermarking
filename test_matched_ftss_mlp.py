from pathlib import Path
import random
import unittest

import numpy as np
import torch
from torch import nn

from run_matched_ftss_mlp import original_definitions, planned_subsets


class MatchedPaperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo = Path(__file__).resolve().parent.parent / "ftss-memorization"
        if not (repo / "train_memorization_mlp.py").exists():
            raise unittest.SkipTest("Sibling source unavailable")
        cls.source = original_definitions(repo, torch.device("cpu"), None)

    def test_original_architecture(self):
        model = self.source["MLPFlow"](D=2, hidden_dim=8)
        self.assertEqual(sum(isinstance(m, nn.Linear) for m in model.net), 5)
        self.assertEqual(sum(isinstance(m, nn.LayerNorm) for m in model.net), 4)
        self.assertEqual(sum(isinstance(m, nn.Dropout) for m in model.net), 4)
        self.assertEqual(model(torch.zeros(3, 2), torch.zeros(3, 1)).shape, (3, 2))

    def test_resume_preserves_original_subset_rng_order(self):
        plans = planned_subsets([50, 100], 2, 1000)
        rng = random.Random(42)
        for key in ((50, 0), (50, 1), (100, 0), (100, 1)):
            expected = rng.sample(range(1000), key[0])
            state, indices = plans[key]
            replay = random.Random()
            replay.setstate(state)
            self.assertEqual(expected, indices)
            self.assertEqual(replay.sample(range(1000), key[0]), indices)

    def test_literal_paper_estimator_times_and_identity(self):
        class Identity(nn.Module):
            def forward(self, x, t):
                return torch.zeros_like(x)
        source = dict(self.source)
        function = self.source["estimate_ftss"]
        old_d = function.__globals__["D"]
        try:
            function.__globals__["D"] = 2
            torch.manual_seed(42)
            result = function(Identity(), torch.device("cpu"), n_samples=2, n_steps=20)
        finally:
            function.__globals__["D"] = old_d
        np.testing.assert_allclose(result["mean_curve"], 1., atol=1e-6)
        self.assertEqual(result["test_times"][-1], 0.95)
        self.assertEqual(len(result["mean_curve"]), 20)


if __name__ == "__main__":
    unittest.main()
