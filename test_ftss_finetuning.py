from pathlib import Path
import unittest

import torch

from run_ftss_finetuning import load_ftss


class ZeroVelocity(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


class FTSSIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo = Path(__file__).resolve().parent.parent / "ftss-memorization"
        if not (repo / "finetune_memorization_ftss.py").is_file():
            raise unittest.SkipTest("The sibling ftss-memorization checkout is unavailable")
        cls.ftss = load_ftss(repo)

    def test_identity_flow_has_unit_gain(self):
        noise = torch.tensor([[0.2, -0.3], [0.4, 0.1]])
        directions = torch.tensor([[1., 0.], [0., 1.]])
        curve, minimum = self.ftss.ftss(ZeroVelocity(), noise, directions, 10, 5, 0.1)
        torch.testing.assert_close(torch.tensor(curve), torch.ones(5))
        self.assertAlmostEqual(minimum, 1., places=5)

    def test_generation_and_neighbor_distance_use_original_helpers(self):
        noise = torch.tensor([[0.2, -0.3], [0.4, 0.1]])
        samples = self.ftss.generate(ZeroVelocity(), noise, 10, 1)
        torch.testing.assert_close(samples, noise)
        distances = self.ftss.nearest_distances(samples, noise)
        self.assertEqual(distances.tolist(), [0., 0.])


if __name__ == "__main__":
    unittest.main()
