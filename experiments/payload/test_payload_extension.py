import unittest
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from run_payload_extension import orthogonal_covariance, score_prefixes, query_trial
from finalize_payload_paper import BEGIN, END, block_hash, finalize, render


class PayloadExtensionTest(unittest.TestCase):
    def test_orthogonal_second_moment_is_not_covariance(self):
        codes = np.eye(8)
        q = orthogonal_covariance(codes)
        np.testing.assert_allclose(codes.T @ codes / 8, np.eye(8) / 8)
        np.testing.assert_allclose(q, np.eye(8) / 8 - np.ones((8, 8)) / 64)
        self.assertEqual(np.linalg.matrix_rank(q), 7)
        self.assertAlmostEqual(np.trace(q), 7 / 8)

    def test_exact_and_bit_metrics_are_distinct(self):
        result = score_prefixes([[1, -1], [1, 1]])
        self.assertEqual(result["exact_recovery_percent"], 50.)
        self.assertEqual(result["bit_accuracy_percent"], 75.)
        self.assertEqual(result["bits_with_nonpositive_trial_mean"], 1)
        self.assertEqual(result["minimum_signed_bit_margin_mean"], 0.)

    def test_hypercube_is_zero_mean_unit_energy(self):
        import itertools
        codes = np.array(list(itertools.product((-1, 1), repeat=3))) / np.sqrt(3)
        np.testing.assert_allclose(np.linalg.norm(codes, axis=1), 1.)
        np.testing.assert_allclose(orthogonal_covariance(codes), np.eye(3) / 3, atol=1e-15)

    def test_zero_coordinate_follows_decoder_zero_bit_tie_rule(self):
        result = score_prefixes([[0., 0.]], truth=[0, 1])
        self.assertEqual(result["bit_accuracy_percent"], 50.)
        self.assertEqual(result["exact_recovery_percent"], 0.)

    def test_query_prefixes_are_accumulated_without_requerying(self):
        from types import SimpleNamespace
        class Constant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.count = 0
            def forward(self, x, t, y=None):
                self.count += len(x)
                return torch.ones_like(x)
        model = Constant()
        cfg = SimpleNamespace(class_cond=False)
        coordinates, _, _ = query_trial(model, cfg, torch.zeros(10, 1, 1, 2),
                                         torch.zeros(10, dtype=torch.long), torch.eye(2), torch.eye(2),
                                         [4, 8], 4, 42, torch.device("cpu"))
        self.assertEqual(model.count, 8)
        np.testing.assert_allclose(coordinates[:, 0], coordinates[:, 1])

    def test_query_stream_is_independent_of_forward_batch_size(self):
        from types import SimpleNamespace
        class Identity(torch.nn.Module):
            def forward(self, x, t, y=None):
                return x
        pool = torch.randn(10, 1, 1, 2)
        labels = torch.zeros(10, dtype=torch.long)
        args = (Identity(), SimpleNamespace(class_cond=False), pool, labels, torch.eye(2), torch.eye(2), [1024])
        a, _, _ = query_trial(*args, 512, 42, torch.device("cpu"))
        b, _, _ = query_trial(*args, 1024, 42, torch.device("cpu"))
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_finalizer_preserves_edits_made_while_job_runs(self):
        rows = [dict(bits=bits, queries=4096, messages=5, exact_recovery_percent_mean=50.,
                     exact_recovery_percent_std=1., bit_accuracy_percent_mean=99.,
                     bit_accuracy_percent_std=.1, fid_ratio_mean=1.05, fid_ratio_std=.01,
                     bits_with_nonpositive_trial_mean_mean=0.) for bits in (64, 96, 128)]
        summary = {"state": "complete", "groups": rows}
        self.assertIn("96 &", render(summary))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text(json.dumps(summary))
            paper = root / "main.tex"
            text = BEGIN + "new user edits" + END
            paper.write_text(text)
            finalize(root, paper, block_hash("old section"))
            self.assertEqual(paper.read_text(), text)
            self.assertEqual(json.loads((root / "paper_update.json").read_text())["state"], "needs_merge")


if __name__ == "__main__":
    unittest.main()
