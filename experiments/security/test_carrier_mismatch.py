import math
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from run_carrier_mismatch import CosineRemoved, estimate_cosine
from finalize_carrier_mismatch import publish, render_rows, table_span


class CarrierMismatchTests(unittest.TestCase):
    def completed_results(self):
        return dict(state="complete", rows=[dict(scheme=scheme, message=str(i), metrics=dict(
            recovery_percent=100., bit_accuracy_percent=100., fid_ratio_teacher=1.))
            for scheme in ("O5", "H32") for i in range(5)])

    def test_publication_requires_complete_distinct_messages(self):
        result = self.completed_results()
        self.assertIn("$100.0{\\pm}0.0$", render_rows(result)["O5"])
        result["state"] = "running"
        with self.assertRaises(ValueError):
            render_rows(result)
        result["state"] = "complete"
        result["rows"][1]["message"] = result["rows"][0]["message"]
        with self.assertRaises(ValueError):
            render_rows(result)

    def test_publication_preserves_other_text_and_guards_table(self):
        text = ("User introduction\n\\begin{table}\n\\label{tab:removal-transfer}\n"
                "O5 & Marked teacher\n& Sine-carrier subtraction & O5 data\n"
                "H32 & Marked teacher\n& Sine-carrier subtraction & H32 data\n\\end{table}\n"
                "encodings, whereas strengths one and 1.5 give near-chance bit accuracy.\n")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paper = root / "main.tex"
            paper.write_text(text)
            start, end = table_span(text)
            (root / "paper_update_plan.json").write_text(json.dumps(dict(
                table_sha256=hashlib.sha256(text[start:end].encode()).hexdigest())))
            (root / "results.json").write_text(json.dumps(self.completed_results()))
            paper.write_text(text.replace("O5 data", "Edited O5 data"))
            with self.assertRaises(ValueError):
                publish(root, paper)
            self.assertIn("Edited O5 data", paper.read_text())
            paper.write_text(text)
            publish(root, paper)
            updated = paper.read_text()
            self.assertTrue(updated.startswith("User introduction\n"))
            self.assertEqual(updated.count("Cosine-carrier subtraction"), 2)
            self.assertIn("& Sine-carrier subtraction & O5 data", updated)

    def test_cosine_component_removed(self):
        direction = torch.randn(3, 4, 4)
        class Cosine(nn.Module):
            def forward(self, x, t, y=None):
                return torch.cos(2 * math.pi * t)[:, None, None, None] * direction
        pool, labels = torch.randn(8, 3, 4, 4), torch.arange(8)
        estimated = estimate_cosine(Cosine(), pool, labels, 256, 32, 42)
        torch.testing.assert_close(estimated, direction)
        removed = CosineRemoved(Cosine(), estimated)(pool, torch.rand(8), labels)
        torch.testing.assert_close(removed, torch.zeros_like(removed), atol=1e-6, rtol=0)

    def test_wrong_carrier_does_not_cancel_sine(self):
        direction = torch.randn(3, 4, 4)
        class Sine(nn.Module):
            def forward(self, x, t, y=None):
                return torch.sin(2 * math.pi * t)[:, None, None, None] * direction
        pool, labels = torch.randn(256, 3, 4, 4), torch.arange(256)
        t = (torch.arange(256) + .5) / 256
        with patch("run_carrier_mismatch.security.query_batch",
                   return_value=(pool, t, labels, pool)):
            estimated = estimate_cosine(Sine(), pool, labels, 256, 256, 42)
        torch.testing.assert_close(estimated, torch.zeros_like(estimated), atol=1e-6, rtol=0)
        attacked = CosineRemoved(Sine(), estimated)(pool, t, labels)
        torch.testing.assert_close(attacked, Sine()(pool, t, labels), atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
