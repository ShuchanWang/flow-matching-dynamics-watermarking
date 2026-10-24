import json
from pathlib import Path
import tempfile
import unittest

import torch

from run_baseline_screen import selected_rows, condition_spec, CONDITIONS


class BaselineScreenTests(unittest.TestCase):
    def test_selection_is_perfect_32bit_then_lowest_quality_cost(self):
        rows = []
        for method in ("dynamics", "weight", "trigger", "naive"):
            for trial, payload, recovery, ratio in [(0, 32, 95, 1.), (1, 32, 100, 1.06),
                                                    (2, 32, 100, 1.1), (3, 5, 100, .9)]:
                rows.append(dict(method=method, trial=trial, payload=payload,
                                 metrics=dict(recovery_percent=recovery, fid_ratio_control=ratio)))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "tune").mkdir()
            (path / "tune/results.json").write_text(json.dumps(dict(state="stopped_by_user", rows=rows)))
            selected = selected_rows(path)
            self.assertEqual(len(selected), 4)
            self.assertTrue(all(r["payload"] == 32 and r["trial"] == 1 for r in selected))

    def test_missing_perfect_model_is_not_silently_included(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "tune").mkdir()
            (path / "tune/results.json").write_text(json.dumps(dict(rows=[])))
            with self.assertRaises(ValueError):
                selected_rows(path)

    def test_condition_settings_are_explicit(self):
        self.assertEqual(condition_spec("prune25"), ("prune", .25))
        self.assertEqual(condition_spec("quant2"), ("quant", 2))
        self.assertEqual(condition_spec("ft_low"), ("full_finetune", 1e-5))
        self.assertEqual(condition_spec("ft_high"), ("full_finetune", 1e-4))
        self.assertEqual(len(CONDITIONS), 12)
        self.assertEqual(len(set(CONDITIONS)), 12)
        with self.assertRaises(ValueError):
            condition_spec("not-an-attack")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
