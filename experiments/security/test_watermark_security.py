import copy
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import watermark_hf_flow_unet as flow
from run_watermark_security import (CarrierRemoved, decode, estimate_carrier, generate,
                                     merge_adapters, train_student, evaluate, verify_merge)


class SecurityStudyTests(unittest.TestCase):
    def test_materialized_linear_adapter_is_exact(self):
        model = nn.Sequential(flow.LoRALinear(nn.Linear(4, 3), 2, 2.))
        torch.nn.init.normal_(model[0].lora_B.weight)
        x = torch.randn(7, 4)
        materialized = merge_adapters(copy.deepcopy(model))
        torch.testing.assert_close(model(x), materialized(x))
        self.assertIsInstance(materialized[0], nn.Linear)
        self.assertNotIn("lora", " ".join(materialized.state_dict()))

    def test_materialized_convolution_is_exact(self):
        model = nn.Sequential(flow.LoRAConv2d(nn.Conv2d(3, 4, 3, padding=1), 2, 2.))
        torch.nn.init.normal_(model[0].lora_B.weight)
        x = torch.randn(7, 3, 8, 8)
        materialized = merge_adapters(copy.deepcopy(model))
        torch.testing.assert_close(model(x), materialized(x), atol=2e-6, rtol=2e-6)
        self.assertIsInstance(materialized[0], nn.Conv2d)

    def test_key_free_estimation_and_subtraction_of_known_carrier(self):
        direction = torch.randn(3, 4, 4)
        class PureCarrier(nn.Module):
            def forward(self, x, t, y=None):
                return torch.sin(2*math.pi*t)[:, None, None, None]*direction
        model = PureCarrier()
        pool = torch.randn(20, 3, 4, 4)
        labels = torch.arange(20) % 10
        estimated = estimate_carrier(model, pool, labels, 100, 16, 123)
        torch.testing.assert_close(estimated, direction)
        removed = CarrierRemoved(model, estimated, 1.)
        result = removed(pool, torch.rand(20), labels)
        torch.testing.assert_close(result, torch.zeros_like(result), atol=1e-6, rtol=0)

    def test_original_decoder_conventions_and_zero_ties(self):
        bits, _ = decode(torch.tensor([0., -1., 2.]), torch.eye(3), torch.eye(3), "hypercube", 3)
        self.assertEqual(bits, [0, 0, 1])
        signature = torch.tensor([0., 0., 0., 1.])
        bits, _ = decode(signature, torch.eye(4), torch.eye(4), "auto", 2)
        self.assertEqual(bits, [1, 1])

    def test_sample_only_training_never_queries_teacher(self):
        class Student(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(.1))
            def forward(self, x, t, y=None):
                return self.scale*x
        class NoVelocityAccess(nn.Module):
            def forward(self, *args, **kwargs):
                raise AssertionError("Sample-only training queried teacher velocity")
        args = SimpleNamespace(lr=1e-4, steps=2, seed=42, batch_size=4)
        pool = torch.randn(10, 3, 4, 4)
        labels = torch.arange(10)
        with tempfile.TemporaryDirectory() as directory:
            student = train_student(Student(), NoVelocityAccess(), pool, labels, "sample_only", args, Path(directory))
            saved = torch.load(Path(directory)/"sample_only.pt", weights_only=True)
            self.assertEqual(saved["step"], 2)
            resumed = train_student(Student(), NoVelocityAccess(), pool, labels, "sample_only", args, Path(directory))
            torch.testing.assert_close(student.scale, resumed.scale)

    def test_generation_has_paired_noise_and_balanced_labels(self):
        class Zero(nn.Module):
            def forward(self, x, t, y=None):
                return torch.zeros_like(x)
        a, labels = generate(Zero(), 20, 4, 7, 42, torch.device("cpu"))
        b, _ = generate(Zero(), 20, 4, 20, 42, torch.device("cpu"))
        torch.testing.assert_close(a, b)
        np.testing.assert_array_equal(np.bincount(labels.numpy()), np.full(10, 2))

    def test_clean_quality_cache_avoids_repeated_generation(self):
        class Zero(nn.Module):
            def forward(self, x, t, y=None):
                return torch.zeros_like(x)
        args = SimpleNamespace(trials=2, seed=42, queries=8, eval_batch=4)
        key = dict(P=torch.eye(3072)[:, :2], codes=torch.eye(2), message="0")
        pool = torch.zeros(10, 3, 32, 32)
        quality = dict(samples=pool, features=torch.zeros(10, 4), fid=7.)
        with tempfile.TemporaryDirectory() as directory:
            with patch("run_watermark_security.generate", side_effect=AssertionError("Repeated clean generation")):
                result = evaluate(Zero(), key, "auto", pool, torch.arange(10), torch.zeros(10, 4),
                                  args, Path(directory)/"clean", 7., 14., quality=quality)
        self.assertEqual(result["fid_ratio_clean"], 1.)
        self.assertEqual(result["fid_ratio_teacher"], .5)

    def test_decimal_attack_severities_have_independent_cache_files(self):
        class Counter(nn.Module):
            def __init__(self):
                super().__init__()
                self.queries = 0
            def forward(self, x, t, y=None):
                self.queries += len(x)
                return torch.zeros_like(x)
        model = Counter()
        args = SimpleNamespace(trials=1, seed=42, queries=8, eval_batch=4)
        key = dict(P=torch.eye(3072)[:, :2], codes=torch.eye(2), message="0")
        pool = torch.zeros(10, 3, 32, 32)
        quality = dict(samples=pool, features=torch.zeros(10, 4), fid=7.)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for severity in ("prune_0.25", "prune_0.5", "subtract_1", "subtract_1.5"):
                evaluate(model, key, "auto", pool, torch.arange(10), torch.zeros(10, 4),
                         args, root / severity, 7., 14., quality=quality)
                for extension in (".json", ".npz", ".png"):
                    self.assertTrue((root / (severity + extension)).exists())
            self.assertEqual(model.queries, 4 * args.queries)

    def test_merge_check_restores_backend_and_rejects_changed_flow(self):
        class Scaled(nn.Module):
            def __init__(self, scale):
                super().__init__()
                self.scale = scale
            def forward(self, x, t, y=None):
                return self.scale*x
        x = torch.ones(2, 3, 4, 4)
        t, y = torch.zeros(2), torch.zeros(2, dtype=torch.long)
        previous = torch.backends.cuda.matmul.allow_tf32
        verify_merge(Scaled(1.), Scaled(1.), x, t, y)
        with self.assertRaises(ValueError):
            verify_merge(Scaled(2.), Scaled(1.), x, t, y)
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, previous)

    def test_independent_reload_audit_decoder(self):
        from audit_watermark_students import decode as independent_decode
        key = dict(P=torch.eye(3), codes=torch.eye(3), message="101")
        result = independent_decode(torch.tensor([1., -1., 1.]), key, "hypercube")
        self.assertTrue(result["exact"])
        self.assertEqual(result["decision_margin"], 1.)
        key["message"] = "010"
        wrong_claim = independent_decode(torch.tensor([1., -1., 1.]), key, "hypercube")
        self.assertEqual(wrong_claim["decoded"], [1, 0, 1])
        self.assertFalse(wrong_claim["exact"])
        key = dict(P=torch.eye(4), codes=torch.eye(4), message="01")
        result = independent_decode(torch.tensor([0., 0., 1., 0.]), key, "auto")
        self.assertTrue(result["exact"])
        self.assertEqual(result["decision_margin"], 1.)


if __name__ == "__main__":
    unittest.main()
