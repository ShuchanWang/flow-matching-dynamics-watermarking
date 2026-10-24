import copy
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from torch import nn

from run_embedding_strategy_robustness import (
    code_from_key, condition_plan, embedding_loss, prepare_model, summarize, whole_model_attack,
)
from run_watermark_security import query_batch


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(4, 4)
        self.time_embed = nn.Linear(1, 4)
        self.out = nn.Linear(4, 2)

    def forward(self, x, t):
        return self.out(torch.tanh(self.features(x) + self.time_embed(t)))


class EmbeddingStrategyTests(unittest.TestCase):
    def config(self):
        return dict(lora_rank=2, lora_alpha=2, lora_targets="linear", train_extra="time_embed,out.",
                    objective="full", wm_eps=1.5, wm_proj_weight=10., wm_lambda=1., wm_tanh_scale=2.)

    def test_both_arms_start_from_same_clean_function(self):
        base = Toy().requires_grad_(False)
        lora = prepare_model(base, self.config(), "lora")
        full = prepare_model(base, self.config(), "full")
        x, t = torch.randn(7, 4), torch.rand(7, 1)
        torch.testing.assert_close(lora(x, t), base(x, t), atol=0, rtol=0)
        torch.testing.assert_close(full(x, t), base(x, t), atol=0, rtol=0)
        self.assertTrue(all(p.requires_grad for p in full.parameters()))
        self.assertFalse(lora.features.layer.weight.requires_grad)
        self.assertTrue(lora.time_embed.layer.weight.requires_grad)
        self.assertTrue(lora.out.layer.weight.requires_grad)
        self.assertFalse(any(p.requires_grad for p in base.parameters()))

    def test_shared_loss_matches_original_objective(self):
        c = self.config()
        prediction = torch.randn(5, 1, 1, 4, requires_grad=True)
        clean, target = torch.randn_like(prediction), torch.randn_like(prediction)
        times = torch.rand(5)
        projection, code = torch.eye(4), torch.tensor([.5, -.5, .5, -.5])
        loss, _ = embedding_loss(prediction, clean, target, times, projection, code, c)
        a = torch.sin(2 * math.pi * times)[:, None]
        watermark = (1.5 * a * code[None]).reshape_as(prediction)
        coordinates = (prediction - clean).flatten(1)
        expected = ((prediction - target - watermark).square().mean()
                    + 10 * (coordinates - 1.5 * a * code[None]).square().mean()
                    - torch.tanh((a * (coordinates * code[None]).sum(1, keepdim=True)).mean()
                                 / (.75 + 1e-8) * 2))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_legacy_orthogonal_key_without_mode_and_hypercube_key(self):
        key = dict(P=torch.eye(4), codes=torch.eye(4), message="10")
        _, code = code_from_key(key, {}, torch.device("cpu"))
        torch.testing.assert_close(code, torch.eye(4)[1])
        key["codes"] = torch.eye(4)[:2]
        _, code = code_from_key(key, dict(codebook_mode="hypercube"), torch.device("cpu"))
        torch.testing.assert_close(code, torch.tensor([1., -1., 0., 0.]) / math.sqrt(2))

    def test_compression_modifies_backbone_not_just_embedding_subset(self):
        model = Toy()
        before = model.features.weight.detach().clone()
        audit = whole_model_attack(model, "quant", 2)
        self.assertFalse(torch.equal(before, model.features.weight))
        self.assertEqual(audit["parameters"], sum(p.numel() for p in model.parameters()))
        self.assertEqual(audit["tensors"], len(list(model.parameters())))

    def test_unmerged_adapters_rejected(self):
        with self.assertRaises(ValueError):
            whole_model_attack(prepare_model(Toy(), self.config(), "lora"), "prune", .25)

    def test_independent_conditions_do_not_change_teacher(self):
        teacher = Toy()
        original = copy.deepcopy(teacher.state_dict())
        whole_model_attack(copy.deepcopy(teacher), "prune", .5)
        whole_model_attack(copy.deepcopy(teacher), "quant", 2)
        for name, value in teacher.state_dict().items():
            torch.testing.assert_close(value, original[name], atol=0, rtol=0)

    def test_training_draws_can_be_paired_independently_of_model_rng(self):
        pool, labels = torch.randn(20, 1, 2, 2), torch.arange(20)
        a, b = torch.Generator().manual_seed(42), torch.Generator().manual_seed(42)
        for _ in range(3):
            first = query_batch(pool, labels, 4, a)
            torch.randn(123)
            second = query_batch(pool, labels, 4, b)
            for x, y in zip(first, second):
                torch.testing.assert_close(x, y, atol=0, rtol=0)

    def test_complete_attack_plan_contains_expected_conditions(self):
        args = SimpleNamespace(attacks=["compression", "finetune", "removal"], prune=[.25, .5],
                               bits=[8, 4, 2], ft_steps=1000, ft_lrs=[1e-5, 1e-4])
        plan = condition_plan(args)
        self.assertEqual(len(plan), 10)
        self.assertEqual(len({r[0] for r in plan}), len(plan))
        self.assertIn(("subtract_cosine", "cosine", 1), plan)
        self.assertIn(("quant_2bit", "quant", 2), plan)

    def test_summary_reports_baseline_failures_instead_of_filtering_them(self):
        row = dict(scheme="H32", arm="full", condition="marked", message="0" * 32,
                   metrics=dict(recovery_percent=0., bit_accuracy_percent=50.,
                                fid_ratio_clean=2., fid_ratio_teacher=1.))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summarize(root, dict(state="running", rows=[row], baselines=[]))
            report = (root / "report.md").read_text()
            self.assertIn("H32 | full | marked | 1 | 0.000", report)
            self.assertIn("A common training budget is not evidence", report)


if __name__ == "__main__":
    unittest.main()
