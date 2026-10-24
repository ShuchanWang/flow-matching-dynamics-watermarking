import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from run_endpoint_loop_distillation import NoVelocityTeacher, paired_metrics, summarize
from analyze_endpoint_loop_distillation import contrast_summary
import run_watermark_security as security


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, x, t, y=None):
        return self.scale * x


class SampleTransferTests(unittest.TestCase):
    def test_velocity_access_is_rejected(self):
        with self.assertRaises(RuntimeError):
            NoVelocityTeacher()(torch.zeros(1))

    def test_sample_training_never_calls_teacher_and_identical_data_match(self):
        torch.set_num_threads(2)
        pool = torch.ones(8, 1, 2, 2)
        labels = torch.zeros(8, dtype=torch.long)
        settings = SimpleNamespace(steps=2, batch_size=4, lr=.001, seed=891)
        models = []
        with tempfile.TemporaryDirectory() as directory:
            for name in ("clean", "identical_endpoints"):
                folder = Path(directory) / name
                folder.mkdir()
                torch.manual_seed(891)
                model = security.train_student(Tiny(), NoVelocityTeacher(), pool, labels,
                                               "sample_only", settings, folder)
                models.append(model.scale.detach().clone())
        self.assertNotEqual(float(models[0]), 0.)
        self.assertTrue(torch.equal(*models))

    def test_paired_metrics_do_not_pool_query_trials_as_message_bits(self):
        marked = dict(bit_accuracy_percent=100., trials=[dict(scores=[2., -1.])] * 20)
        control = dict(bit_accuracy_percent=50., trials=[dict(scores=[-1., -2.])] * 20)
        result = paired_metrics(marked, control, [1, 0])
        self.assertEqual(result["total_message_bits"], 2)
        self.assertEqual(result["consensus_correct_bits"], 2)
        self.assertEqual(result["bits_improved"], 1)
        self.assertEqual(result["bits_harmed"], 0)
        self.assertEqual(result["bit_accuracy_gain_pp"], 50.)

    def test_identical_student_responses_have_no_paired_transfer(self):
        metrics = dict(bit_accuracy_percent=50., trials=[dict(scores=[2., 1.])] * 20)
        result = paired_metrics(metrics, metrics, [1, 0])
        self.assertEqual(result["bit_accuracy_gain_pp"], 0.)
        self.assertEqual(result["bits_improved"], 0)
        self.assertEqual(result["bits_harmed"], 0)
        self.assertEqual(result["mean_signed_score_change"], 0.)

    def test_group_uncertainty_uses_student_seeds_not_query_repetitions(self):
        rows = [dict(role="clean", seed=i,
                     metrics=dict(bit_accuracy_percent=bits, recovery_percent=0., fid=100., trials=[{}] * 20))
                for i, bits in enumerate((10., 20., 30.))]
        results = dict(state="complete", rows=rows)
        with tempfile.TemporaryDirectory() as directory:
            summarize(Path(directory), results)
        group = results["groups"][0]
        self.assertEqual(group["student_seeds"], 3)
        self.assertEqual(group["bit_accuracy_percent_mean"], 20.)
        self.assertEqual(group["bit_accuracy_percent_std"], 10.)

    def test_contrast_is_labeled_separately_from_raw_prediction(self):
        marked = [dict(scores=[-1., 1.])] * 20
        control = [dict(scores=[-2., 2.])] * 20
        result = contrast_summary(marked, control, [1, 0])
        self.assertEqual(result["contrast_consensus_correct_bits"], 2)
        self.assertEqual(result["message_bits"], 2)
        self.assertEqual(result["query_trials"], 20)
        self.assertEqual(result["contrast_accuracy_percent"], 100.)

    def test_exactly_identical_scores_do_not_create_a_contrast_message(self):
        trials = [dict(scores=[-2., 2.])] * 20
        result = contrast_summary(trials, trials, [1, 0])
        self.assertEqual(result["score_change_l2"], 0.)
        self.assertIsNone(result["contrast_consensus_correct_bits"])
        self.assertIsNone(result["contrast_accuracy_percent"])

    def test_per_trial_contrast_and_pooled_consensus_are_separate(self):
        marked = [dict(scores=[-1., 1.]), dict(scores=[-4., 4.])]
        control = [dict(scores=[-2., 2.]), dict(scores=[-2., 2.])]
        result = contrast_summary(marked, control, [1, 0])
        self.assertEqual(result["contrast_accuracy_percent"], 50.)
        self.assertEqual(result["contrast_consensus_accuracy_percent"], 0.)
        self.assertEqual(result["nonzero_contrast_trials"], 2)


if __name__ == "__main__":
    unittest.main()
