from repo_layout import ROOT, source_path
import ast
import csv
import json
import math
import os
import random
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import numpy as np
from types import SimpleNamespace

from collect_results import hf_rows, infer_group
from prepare_audio_latents import audio_records, stable_audio_clip_samples
from run_main_architectures import build_jobs, MESSAGES_32, compatible_legacy_result, sd_loss_matches, mlp_loss_matches
from watermark_audio_flow import AudioFlow, StableVelocity, clap_embedding, stable_conditioning, checkpoint_audit, load_adapter_checkpoint, estimate_clean_bias
from watermark_codebooks import make_key, message_code, decode_signature, target_score, hypercube_margin_loss
from diagnose_sd35_hypercube import summarize
from repair_mlp_watermarks import score_metrics, signatures, decoding_margin_loss
from run_sd35_repair_suite import commands as sd35_repair_commands, stage_complete
from run_fma_audio_study import cache_complete as audio_cache_complete, smoke_commands
from run_stable_audio_smoke import commands as stable_smoke_commands


class CodebookTests(unittest.TestCase):
    def test_metric_audit_recognizes_h32_preparation_without_scheme_directory(self):
        from audit_saved_metrics import extract
        message = MESSAGES_32[0]
        values = [2 * int(b) - 1 for b in message]
        payload = {"step": 300, "queries": 512, "trials": [
            {"wm": {"exact": True, "bit_accuracy": 1, "scores": values},
             "clean": {"exact": False, "bit_accuracy": 0, "scores": [-v for v in values]}}]}
        path = Path("preparation/fma_stable_prep") / message / "recovery_step_300.json"
        row = extract(path, payload)[0]
        self.assertEqual(row["codebook_mode"], "hypercube")
        self.assertEqual(row["roc_auc"], 1)

    def test_margin_suite_matches_pilot_stages_and_reuses_saved_300(self):
        from unittest.mock import patch
        from run_stable_audio_margin_suite import commands
        def exists(path):
            return "fma_stable_more_steps" in str(path) and MESSAGES_32[1] in str(path)
        with patch("run_stable_audio_margin_suite.Path.exists", exists):
            reused = commands(MESSAGES_32[1])
            fresh = commands(MESSAGES_32[2])
        self.assertEqual(len(reused), 1)
        self.assertEqual(len(fresh), 2)
        prep = fresh[0][1]
        self.assertIn("--training_only", prep)
        self.assertNotIn("--compensate_clean_bias", prep)
        self.assertEqual(prep[prep.index("--steps") + 1], "200")
        for jobs in (reused, fresh):
            repair = jobs[-1][1]
            self.assertIn("--compensate_clean_bias", repair)
            self.assertEqual(repair[repair.index("--steps") + 1], "100")
            self.assertEqual(repair[repair.index("--n_detect_trials") + 1], "20")

    def test_clean_bias_uses_carrier_and_preserves_rng(self):
        class CarrierFlow(torch.nn.Module):
            def forward(self, x, t):
                return torch.sin(2 * math.pi * t).view(-1, 1, 1) * torch.ones_like(x)
        P = torch.eye(64)[:, :32]
        pool = torch.randn(4, 8, 8)
        state = torch.get_rng_state().clone()
        bias = estimate_clean_bias(CarrierFlow(), pool, P, 2048, 32, torch.device("cpu"), 42)
        torch.testing.assert_close(torch.get_rng_state(), state)
        torch.testing.assert_close(bias, torch.full((32,), 0.5), atol=0.03, rtol=0)

    def test_margin_pilot_preserves_rank_strength_and_raw_decoder(self):
        from run_stable_audio_margin_pilot import command
        cmd = command()
        self.assertIn("--compensate_clean_bias", cmd)
        self.assertEqual(cmd[cmd.index("--wm_bit_weight") + 1], "1")
        self.assertIn("adapter_step_300.pt", cmd[cmd.index("--continue_adapter") + 1])
        self.assertNotIn("--wm_eps", cmd)
        self.assertNotIn("--lora_rank", cmd)

    def test_more_steps_reuses_adapters_and_skips_successful_message(self):
        from unittest.mock import patch
        from run_stable_audio_more_steps import commands
        saved = [json.dumps({"metrics": {"wm_acc": value}}) for value in (55, 45, 85, 100, 95)]
        with patch("run_stable_audio_more_steps.Path.read_text", side_effect=saved):
            jobs = list(commands())
        self.assertEqual(len(jobs), 4)
        for message, command in jobs:
            self.assertIn("--continue_adapter", command)
            self.assertEqual(command[command.index("--steps") + 1], "200")
            self.assertNotIn("--wm_eps", command)
            self.assertNotIn("--wm_proj_weight", command)

    def test_empirical_auc_handles_ties(self):
        from audit_saved_metrics import empirical_auc
        self.assertEqual(empirical_auc([2, 3], [0, 1]), 1)
        self.assertEqual(empirical_auc([0, 1], [2, 3]), 0)
        self.assertEqual(empirical_auc([1], [1]), 0.5)
        self.assertIsNone(empirical_auc([], [1]))

    def test_saved_metrics_do_not_confuse_recovery_with_ownership(self):
        from audit_saved_metrics import metric_row
        row = metric_row(Path("results.json"), {"wm_message": "10101"},
                         {"wm_acc": 80, "clean_fp": 5})
        self.assertEqual(row["recovery_failure_pct"], 20)
        self.assertIsNone(row["ber_pct"])
        self.assertEqual(row["ber_lower_bound_pct"], 4)
        self.assertEqual(row["ber_upper_bound_pct"], 20)
        self.assertEqual(row["clean_target_match_pct"], 5)
        self.assertIsNone(row["ownership_fpr_pct"])
        self.assertIsNone(row["ownership_fnr_pct"])
        exact = metric_row(Path("results.json"), {"wm_message": "10101"}, {"wm_acc": 100})
        self.assertEqual(exact["ber_pct"], 0)

    def test_saved_mlp_gate_supports_empirical_ownership_rates(self):
        from audit_saved_metrics import mlp_repair_rows
        payload = {"config": {"trials": 20}, "gate_caveat": "One clean model",
                   "per_message": [{"message": "10101", "after": {"raw_exact_recovery": 100,
                        "bit_accuracy": 100, "ownership_acceptance": 90, "threshold": 0.2},
                        "clean": {"raw_exact_recovery": 5, "ownership_acceptance": 0}}]}
        row = mlp_repair_rows(Path("orthogonal5/repair_results.json"), payload)[0]
        self.assertEqual(row["ownership_fpr_pct"], 0)
        self.assertEqual(row["ownership_fnr_pct"], 10)
        self.assertEqual(row["ownership_tpr_pct"], 90)
        self.assertEqual(row["clean_target_match_pct"], 5)
        self.assertEqual(row["ownership_caveat"], "One clean model")

    def test_stable_validation_reuses_original_checkpoint_and_full_recovery_budget(self):
        from run_stable_audio_validation import commands
        jobs = list(commands())
        self.assertEqual(len(jobs), 10)
        self.assertIn("--evaluate_adapter", jobs[0][1])
        for name, command in jobs:
            self.assertEqual(command[command.index("--n_queries") + 1], "4096")
            self.assertEqual(command[command.index("--n_detect_trials") + 1], "20")
            self.assertEqual(command[command.index("--steps") + 1], "100")

    def test_adapter_loader_restores_parameters_and_rejects_invalid_names(self):
        model = AudioFlow((8, 8), "mlp", 8, 1)
        values = {name: value.detach().clone() for name, value in model.named_parameters()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            torch.save({"step": 100, "trainable_parameters": values}, path)
            with torch.no_grad():
                for value in model.parameters():
                    value.zero_()
            self.assertEqual(load_adapter_checkpoint(model, path), 100)
            for name, value in model.named_parameters():
                torch.testing.assert_close(value, values[name])
            torch.save({"step": 100, "trainable_parameters": {}}, path)
            with self.assertRaises(ValueError):
                load_adapter_checkpoint(model, path)

    def test_checkpoint_audit_preserves_rng_and_training_mode(self):
        model = AudioFlow((8, 8), "mlp", 8, 1).train()
        base = AudioFlow((8, 8), "mlp", 8, 1).eval()
        P, codes = make_key(64, 5, 32, torch.device("cpu"), "orthogonal")
        pool = torch.randn(4, 8, 8)
        args = SimpleNamespace(wm_message="10101", wm_K=32, codebook_mode="orthogonal",
                               checkpoint_queries=4, checkpoint_trials=2, eval_batch_size=2, seed=42)
        state = torch.get_rng_state().clone()
        python_state, numpy_state = random.getstate(), np.random.get_state()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_audit(model, base, pool, P, codes, args, torch.device("cpu"), 100, Path(directory))
            self.assertTrue((Path(directory) / "adapter_step_100.pt").exists())
        self.assertTrue(model.training)
        torch.testing.assert_close(torch.get_rng_state(), state)
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])

    def test_stable_pilot_is_bounded_and_separate_from_smoke(self):
        from run_stable_audio_pilot import commands
        jobs = list(commands())
        self.assertEqual(len(jobs), 3)
        for name, command in jobs[1:]:
            self.assertEqual(command[command.index("--steps") + 1], "500")
            self.assertEqual(command[command.index("--checkpoint_every") + 1], "100")
            self.assertIn("stable_audio_pilot", command[command.index("--output_dir") + 1])

    def test_stable_smoke_uses_short_clips_both_payloads_and_small_batches(self):
        jobs = stable_smoke_commands()
        self.assertEqual(len(jobs), 3)
        preparation = jobs[0][1]
        self.assertEqual(preparation[preparation.index("--seconds") + 1], "6")
        self.assertEqual(preparation[preparation.index("--stable-model") + 1],
                         "stabilityai/stable-audio-3-small-music-base")
        for name, command in jobs[1:]:
            self.assertEqual(command[command.index("--batch_size") + 1], "1")
            self.assertEqual(command[command.index("--steps") + 1], "5")
            self.assertNotIn("--skip_quality", command)
            self.assertEqual(len(command[command.index("--wm_message") + 1]),
                             32 if name == "hypercube32" else 5)

    def test_stable_clip_padding_preserves_requested_duration(self):
        requested, padded = stable_audio_clip_samples(6, 44100, 4096)
        self.assertEqual(requested, 264600)
        self.assertEqual(padded, 266240)
        self.assertEqual(padded % 4096, 0)
        self.assertLess(padded - requested, 4096)
        with self.assertRaises(ValueError):
            stable_audio_clip_samples(0, 44100, 4096)

    def test_stable_inpainting_conditioning_is_unconditional_generation(self):
        class Codec:
            local_add_cond_ids = ["inpaint_mask", "inpaint_masked_input"]
            def conditioner(self, metadata, device):
                self.metadata = metadata
                return {"prompt": [torch.ones(1, 2, 3)]}
            def get_conditioning_inputs(self, conditioning):
                return {"local_add_cond": torch.cat([conditioning[k][0]
                        for k in self.local_add_cond_ids], dim=1)}
        codec = Codec()
        conditioning = stable_conditioning(codec, (256, 65), "Piano music", 6, "cpu")
        self.assertEqual(tuple(conditioning["local_add_cond"].shape), (1, 257, 65))
        self.assertEqual(float(conditioning["local_add_cond"].abs().sum()), 0)
        self.assertEqual(codec.metadata[0]["seconds_total"], 6)

    def test_clap_features_support_tensor_and_structured_outputs(self):
        features = torch.tensor([[3.0, 4.0]])
        expected = torch.tensor([[0.6, 0.8]])
        self.assertTrue(torch.allclose(clap_embedding(features), expected))
        self.assertTrue(torch.allclose(clap_embedding(SimpleNamespace(pooler_output=features)), expected))
        with self.assertRaises(ValueError):
            clap_embedding(torch.tensor([[float("nan")]]))

    def test_fma_smoke_checks_quality_for_both_architectures_and_encodings(self):
        jobs = smoke_commands(Path("pool.pt"))
        self.assertEqual(len(jobs), 4)
        settings = set()
        for command in jobs:
            self.assertNotIn("--skip_quality", command)
            self.assertEqual(command[command.index("--n_fid_samples") + 1], "2")
            settings.add((command[command.index("--architecture") + 1],
                          command[command.index("--codebook_mode") + 1]))
        self.assertEqual(settings, {(a, c) for a in ("mlp", "transformer")
                                   for c in ("orthogonal", "hypercube")})

    def test_audio_cache_rejects_incomplete_nonfinite_or_leaked_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.pt"
            data = {"train": torch.zeros(2, 4, 4), "test": torch.zeros(2, 4, 4),
                    "test_audio": torch.zeros(2, 20),
                    "sources": {"train": ["a", "b"], "test": ["c", "d"]},
                    "config": {"codec": "music2latent", "dataset": "fma", "seconds": 6.0,
                               "root": "data/audio/fma_small",
                               "metadata": "data/audio/fma_metadata/tracks.csv"}}
            torch.save(data, path)
            self.assertTrue(audio_cache_complete(path, 2, 2))
            self.assertFalse(audio_cache_complete(path, 2000, 500))
            data["sources"]["test"][0] = "a"
            torch.save(data, path)
            self.assertFalse(audio_cache_complete(path, 2, 2))
            data["sources"]["test"][0] = "c"
            data["train"][0, 0, 0] = float("nan")
            torch.save(data, path)
            self.assertFalse(audio_cache_complete(path, 2, 2))

    def test_sd35_repair_stage_does_not_skip_unfinished_quality(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root)
            command = sd35_repair_commands(Path("original"), out, Path("calibration.json"))[-1]
            payload = {"source": "original", "input_adapter": str(out / "compensated"),
                       "queries": 256, "trials": 20, "seed": 20261014,
                       "repair": {"repair_steps": 50, "repair_lr": 5e-5, "repair_proj_weight": 10,
                                  "repair_bit_weight": 1, "repair_batch_size": 2, "repair_grad_accum": 2,
                                  "repair_epsilon": 3, "repair_clean_calibration": "calibration.json"},
                       "quality_evaluated": False}
            (out / "final").mkdir()
            (out / "final" / "adapter_config.json").write_text("{}")
            (out / "final.json").write_text(json.dumps(payload))
            self.assertFalse(stage_complete(command))
            payload.update(quality_evaluated=True, quality={"n_samples": 30, "fid_ratio": 1.03})
            (out / "final.json").write_text(json.dumps(payload))
            self.assertTrue(stage_complete(command))
            payload["input_adapter"] = "wrong_adapter"
            (out / "final.json").write_text(json.dumps(payload))
            self.assertFalse(stage_complete(command))

    def test_sd35_suite_preserves_sources_and_uses_a_common_200_update_budget(self):
        jobs = sd35_repair_commands(Path("original"), Path("repairs"), Path("calibration.json"))
        self.assertEqual(sum(int(c[c.index("--repair-steps") + 1]) for c in jobs), 200)
        for c in jobs:
            self.assertEqual(c[c.index("--run-dir") + 1], "original")
            self.assertTrue(c[c.index("--output") + 1].startswith("repairs/"))
            self.assertEqual(int(c[c.index("--repair-batch-size") + 1]) * int(c[c.index("--repair-grad-accum") + 1]), 4)
        self.assertEqual(jobs[-1][jobs[-1].index("--trials") + 1], "20")
        self.assertEqual(jobs[-1][jobs[-1].index("--quality-samples") + 1], "30")

    def test_retuned_mlp_loss_is_not_reused_as_original_objective(self):
        command = ["python", str(source_path('flow_watermark_mnist_mlp.py'))]
        self.assertTrue(mlp_loss_matches(command, {}))
        self.assertFalse(mlp_loss_matches(command, {"wm_query_bit_weight": 10}))
        command += ["--wm_query_bit_weight", "10"]
        self.assertTrue(mlp_loss_matches(command, {"wm_query_bit_weight": 10, "wm_bit_margin": 0.15}))
        self.assertFalse(mlp_loss_matches(command, {"wm_query_bit_weight": 10, "wm_bit_margin": 0.3}))

    def test_orthogonal_margin_gradient_improves_target_over_rival(self):
        scores = torch.tensor([0.1, 0.3, -0.1, -0.2], requires_grad=True)
        loss = decoding_margin_loss(scores, torch.eye(4), "00", "orthogonal", 0.15)
        loss.backward()
        self.assertLess(float(scores.grad[0]), 0)
        self.assertGreater(float(scores.grad[1]), 0)
        self.assertEqual(float(scores.grad[2:].abs().sum()), 0)

    def test_ownership_gate_preserves_raw_matching_and_rejects_weak_signatures(self):
        codes = torch.eye(5)
        weak = torch.full((4, 5), 0.01)
        result = score_metrics(weak, codes, "11111", "hypercube", 0.1)
        self.assertEqual(result["raw_exact_recovery"], 100)
        self.assertEqual(result["ownership_acceptance"], 0)
        strong = weak * 100
        self.assertEqual(score_metrics(strong, codes, "11111", "hypercube", 0.1)["ownership_acceptance"], 100)
        self.assertEqual(score_metrics(strong, codes, "01111", "hypercube", 0.1)["ownership_acceptance"], 0)

    def test_mlp_signature_queries_are_seeded_and_use_original_noise_distribution(self):
        class CarrierModel(torch.nn.Module):
            def forward(self, x, t):
                return torch.sin(2 * math.pi * t).expand_as(x)
        p = torch.eye(784)[:, :5]
        a = signatures(CarrierModel(), p, 4096, 2, 7)
        b = signatures(CarrierModel(), p, 4096, 2, 7)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(a, torch.full_like(a, 0.5), atol=0.02, rtol=0)

    def test_hypercube_bit_loss_penalizes_a_weak_bit_despite_strong_total_score(self):
        axes = torch.eye(32)
        carrier = torch.tensor([1.0, -1.0])
        velocity = carrier[:, None] * torch.ones(2, 32)
        self.assertEqual(float(hypercube_margin_loss(velocity, axes, "1" * 32, carrier, 1.5)), 0)
        velocity[:, 0] *= -1
        velocity.requires_grad_(True)
        loss = hypercube_margin_loss(velocity, axes, "1" * 32, carrier, 1.5)
        self.assertGreater(float(loss), 0)
        loss.backward()
        self.assertLess(float(velocity.grad[0, 0]), 0)
        self.assertGreater(float(velocity.grad[1, 0]), 0)
        self.assertEqual(float(velocity.grad[:, 1:].abs().sum()), 0)

    def test_orthogonal_all_messages(self):
        P, codes = make_key(64, 5, 32, "cpu", "orthogonal")
        torch.testing.assert_close(P.T @ P, torch.eye(32), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(codes @ codes.T, torch.eye(32), atol=1e-5, rtol=1e-5)
        for index in range(32):
            bits = tuple((index >> i) & 1 for i in range(5))
            self.assertEqual(decode_signature(message_code(bits, codes, "orthogonal"), codes, 5, "orthogonal")[0], bits)

    def test_implicit_32bit_storage_and_recovery(self):
        _, axes = make_key(64, 32, 32, "cpu", "hypercube")
        self.assertEqual(axes.numel(), 1024)
        for message in MESSAGES_32:
            bits = tuple(map(int, message))
            code = message_code(bits, axes, "hypercube")
            torch.testing.assert_close(code.norm(), torch.tensor(1.0))
            self.assertEqual(decode_signature(code, axes, 32, "hypercube")[0], bits)

    def test_rejects_impossible_explicit_payload(self):
        with self.assertRaises(ValueError):
            make_key(64, 32, 32, "cpu", "orthogonal")
        with self.assertRaises(ValueError):
            make_key(64, 32, 32, "cpu", "auto")

    def test_sd_legacy_fivebit_key_unchanged(self):
        torch.manual_seed(12345)
        expected_P = torch.linalg.qr(torch.randn(64, 32))[0]
        expected_codes = torch.linalg.qr(torch.randn(32, 32).T)[0].T
        expected_codes /= expected_codes.norm(dim=1, keepdim=True)
        torch.manual_seed(12345)
        P, codes = make_key(64, 5, 32, "cpu", "auto")
        torch.testing.assert_close(P, expected_P)
        torch.testing.assert_close(codes, expected_codes)


class RunnerTests(unittest.TestCase):
    def test_sd_main_grid_does_not_reuse_changed_training_objectives(self):
        command = ["python", str(source_path('watermark_sd35.py'))]
        self.assertTrue(sd_loss_matches(command, {}))
        self.assertFalse(sd_loss_matches(command, {"wm_bit_weight": 20}))
        self.assertFalse(sd_loss_matches(command, {"wm_proj_weight": 10}))
        self.assertTrue(sd_loss_matches(command + ["--wm_proj_weight", "10"], {"wm_proj_weight": 10}))

    def test_sd_bit_diagnostics_distinguish_recovery_from_aggregate_score(self):
        codes = torch.eye(32)
        message = "1" * 32
        signatures = torch.ones(2, 32)
        signatures[:, 0] = -0.1
        result = summarize(signatures, codes, message, "hypercube")
        self.assertEqual(result["exact_recovery"], 0)
        self.assertEqual(result["bit_accuracy"], 100 * 31 / 32)
        self.assertEqual(result["wrong_bit_frequencies"][0], 1)
        self.assertGreater(signatures.mean(), 0)

    def test_legacy_reuse_checks_protocol_and_normalizes_numeric_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"wm_message": "10101", "wm_K": 32, "steps": 1000,
                      "lora_alpha": 16.0, "n_detect_trials": 40, "n_queries": 4096}
            (root / "results.json").write_text(json.dumps({"config": config,
                                                          "metrics": {"wm_acc": 100, "fid_ratio": 1.02}}))
            command = ["python", str(source_path('watermark_hf_flow_unet.py')), "--wm_message", "10101",
                       "--wm_K", "32", "--steps", "1000", "--lora_alpha", "16",
                       "--n_detect_trials", "20", "--n_queries", "4096", "--codebook_mode", "orthogonal"]
            self.assertTrue(compatible_legacy_result(command, root))
            command[-1] = "hypercube"
            self.assertFalse(compatible_legacy_result(command, root))
            command[-1] = "orthogonal"
            command[command.index("--steps") + 1] = "500"
            self.assertFalse(compatible_legacy_result(command, root))
            command[command.index("--steps") + 1] = "1000"
            command[command.index("--n_detect_trials") + 1] = "80"
            self.assertFalse(compatible_legacy_result(command, root))

    def test_image_mlp_grid_payloads(self):
        jobs = build_jobs(Path("runs"), {"mlp"}, ["orthogonal5", "hypercube32", "hypercube5"])
        self.assertEqual(len(jobs), 3)
        for job in jobs:
            bits = int(job.command[job.command.index("--wm_bits") + 1])
            messages = job.command[job.command.index("--wm_messages") + 1].split(",")
            self.assertEqual(len(messages), 5)
            self.assertTrue(all(len(message) == bits for message in messages))
            self.assertEqual(bits, 32 if "hypercube32" in job.name else 5)

    def test_image_mlp_detection_and_implicit_competitor_score(self):
        # Extract helpers because the legacy script starts training at import time.
        tree = ast.parse(source_path('flow_watermark_mnist_mlp.py').read_text())
        helpers = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                                  and node.name in {"message_to_str", "decode_watermark", "compute_signature_stats"}],
                             type_ignores=[])
        for mode, bits in (("orthogonal", 5), ("hypercube", 5), ("hypercube", 32)):
            namespace = {"torch": torch, "np": np, "math": math, "D": 64, "device": "cpu",
                         "N_BITS": bits, "N_QUERIES": 8, "args": SimpleNamespace(codebook_mode=mode),
                         "decode_signature": decode_signature, "target_score": target_score}
            exec(compile(helpers, str(source_path('flow_watermark_mnist_mlp.py')), "exec"), namespace)
            P, codes = make_key(64, bits, 32, "cpu", mode)
            message = tuple([1] * bits)
            code = message_code(message, codes, mode)
            def model(x, t):
                return torch.sin(2 * math.pi * t) * (code @ P.T)
            self.assertEqual(namespace["decode_watermark"](model, P, codes)[0], message)
            stats = namespace["compute_signature_stats"](model, P, codes, message, n_trials=2)
            self.assertGreater(stats["margin"], 0)
            if mode == "hypercube":
                self.assertAlmostEqual(stats["other_mean"] / stats["true_mean"], 1 - 2 / bits, places=5)

    def test_grid_keeps_main_and_encoding_settings_separate(self):
        jobs = build_jobs(Path("runs"), {"unet", "sd35", "audio"},
                          ["orthogonal5", "hypercube32", "hypercube5"], {"maestro": "latents.pt"})
        self.assertEqual(len(jobs), 90)
        for job in jobs:
            mode = job.command[job.command.index("--codebook_mode") + 1]
            message = job.command[job.command.index("--wm_message") + 1]
            self.assertEqual(len(message), 32 if "hypercube32" in job.name else 5)
            self.assertEqual(mode, "orthogonal" if "orthogonal5" in job.name else "hypercube")

    def test_stable_clock_and_velocity_conversion(self):
        class Backbone(torch.nn.Module):
            def forward(self, x, t, **kwargs):
                self.t = t
                self.condition_batch = len(kwargs["global_cond"])
                return torch.ones_like(x)
        backbone = Backbone()
        model = StableVelocity(backbone, {"global_cond": torch.zeros(1, 8)})
        result = model(torch.zeros(2, 4, 8), torch.tensor([0.2, 0.4]))
        torch.testing.assert_close(result, -torch.ones_like(result))
        torch.testing.assert_close(backbone.t, torch.tensor([0.8, 0.6]))
        self.assertEqual(backbone.condition_batch, 2)

    def test_audio_architectures_backward(self):
        for architecture in ("mlp", "transformer"):
            model = AudioFlow((8, 8), architecture, 8, 1)
            x = torch.randn(2, 8, 8)
            prediction = model(x, torch.rand(2))
            self.assertEqual(prediction.shape, x.shape)
            prediction.square().mean().backward()
            self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_maestro_split_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata.csv"
            with metadata.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["audio_filename", "split"])
                writer.writeheader()
                writer.writerow({"audio_filename": "2018/example.wav", "split": "test"})
            self.assertEqual(list(audio_records("maestro", root, metadata)), [(root / "2018/example.wav", "test")])

    def test_collector_does_not_pool_codebooks_or_architectures(self):
        first = {"architecture": "mlp", "dataset": "maestro", "bits": 5, "wm_K": 32, "codebook_mode": "orthogonal"}
        second = first | {"codebook_mode": "hypercube"}
        self.assertNotEqual(infer_group("main_first", first), infer_group("main_second", second))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "results.json").write_text(json.dumps({"config": first | {"model_family": "audio_flow", "wm_message": "10101"},
                                                          "metrics": {"fd_clap_ratio": 1.02}}))
            row = hf_rows(path)[0]
            self.assertEqual(row["model_family"], "audio_flow")
            self.assertIsNone(row.get("fid_real_wm"))

    def test_end_to_end_audio_smoke_both_encodings_and_architectures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latent_path = root / "pool.pt"
            torch.save({"train": torch.randn(8, 8, 8), "test": torch.randn(4, 8, 8),
                        "sources": {"train": ["train"], "test": ["test"]}}, latent_path)
            for architecture in ("mlp", "transformer"):
                for mode, message in (("orthogonal", "10101"), ("hypercube", MESSAGES_32[0])):
                    output = root / architecture / mode
                    command = [sys.executable, str(source_path('watermark_audio_flow.py')), "--latents", str(latent_path),
                               "--dataset", "synthetic", "--architecture", architecture, "--codebook_mode", mode,
                               "--wm_message", message, "--output_dir", str(output), "--base_dir", str(root / "bases"),
                               "--base_steps", "2", "--steps", "2", "--width", "8", "--depth", "1",
                               "--batch_size", "2", "--eval_batch_size", "2", "--n_queries", "4",
                               "--n_detect_trials", "2", "--n_fid_samples", "2", "--device", "cpu", "--skip_quality"]
                    command.extend(["--checkpoint_every", "2", "--checkpoint_queries", "4", "--checkpoint_trials", "1"])
                    completed = subprocess.run(command, capture_output=True, text=True,
                                               env=os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    result = json.loads((output / "synthetic" / message / "results.json").read_text())
                    self.assertFalse(result["metrics"]["quality_complete"])
                    self.assertFalse(result["metrics"]["ownership_evaluated"])
                    self.assertEqual(result["metrics"]["clean_target_match"], result["metrics"]["clean_fp"])
                    self.assertEqual(result["metrics"]["bits"], len(message))
                    key = torch.load(output / "synthetic" / message / "watermark_key.pt", weights_only=False)
                    self.assertEqual(tuple(key["codes"].shape), (32, 32))
                    self.assertEqual(result["metrics"]["ber_pct"], 100 - result["metrics"]["bit_accuracy"])
                    if architecture == "mlp":
                        continued = root / "continued"
                        followup = command.copy()
                        followup[followup.index("--output_dir") + 1] = str(continued)
                        followup[followup.index("--steps") + 1] = "1"
                        followup.extend(["--continue_adapter", str(output / "synthetic" / message / "adapter_step_2.pt")])
                        if mode == "hypercube":
                            followup[followup.index("--output_dir") + 1] = str(continued / mode)
                            followup.extend(["--wm_bit_weight", "1", "--compensate_clean_bias", "--clean_bias_queries", "4"])
                        completed = subprocess.run(followup, capture_output=True, text=True,
                                                   env=os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
                        self.assertEqual(completed.returncode, 0, completed.stderr)
                        continued_directory = Path(followup[followup.index("--output_dir") + 1]) / "synthetic" / message
                        resumed = json.loads((continued_directory / "results.json").read_text())
                        self.assertEqual(resumed["config"]["embedding_updates"], 3)
                        self.assertTrue(resumed["config"]["optimizer_restarted"])
                        continued_key = torch.load(continued_directory / "watermark_key.pt", weights_only=False)
                        torch.testing.assert_close(continued_key["P"], key["P"])
                        if mode == "hypercube":
                            self.assertEqual(resumed["config"]["clean_bias_split"], "train")
                            self.assertTrue((continued_directory / "clean_bias.pt").exists())
                        else:
                            preparation = followup.copy()
                            preparation[preparation.index("--output_dir") + 1] = str(root / "preparation_only")
                            preparation.append("--training_only")
                            completed = subprocess.run(preparation, capture_output=True, text=True,
                                                       env=os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
                            self.assertEqual(completed.returncode, 0, completed.stderr)
                            directory = root / "preparation_only" / "synthetic" / message
                            training = json.loads((directory / "training_results.json").read_text())
                            self.assertTrue(training["training_complete"])
                            self.assertFalse(training["evaluation_complete"])
                            self.assertTrue((directory / "adapter_step_3.pt").exists())
                            self.assertFalse((directory / "results.json").exists())
                            self.assertFalse((directory / "recovery_progress.json").exists())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
