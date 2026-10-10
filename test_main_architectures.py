import ast
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import numpy as np
from types import SimpleNamespace

from collect_results import hf_rows, infer_group
from prepare_audio_latents import audio_records
from run_main_architectures import build_jobs, MESSAGES_32, compatible_legacy_result
from watermark_audio_flow import AudioFlow, StableVelocity
from watermark_codebooks import make_key, message_code, decode_signature, target_score


class CodebookTests(unittest.TestCase):
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
    def test_legacy_reuse_checks_protocol_and_normalizes_numeric_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"wm_message": "10101", "wm_K": 32, "steps": 1000,
                      "lora_alpha": 16.0, "n_detect_trials": 40, "n_queries": 4096}
            (root / "results.json").write_text(json.dumps({"config": config,
                                                          "metrics": {"wm_acc": 100, "fid_ratio": 1.02}}))
            command = ["python", "watermark_hf_flow_unet.py", "--wm_message", "10101",
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
        tree = ast.parse(Path("flow_watermark_mnist_mlp.py").read_text())
        helpers = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                                  and node.name in {"message_to_str", "decode_watermark", "compute_signature_stats"}],
                             type_ignores=[])
        for mode, bits in (("orthogonal", 5), ("hypercube", 5), ("hypercube", 32)):
            namespace = {"torch": torch, "np": np, "math": math, "D": 64, "device": "cpu",
                         "N_BITS": bits, "N_QUERIES": 8, "args": SimpleNamespace(codebook_mode=mode),
                         "decode_signature": decode_signature, "target_score": target_score}
            exec(compile(helpers, "flow_watermark_mnist_mlp.py", "exec"), namespace)
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
                    command = [sys.executable, "watermark_audio_flow.py", "--latents", str(latent_path),
                               "--dataset", "synthetic", "--architecture", architecture, "--codebook_mode", mode,
                               "--wm_message", message, "--output_dir", str(output), "--base_dir", str(root / "bases"),
                               "--base_steps", "2", "--steps", "2", "--width", "8", "--depth", "1",
                               "--batch_size", "2", "--eval_batch_size", "2", "--n_queries", "4",
                               "--n_detect_trials", "2", "--n_fid_samples", "2", "--device", "cpu", "--skip_quality"]
                    completed = subprocess.run(command, capture_output=True, text=True,
                                               env=os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    result = json.loads((output / "synthetic" / message / "results.json").read_text())
                    self.assertFalse(result["metrics"]["quality_complete"])
                    self.assertEqual(result["metrics"]["bits"], len(message))
                    key = torch.load(output / "synthetic" / message / "watermark_key.pt", weights_only=False)
                    self.assertEqual(tuple(key["codes"].shape), (32, 32))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
