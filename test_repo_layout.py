import os
from pathlib import Path
import subprocess
import sys
import unittest

from repo_layout import ROOT, SOURCE_DIRS, setup, source_path


class LayoutTest(unittest.TestCase):
    def test_sources_are_unique(self):
        names = [p.name for directory in SOURCE_DIRS for p in directory.glob("*.py")]
        self.assertEqual(len(names), len(set(names)))

    def test_historical_name_resolves_without_root_wrapper(self):
        path = source_path("run_endpoint_loop_training.py")
        self.assertEqual(path.parent, ROOT / "experiments/endpoint")
        self.assertFalse((ROOT / path.name).exists())

    def test_subprocess_imports_and_cwd_independent_launcher(self):
        setup()
        command = [sys.executable, "-c", "import run_watermark_security as s; print(s.ROOT)"]
        result = subprocess.run(command, cwd="/tmp", text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout.strip(), str(ROOT))
        result = subprocess.run([sys.executable, str(ROOT / "run.py"), "--list"],
                                cwd="/tmp", text=True, capture_output=True, check=True)
        self.assertIn("experiments/endpoint/run_endpoint_loop_training.py", result.stdout)


if __name__ == "__main__":
    unittest.main()
