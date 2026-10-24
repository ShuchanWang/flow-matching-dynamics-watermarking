"""Launch task runners: python run.py --list, or python run.py <script> [args]."""

import os
from pathlib import Path
import runpy
import sys
import unittest

from repo_layout import ROOT, SOURCE_DIRS, setup, source_path


def main():
    setup()
    os.chdir(ROOT)
    args = sys.argv[1:]
    if not args or args == ["--list"]:
        for directory in SOURCE_DIRS:
            for path in sorted(directory.glob("*.py")):
                if not path.name.startswith("test_"):
                    print(path.relative_to(ROOT))
        return
    if args[0] == "test":
        if len(args) > 1:
            suite = unittest.defaultTestLoader.loadTestsFromNames(args[1:])
        else:
            suite = unittest.TestSuite()
            for directory in SOURCE_DIRS:
                for path in sorted(directory.glob("test_*.py")):
                    suite.addTests(unittest.defaultTestLoader.loadTestsFromName(path.stem))
            suite.addTests(unittest.defaultTestLoader.loadTestsFromName("test_repo_layout"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(not result.wasSuccessful())
    path = Path(args[0])
    if len(path.parts) == 1:
        path = source_path(path.name)
    else:
        path = (ROOT / path).resolve()
        if not path.is_relative_to(ROOT) or path.suffix != ".py" or not path.is_file():
            raise SystemExit(f"Not a repository Python script: {args[0]}")
    sys.argv = [str(path), *args[1:]]
    runpy.run_path(str(path), run_name="__main__")


if __name__ == "__main__":
    main()
