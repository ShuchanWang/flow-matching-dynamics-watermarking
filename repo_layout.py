"""Repository paths and flat-name compatibility for task-organized runners."""

import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SOURCE_DIRS = [ROOT / "models", *sorted((ROOT / "experiments").glob("*/")), ROOT / "tools"]


def setup():
    paths = [str(ROOT), *(str(p) for p in SOURCE_DIRS)]
    sys.path[:0] = [p for p in paths if p not in sys.path]
    inherited = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths + [p for p in inherited if p]))


def source_path(name):
    """Resolve historical source basenames without root-level wrapper files."""
    matches = [p / name for p in SOURCE_DIRS if (p / name).is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one source named {name!r}; found {matches}")
    return matches[0]
