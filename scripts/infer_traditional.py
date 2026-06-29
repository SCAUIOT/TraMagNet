#!/usr/bin/env python3
"""Run traditional morphological filters on a dataset."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "main" / "traditional" / "run_py_denoise_methods.py"


def main() -> None:
    cmd = [sys.executable, str(SCRIPT), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, cwd=str(SCRIPT.parent)))


if __name__ == "__main__":
    main()
