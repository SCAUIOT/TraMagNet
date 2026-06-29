#!/usr/bin/env python3
"""Unified evaluation entry point."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main"


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate denoising methods (metrics + SNR)")
    p.add_argument("mode", choices=("metrics", "loss"), default="metrics", nargs="?")
    p.add_argument("extra", nargs=argparse.REMAINDER)
    args = p.parse_args()
    script = MAIN / ("eval_metrics.py" if args.mode == "metrics" else "loss_eval.py")
    cmd = [sys.executable, str(script), *args.extra]
    raise SystemExit(subprocess.call(cmd, cwd=str(MAIN)))


if __name__ == "__main__":
    main()
