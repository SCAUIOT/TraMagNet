#!/usr/bin/env python3
"""Unified training entry point."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main"


def main() -> None:
    p = argparse.ArgumentParser(description="Train TraMagNet / DnCNN / ablation models")
    p.add_argument(
        "model",
        choices=("tramagnet", "dncnn", "dncnn_ablation", "unet_ablation"),
    )
    p.add_argument("extra", nargs=argparse.REMAINDER, help="Args passed to underlying trainer")
    args = p.parse_args()
    mapping = {
        "tramagnet": MAIN / "TraMagNet" / "train.py",
        "dncnn": MAIN / "DnCNN" / "train.py",
        "dncnn_ablation": ROOT / "ablation" / "dncnn_only" / "train.py",
        "unet_ablation": ROOT / "ablation" / "unet_single" / "train.py",
    }
    script = mapping[args.model]
    cmd = [sys.executable, str(script), *args.extra]
    raise SystemExit(subprocess.call(cmd, cwd=str(script.parent)))


if __name__ == "__main__":
    main()
