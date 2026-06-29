"""Shared command-line argument definitions for the three subproject train.py scripts.

Minimal usage (run from subdirectories 1/2/3; all other args use defaults from this file)::

    python train.py --data-root ../data1 --epochs 20 --seed 42

``--epoch`` and ``--epochs`` are equivalent (e.g. ``--epoch 2000``).

If defaults are already epochs=20 and seed=42, only change the data path::

    python train.py --data-root ../data1
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
from pathlib import Path
from typing import Any

from data_common.pooled_data_split import ZTEST5_DEFAULT_POOL_TAG


def data_tag_from_root(data_root: str) -> str:
    """For output/<tag>/…: map physical dirs like high-voltage_cable back to legacy tags like data1."""
    from data_common.dataset_paths import dataset_tag_for_path, resolve_dataset_root

    p = Path(resolve_dataset_root(data_root))
    return dataset_tag_for_path(p)


def default_train_out_dir(data_root: str) -> str:
    """When --out-dir is omitted: output/<dataset-dir-name>/runs (relative to cwd)."""
    return str(Path("output") / data_tag_from_root(data_root) / "runs")


def resolve_train_out_dir(args: argparse.Namespace, data_root: str) -> str:
    """``--out-dir`` takes priority; for pooled ``--data-roots``, default ``output/<pool-tag>/runs``."""
    if getattr(args, "out_dir", None):
        return str(args.out_dir)
    roots = getattr(args, "data_roots", None)
    if roots and str(roots).strip():
        tag = str(getattr(args, "pool_tag", None) or "data134").strip() or "data134"
        return str(Path("output") / tag / "runs")
    return default_train_out_dir(data_root)


def save_torch_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Write checkpoint: BytesIO first, then atomic replace on disk to avoid partial files."""
    import torch

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    torch.save(payload, buf)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(buf.getvalue())
    os.replace(tmp, p)


def migrate_job_root_checkpoints_into_runs(out_dir: str | Path, *, log_prefix: str = "[train]") -> None:
    """If ``--out-dir`` is ``…/<job-name>/runs`` but ``last.pt`` / ``best.pt`` remain under ``…/<job-name>/``, move them into ``runs/``."""
    try:
        out_path = Path(out_dir).expanduser().resolve()
    except OSError:
        return
    if out_path.name != "runs":
        return
    parent = out_path.parent
    for name in ("last.pt", "best.pt"):
        src = parent / name
        dst = out_path / name
        if src.is_file() and not dst.is_file():
            try:
                shutil.move(str(src), str(dst))
                print(f"{log_prefix} legacy layout: moved {name} into runs/", flush=True)
            except OSError as e:
                print(f"{log_prefix} WARN failed to migrate {name}: {e}", flush=True)


def add_common_train_arguments(parser: argparse.ArgumentParser) -> None:
    """Same style as data1/data2: reference_signal + noise_signal + sample<id>+band_axis pairing."""
    parser.add_argument(
        "--data-root",
        type=str,
        default=".",
        help="Single data root directory (mutually exclusive with --data-roots).",
    )
    parser.add_argument(
        "--data-roots",
        type=str,
        default=None,
        dest="data_roots",
        help="Comma-separated data roots (e.g. data1,data3,data4); requires --split-manifest for pooled training.",
    )
    parser.add_argument(
        "--split-manifest",
        type=str,
        default=None,
        dest="split_manifest",
        help="Pooled split JSON (ztest5 default: splits/ztest5_data134_manifest.json).",
    )
    parser.add_argument(
        "--pool-tag",
        type=str,
        default=ZTEST5_DEFAULT_POOL_TAG,
        dest="pool_tag",
        help=f"Pooled multi-root tag (default {ZTEST5_DEFAULT_POOL_TAG}); checkpoints go to output/<pool-tag>/runs when --out-dir is omitted.",
    )
    parser.add_argument(
        "--reference-subdir",
        type=str,
        default="reference_signal",
        help="reference signal subdirectory name (relative to data-root).",
    )
    parser.add_argument(
        "--noisy-subdir",
        type=str,
        default="noise_signal",
        help="Noisy signal subdirectory name (relative to data-root). Usually noise_signal (not noisy_signal).",
    )
    parser.add_argument(
        "--band",
        type=str,
        default="all",
        choices=("low", "middle", "high", "all"),
        help="Interference band.",
    )
    parser.add_argument(
        "--segment-length",
        type=int,
        default=1024,
        help="Fixed sequence length (resample/pad strategy may differ per network).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, dest="train_ratio")
    parser.add_argument(
        "--shuffle-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to shuffle before splitting by sample id.",
    )
    parser.add_argument(
        "--epochs",
        "--epoch",
        type=int,
        default=500,
        dest="epochs",
        help="Training epochs; --epoch and --epochs are equivalent.",
    )
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate; when omitted: UNet default 1e-3, DnCNN default 1e-4, GAN uses --lr-g/--lr-d or config defaults.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        type=str,
        default=None,
        dest="out_dir",
        help="Training checkpoint directory; default output/<dataset-dir-name>/runs (pooled --data-roots: output/<pool-tag>/runs).",
    )
    parser.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="auto_resume",
        help="Auto-resume from last.pt in out-dir by default; use --no-auto-resume to disable.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0, dest="num_workers")

    from data_common.cv_train import add_cv_train_arguments

    add_cv_train_arguments(parser)

    parser.epilog = (
        "Minimal example (all other args use defaults above):\n"
        "  python train.py --data-root ../data1 --epochs 20 --seed 42\n"
        "When defaults are already epochs=20 and seed=42, you can write:\n"
        "  python train.py --data-root ../data1\n"
        "Without --out-dir: single dataset → output/<dataset-dir-name>/runs; pooled --data-roots → output/<pool-tag>/runs (default output/data134/runs).\n"
        "For GAN-specific flags under directory 1, append them at the end; full list: python gan_train.py -h"
    )
