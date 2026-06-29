"""K-fold cross-validation training: CLI args, output dirs, and split manifests."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable, TypeVar

from data_common.our_data_split import build_cv_split_manifest, write_cv_split_manifest
from data_common.pair_specs import list_pair_specs
from data_common.viz_split import BooleanOptionalAction

_reference_RE = re.compile(r"^sample(\d+)_([xyz])\.txt$", re.IGNORECASE)

T = TypeVar("T")


def add_cv_train_arguments(parser: argparse.ArgumentParser) -> None:
    """May be called after ``common_train_cli.add_common_train_arguments``."""
    g = parser.add_argument_group("Cross-validation (K-fold within 8:2 train pool)")
    g.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        dest="cv_folds",
        help="K-fold count on train pool; 0 disables CV and uses fixed 8:2 single train/val (legacy). Default 5.",
    )
    g.add_argument(
        "--cv-fold",
        type=int,
        default=0,
        dest="cv_fold",
        help="Current fold index [0, cv-folds); when exclusive with --run-all-cv-folds, train only this fold.",
    )
    g.add_argument(
        "--run-all-cv-folds",
        action=BooleanOptionalAction,
        default=True,
        dest="run_all_cv_folds",
        help="Train all folds in sequence; weights go to out-dir/fold_<k>/ (default on).",
    )
    g.add_argument(
        "--no-cv-split-manifest",
        action="store_true",
        dest="no_cv_split_manifest",
        help="Do not write split_manifest.json.",
    )


def format_cv_fold_info(cv_fold: int, cv_folds: int) -> str:
    """Human-readable fold info (1-based), e.g. ``fold 2/5``; empty when ``cv_folds<=0``."""
    nf = int(cv_folds)
    if nf <= 0:
        return ""
    return f"fold {int(cv_fold) + 1}/{nf}"


def cv_fold_log_prefix(cv_fold: int, cv_folds: int) -> str:
    """Log line prefix, e.g. ``[fold 2/5] ``; empty when CV is off."""
    info = format_cv_fold_info(cv_fold, cv_folds)
    return f"[{info}] " if info else ""


def resolve_fold_out_dir(base_out: str | Path, *, cv_folds: int, cv_fold: int) -> Path:
    base = Path(base_out)
    if int(cv_folds) <= 0:
        return base
    return base / f"fold_{int(cv_fold)}"


def iter_cv_folds(args: argparse.Namespace) -> list[int]:
    nf = int(getattr(args, "cv_folds", 0) or 0)
    if nf <= 0:
        return [0]
    if bool(getattr(args, "run_all_cv_folds", False)):
        return list(range(nf))
    return [int(getattr(args, "cv_fold", 0) or 0) % nf]


def run_per_cv_fold(
    args: argparse.Namespace,
    *,
    base_out_dir: str | Path,
    train_fn: Callable[[argparse.Namespace, int, Path], T],
) -> list[T]:
    """
    Call ``train_fn(args, fold_index, fold_out_dir)`` per fold.
    When ``cv_folds==0``, call once with ``fold_index=0``, ``fold_out_dir=base_out_dir``.
    """
    results: list[T] = []
    folds = iter_cv_folds(args)
    nf = int(getattr(args, "cv_folds", 0) or 0)
    base = Path(base_out_dir)
    for fold in folds:
        out = resolve_fold_out_dir(base, cv_folds=nf, cv_fold=fold)
        results.append(train_fn(args, fold, out))
    return results


def collect_sorted_sample_ids(
    data_root: str | Path,
    *,
    reference_subdir: str = "reference_signal",
    noisy_subdir: str = "noise_signal",
    band: str = "all",
    subway_dual_channels: bool = True,
    strict_all_bands: bool = True,
) -> list[str]:
    """Same as ``OurDataDataset``: dedupe by sample id, then sort ascending."""
    root = Path(data_root)
    strict = bool(strict_all_bands) if str(band).lower() == "all" else False
    specs = list_pair_specs(
        root,
        reference_subdir=reference_subdir,
        noisy_subdir=noisy_subdir,
        band=band,  # type: ignore[arg-type]
        subway_dual_channels=subway_dual_channels,
        strict_all_bands=strict,
    )
    sids: set[str] = set()
    for sp in specs:
        m = _reference_RE.match(sp.reference_path.name)
        if m:
            sids.add(m.group(1))
    return sorted(sids, key=lambda s: int(s))


def maybe_write_split_manifest(
    *,
    manifest_path: Path,
    sids: list[str],
    train_ratio: float,
    seed: int,
    shuffle_split: bool,
    cv_folds: int,
    skip: bool = False,
) -> None:
    if skip or int(cv_folds) <= 0:
        return
    manifest = build_cv_split_manifest(
        sids,
        train_ratio=float(train_ratio),
        seed=int(seed),
        shuffle_split=bool(shuffle_split),
        cv_folds=int(cv_folds),
    )
    write_cv_split_manifest(manifest_path, manifest)
