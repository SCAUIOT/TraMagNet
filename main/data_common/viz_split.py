"""
Data splits shared by visualize_data backends (aligned with training ``our_data_split`` / K-fold).

Default ``--split test``: fixed **20% holdout test set** (``seed`` + ``shuffle_split`` + ``train_ratio``).
``--split train``: full 80% train pool (not a single CV fold); ``cv_train`` / ``cv_val`` inspect the current fold.
"""

from __future__ import annotations

import argparse
import ast
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Literal

from data_common.our_data_split import DataSplitRole, sample_ids_for_data_split
from data_common.flat_pairing import sample_id_from_reference_path

try:
    BooleanOptionalAction = argparse.BooleanOptionalAction
except AttributeError:

    class BooleanOptionalAction(argparse.Action):
        """Python 3.8: ``--flag`` / ``--no-flag`` matching 3.9+ behavior."""

        def __init__(
            self,
            option_strings,
            dest,
            default=None,
            type=None,
            choices=None,
            required=False,
            help=None,
            metavar=None,
        ):
            if type is not None or choices is not None:
                raise ValueError("BooleanOptionalAction does not support type or choices")
            if metavar is not None:
                raise ValueError("BooleanOptionalAction does not support metavar")
            if required:
                raise ValueError("BooleanOptionalAction is not compatible with required=True")
            if len(option_strings) != 1:
                raise ValueError("BooleanOptionalAction only accepts a single option string")
            opt = option_strings[0]
            if not opt.startswith("--") or "." in opt:
                raise ValueError("BooleanOptionalAction only accepts long options without a dot")
            option_strings = [opt, opt.replace("--", "--no-", 1)]
            super().__init__(
                option_strings=option_strings,
                dest=dest,
                nargs=0,
                default=default,
                required=False,
                help=help,
            )

        def __call__(self, parser, namespace, values, option_string=None):
            if option_string == self.option_strings[0]:
                setattr(namespace, self.dest, True)
            else:
                setattr(namespace, self.dest, False)

VizSplitName = Literal["test", "holdout", "train", "all", "cv_train", "cv_val"]

VIZ_SPLIT_CHOICES: tuple[str, ...] = ("test", "holdout", "train", "all", "cv_train", "cv_val")


def add_viz_split_arguments(
    parser: ArgumentParser,
    *,
    default_split: str = "test",
    default_train_ratio: float = 0.8,
    default_seed: int = 42,
    default_shuffle: bool = True,
) -> None:
    g = parser.add_argument_group("Data split (aligned with training 8:2 + optional K-fold)")
    g.add_argument(
        "--split",
        type=str,
        default=default_split,
        choices=VIZ_SPLIT_CHOICES,
        help="Default test: fixed holdout test set (20%%). train=full train pool; cv_*=K-fold subset; all=all samples.",
    )
    g.add_argument("--train-ratio", type=float, default=default_train_ratio, dest="train_ratio")
    g.add_argument(
        "--shuffle-split",
        action=BooleanOptionalAction,
        default=default_shuffle,
        dest="shuffle_split",
        help="Shuffle sample ids before splitting (matches training default).",
    )
    g.add_argument("--seed", type=int, default=default_seed, help="Split random seed (same as training --seed).")
    g.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        dest="cv_folds",
        help="K-fold count; 0=8:2 only. Applies to cv_train/cv_val or after sync from runs.",
    )
    g.add_argument("--cv-fold", type=int, default=0, dest="cv_fold", help="Current fold index for cv_train/cv_val.")
    g.add_argument(
        "--sync-runs-split",
        action=BooleanOptionalAction,
        default=True,
        dest="sync_runs_split",
        help="When --runs-dir is set and config.txt exists, sync train_ratio/seed/shuffle_split/cv_folds/cv_fold.",
    )


def split_to_data_role(split: str, *, cv_folds: int = 0) -> DataSplitRole | None:
    """
    Map CLI ``--split`` to ``sample_ids_for_data_split`` role.

    ``train`` in visualization means **full 80% train pool** (forces ``cv_folds=0``), not a CV fold.
    """
    s = str(split).lower().strip()
    if s == "all":
        return None
    if s in ("test", "holdout"):
        return "holdout_test"
    if s == "cv_train":
        return "cv_train"
    if s == "cv_val":
        return "cv_val"
    if s == "train":
        return "train"
    raise ValueError(f"unknown split: {split!r}")


def our_data_dataset_split_kwargs(
    split: str,
    *,
    train_ratio: float,
    seed: int,
    shuffle_split: bool,
    cv_folds: int = 0,
    cv_fold: int = 0,
) -> dict[str, object]:
    """Build train / holdout_eval / cv_* fields for ``OurDataConfig`` / multiprocess packs."""
    s = str(split).lower().strip()
    if s == "all":
        return dict(
            train=True,
            holdout_eval=False,
            cv_folds=0,
            cv_fold=0,
            train_ratio=float(train_ratio),
            seed=int(seed),
            shuffle_split=bool(shuffle_split),
        )
    role = split_to_data_role(s, cv_folds=cv_folds)
    assert role is not None
    nf = int(cv_folds)
    fi = int(cv_fold)
    if s == "train":
        nf = 0
    if role in ("cv_train", "cv_val") and nf < 2:
        raise ValueError(f"split={split!r} requires --cv-folds >= 2 (current {nf})")
    holdout = role == "holdout_test"
    use_cv = role in ("cv_train", "cv_val")
    return dict(
        train=role in ("train", "cv_train"),
        holdout_eval=holdout,
        cv_folds=nf if use_cv else 0,
        cv_fold=fi if use_cv else 0,
        train_ratio=float(train_ratio),
        seed=int(seed),
        shuffle_split=bool(shuffle_split),
    )


def folder_dataset_split_kwargs(
    split: str,
    *,
    train_ratio: float,
    seed: int,
    shuffle_split: bool,
    cv_folds: int = 0,
    cv_fold: int = 0,
) -> dict[str, object]:
    """``OurDataFolderConfig`` (1gan) uses the same field names as ``OurDataDataset``."""
    kw = our_data_dataset_split_kwargs(
        split,
        train_ratio=train_ratio,
        seed=seed,
        shuffle_split=shuffle_split,
        cv_folds=cv_folds,
        cv_fold=cv_fold,
    )
    return dict(
        train=bool(kw["train"]),
        holdout_eval=bool(kw["holdout_eval"]),
        cv_folds=int(kw["cv_folds"]),
        cv_fold=int(kw["cv_fold"]),
        train_ratio=float(train_ratio),
        split_seed=int(seed),
        shuffle_split=bool(shuffle_split),
    )


def chosen_sample_ids_from_specs(
    specs,
    *,
    split: str,
    train_ratio: float,
    seed: int,
    shuffle_split: bool,
    cv_folds: int = 0,
    cv_fold: int = 0,
    data_root: Path | None = None,
) -> set[str]:
    """Filter sample ids from ``list_pair_specs`` results (subway dual-channel shares sid)."""
    root = Path(data_root) if data_root is not None else None
    by_sid: dict[str, list[int]] = {}
    for i, sp in enumerate(specs):
        sid = sample_id_from_reference_path(sp.reference_path, data_root=root) if root else sample_id_from_reference_path(sp.reference_path)
        if not sid:
            continue
        by_sid.setdefault(sid, []).append(i)
    sids = sorted(by_sid.keys(), key=lambda s: int(s))
    if not sids:
        return set()
    s = str(split).lower().strip()
    if s == "all":
        return set(sids)
    role = split_to_data_role(s, cv_folds=cv_folds)
    assert role is not None
    nf = int(cv_folds)
    fi = int(cv_fold)
    if s == "train":
        nf = 0
    chosen = sample_ids_for_data_split(
        sids,
        role=role,
        train_ratio=float(train_ratio),
        seed=int(seed),
        shuffle_split=bool(shuffle_split),
        cv_folds=nf,
        cv_fold=fi,
    )
    return set(chosen)


def _infer_cv_fold_from_path(runs_dir: Path) -> int | None:
    try:
        cur = runs_dir.resolve()
    except OSError:
        return None
    for p in (cur, *cur.parents):
        name = p.name
        if name.startswith("fold_"):
            try:
                return int(name.split("_", 1)[1])
            except (IndexError, ValueError):
                pass
    return None


def maybe_sync_split_from_runs_config(
    args: Namespace,
    *,
    runs_dir: Path,
    log_prefix: str = "[viz]",
) -> None:
    """Sync split params from ``runs/config.txt`` or ``fold_k/.../config.txt``."""
    if not bool(getattr(args, "sync_runs_split", True)):
        return
    cfg_candidates = [runs_dir / "config.txt", runs_dir.parent / "config.txt"]
    cfg_path = next((p for p in cfg_candidates if p.is_file()), None)
    if cfg_path is None:
        fi = _infer_cv_fold_from_path(runs_dir)
        if fi is not None and int(getattr(args, "cv_folds", 0) or 0) > 0:
            args.cv_fold = int(fi)
            print(f"{log_prefix} inferred cv_fold={args.cv_fold} from path", flush=True)
        return
    try:
        raw = cfg_path.read_text(encoding="utf-8").strip()
        d = ast.literal_eval(raw)
    except (OSError, SyntaxError, TypeError, ValueError, MemoryError):
        return
    if not isinstance(d, dict):
        return
    updated: list[str] = []
    for key, attr in (
        ("train_ratio", "train_ratio"),
        ("seed", "seed"),
        ("shuffle_split", "shuffle_split"),
        ("cv_folds", "cv_folds"),
        ("cv_fold", "cv_fold"),
    ):
        v = d.get(key)
        if v is None:
            continue
        if key in ("seed", "cv_folds", "cv_fold"):
            setattr(args, attr, int(v))
        elif key == "train_ratio":
            setattr(args, attr, float(v))
        else:
            setattr(args, attr, bool(v))
        updated.append(f"{attr}={getattr(args, attr)}")
    if not updated and cfg_path.parent.name.startswith("fold_"):
        fi = _infer_cv_fold_from_path(runs_dir)
        if fi is not None:
            args.cv_fold = int(fi)
            updated.append(f"cv_fold={args.cv_fold}(path)")
    if updated:
        print(f"{log_prefix} synced from {cfg_path}: " + " ".join(updated), flush=True)


def describe_split_for_log(split: str, *, cv_folds: int, cv_fold: int) -> str:
    s = str(split).lower()
    if s in ("test", "holdout"):
        return "holdout_test(20%)"
    if s == "train":
        return "train_pool(80%)"
    if s == "cv_train":
        return f"cv_train(fold={cv_fold}/{cv_folds})"
    if s == "cv_val":
        return f"cv_val(fold={cv_fold}/{cv_folds})"
    if s == "all":
        return "all_samples"
    return s
