"""
Pooled visualize_data export (``data134`` = data1 + data3 + data4).

With ``--data-root data134``, load ``data134_*`` weights once and export each library's segments to
``output/<backend>/data134/{data1|data3|data4}/{image,result}`` (not mixed in one folder).
Split is computed inline from ``train_ratio`` / ``seed`` (same as training).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from data_common.eval_split import build_eval_segment_keys, format_eval_split_banner
from data_common.pooled_data_split import (
    ZTEST5_DEFAULT_DATA_ROOTS,
    ZTEST5_DEFAULT_POOL_TAG,
    parse_data_roots_arg,
)
from data_common.viz_split import our_data_dataset_split_kwargs


@dataclass(frozen=True)
class VizExportTarget:
    dataset_tag: str
    data_root: str
    allowed_keys: set[tuple[str, str]] | None


@dataclass(frozen=True)
class VizPoolPlan:
    output_tag: str
    is_pooled: bool
    targets: tuple[VizExportTarget, ...]


def add_viz_pooled_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Pooled data134 (data1+data3+data4)")
    g.add_argument(
        "--pool-tag",
        type=str,
        default=ZTEST5_DEFAULT_POOL_TAG,
        dest="pool_tag",
        help=f"Pooled pool tag (default {ZTEST5_DEFAULT_POOL_TAG}).",
    )
    g.add_argument(
        "--data-roots",
        type=str,
        default=None,
        dest="data_roots",
        help=f"Comma-separated data roots for pooled train/eval (default {','.join(ZTEST5_DEFAULT_DATA_ROOTS)}).",
    )


def is_pooled_viz_data_root_arg(path_str: str, *, repo: Path) -> bool:
    """Treat ``data134`` or pool-only identifiers (not a real single-dataset dir) as pooled."""
    raw = (path_str or ".").strip()
    if raw in (".", ""):
        return False
    slug = Path(raw).expanduser().name.casefold()
    if slug == ZTEST5_DEFAULT_POOL_TAG.casefold():
        p = (repo / raw).expanduser() if not Path(raw).expanduser().is_absolute() else Path(raw).expanduser()
        if p.is_dir():
            clean = p / "reference_signal"
            if clean.is_dir():
                return False
        return True
    return False


def _data_tag_from_root(data_root: str) -> str:
    from data_common.dataset_paths import dataset_tag_for_path

    return dataset_tag_for_path(Path(data_root))


def _resolve_data_root(path_str: str, *, repo: Path) -> str:
    from data_common.resolve_dataset_root import resolve_dataset_root

    return resolve_dataset_root(path_str, repo=repo)


def resolve_viz_pool_plan(
    args: argparse.Namespace,
    repo: Path,
    *,
    split: str,
) -> VizPoolPlan:
    """Resolve single-dataset or data134 multi-dataset export targets."""
    data_root_arg = str(getattr(args, "data_root", ".") or ".")
    pooled = is_pooled_viz_data_root_arg(data_root_arg, repo=repo)

    if pooled:
        pool_tag = str(getattr(args, "pool_tag", None) or ZTEST5_DEFAULT_POOL_TAG).strip() or ZTEST5_DEFAULT_POOL_TAG
        roots_raw = getattr(args, "data_roots", None)
        if roots_raw and str(roots_raw).strip():
            root_entries = parse_data_roots_arg(str(roots_raw), repo=repo)
        else:
            root_entries = parse_data_roots_arg(",".join(ZTEST5_DEFAULT_DATA_ROOTS), repo=repo)
        targets: list[VizExportTarget] = []
        for ds_tag, root_p in root_entries:
            allowed = build_eval_segment_keys(
                root_p,
                split=split,
                train_ratio=float(args.train_ratio),
                seed=int(args.seed),
                shuffle_split=bool(args.shuffle_split),
                band=str(args.band),
                reference_subdir=str(args.reference_subdir),
                noisy_subdir=str(args.noisy_subdir),
                cv_folds=int(getattr(args, "cv_folds", 0)),
                cv_fold=int(getattr(args, "cv_fold", 0)),
            )
            print(
                format_eval_split_banner(
                    split=split,
                    keys=allowed,
                    train_ratio=float(args.train_ratio),
                    seed=int(args.seed),
                    shuffle_split=bool(args.shuffle_split),
                )
                + f" data_root={ds_tag}",
                flush=True,
            )
            targets.append(
                VizExportTarget(
                    dataset_tag=str(ds_tag),
                    data_root=str(root_p.resolve()),
                    allowed_keys=allowed,
                )
            )
        return VizPoolPlan(
            output_tag=pool_tag,
            is_pooled=True,
            targets=tuple(targets),
        )

    data_root = _resolve_data_root(data_root_arg, repo=repo)
    tag = _data_tag_from_root(data_root)
    allowed = build_eval_segment_keys(
        Path(data_root),
        split=split,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
        shuffle_split=bool(args.shuffle_split),
        band=str(args.band),
        reference_subdir=str(args.reference_subdir),
        noisy_subdir=str(args.noisy_subdir),
        cv_folds=int(getattr(args, "cv_folds", 0)),
        cv_fold=int(getattr(args, "cv_fold", 0)),
    )
    print(
        format_eval_split_banner(
            split=split,
            keys=allowed,
            train_ratio=float(args.train_ratio),
            seed=int(args.seed),
            shuffle_split=bool(args.shuffle_split),
        )
        + f" data_root={tag}",
        flush=True,
    )
    return VizPoolPlan(
        output_tag=tag,
        is_pooled=False,
        targets=(
            VizExportTarget(
                dataset_tag=tag,
                data_root=data_root,
                allowed_keys=allowed,
            ),
        ),
    )


def split_kwargs_for_viz_target(
    split: str,
    *,
    allowed_keys: set[tuple[str, str]] | None,
    train_ratio: float,
    seed: int,
    shuffle_split: bool,
    cv_folds: int,
    cv_fold: int,
) -> dict[str, object]:
    """Build dataset for ``split`` (e.g. test holdout); ``allowed_keys`` filters export indices only."""
    return our_data_dataset_split_kwargs(
        split,
        train_ratio=train_ratio,
        seed=seed,
        shuffle_split=shuffle_split,
        cv_folds=cv_folds,
        cv_fold=cv_fold,
    )


def dataset_export_indices(ds, allowed_keys: set[tuple[str, str]] | None) -> list[int]:
    if allowed_keys is None:
        return list(range(len(ds)))
    from data_common.eval_split import _channel_tag_for_pair

    out: list[int] = []
    pairs = ds._pairs
    for out_i, flat_i in enumerate(ds._indices):
        _sid, _axis, _c, n_fn, vcol = pairs[flat_i]
        ch = _channel_tag_for_pair(value_column=int(vcol), noisy_path=Path(n_fn))
        if (Path(n_fn).name, ch) in allowed_keys:
            out.append(out_i)
    return out


def merged_dataset_export_indices(
    datasets: list,
    allowed_keys: set[tuple[str, str]] | None,
) -> list[tuple[int, int]]:
    """Same as ``loss_eval._iter_indices``: ``(ds_idx, sample_idx)``."""
    if allowed_keys is None:
        return [(0, i) for i in range(len(datasets[0]))]
    out: list[tuple[int, int]] = []
    for di, ds in enumerate(datasets):
        for ii in dataset_export_indices(ds, allowed_keys):
            out.append((di, ii))
    return out


def prefixed_export_stem(key: str, *, dataset_tag: str | None, filename_suffix: str) -> str:
    """Single-dataset export; pooled mode uses per-dataset subdirs, so pass ``dataset_tag=None``."""
    base = f"{dataset_tag}_{key}" if dataset_tag else str(key)
    s = (filename_suffix or "").strip()
    return f"{base}{s}" if s else base


def resolve_viz_target_output_dirs(
    *,
    out_base: Path,
    dataset_tag: str,
    is_pooled: bool,
    output_dir: Path | None = None,
    result_dir: Path | None = None,
) -> tuple[Path, Path]:
    """
    Pooled: ``out_base/data1/image``, ``out_base/data1/result``, etc.
    Single-dataset: ``out_base/image``, ``out_base/result`` (unchanged from before).
    """
    if is_pooled:
        root = out_base / str(dataset_tag)
        img = (output_dir / dataset_tag) if output_dir is not None else (root / "image")
        res = (result_dir / dataset_tag) if result_dir is not None else (root / "result")
        return img, res
    return (
        output_dir if output_dir is not None else (out_base / "image"),
        result_dir if result_dir is not None else (out_base / "result"),
    )


def clear_viz_output_base(out_base: Path) -> None:
    import shutil

    if out_base.is_dir():
        shutil.rmtree(out_base)


def spec_in_allowed_keys(sp, allowed_keys: set[tuple[str, str]] | None) -> bool:
    if allowed_keys is None:
        return True
    from data_common.eval_split import _channel_tag_for_pair

    ch = _channel_tag_for_pair(value_column=int(sp.value_column), noisy_path=sp.noisy_path)
    return (sp.noisy_path.name, ch) in allowed_keys
