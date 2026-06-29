"""
Sample split filtering shared by ``eval_metrics`` / ``loss_eval``, training, and ``visualize_data``.

By default only the fixed **20% holdout test set** is evaluated (``split=test``), matching ``our_data_split`` / ``OurDataDataset(holdout_eval=True)``.
"""

from __future__ import annotations

import re
from pathlib import Path

from data_common.pair_specs import list_pair_specs
from data_common.pooled_data_split import default_ztest5_pool_manifest_path
from data_common.rename_manifest import sample_id_from_reference_path
from data_common.dataset_paths import dataset_tag_for_path
from data_common.viz_split import VIZ_SPLIT_CHOICES, chosen_sample_ids_from_specs

_REFERENCE_SID_RE = re.compile(r"^sample(\d+)_([xyz])\.txt$", re.IGNORECASE)


def resolve_eval_split_manifest_path(
    repo: Path,
    explicit: str | Path | None = None,
    *,
    pool_tag: str = "data134",
) -> Path | None:
    """Explicit path wins; otherwise use ``splits/ztest5_<pool>_manifest.json`` if present."""
    if explicit is not None and str(explicit).strip():
        p = Path(str(explicit).strip()).expanduser()
        return p.resolve() if p.is_absolute() else (repo / p).resolve()
    default = default_ztest5_pool_manifest_path(repo, pool_tag)
    return default if default.is_file() else None


def channel_tag_for_eval(*, value_column: int, noisy_path: Path) -> str:
    """Matches ``ch_tag`` naming in ``eval_metrics.evaluate_methods_on_data1``."""
    return _channel_tag_for_pair(value_column=value_column, noisy_path=noisy_path)


def _channel_tag_for_pair(*, value_column: int, noisy_path: Path) -> str:
    from data_common.txt_io import subway_noisy_has_four_value_columns

    is_subway = "+subway" in noisy_path.name.lower()
    dual = bool(is_subway and subway_noisy_has_four_value_columns(noisy_path))
    if dual and int(value_column) == 3:
        return "ch1"
    return "ch0"


def segment_in_eval_split(
    noisy_name: str,
    eval_keys: set[tuple[str, str]] | None,
    *,
    channel: str,
) -> bool:
    """No filtering when ``eval_keys`` is ``None`` (split=all)."""
    if eval_keys is None:
        return True
    return (noisy_name, str(channel)) in eval_keys


def noisy_in_eval_split(
    noisy_name: str,
    eval_keys: set[tuple[str, str]] | None,
    *,
    channel: str = "ch0",
) -> bool:
    """Whether (noisy filename, channel) belongs to the current split (default ch0)."""
    return segment_in_eval_split(noisy_name, eval_keys, channel=channel)


def build_eval_segment_keys(
    data_root: Path,
    *,
    split: str = "test",
    train_ratio: float = 0.8,
    seed: int = 42,
    shuffle_split: bool = True,
    band: str = "all",
    subway_dual_channels: bool = True,
    reference_subdir: str = "reference_signal",
    noisy_subdir: str = "noise_signal",
    cv_folds: int = 0,
    cv_fold: int = 0,
    split_manifest_path: Path | None = None,
) -> set[tuple[str, str]] | None:
    """
    Return the set of ``(noisy filename, channel)`` pairs allowed in metrics; ``None`` when ``split=all`` (no filter).

    ``channel`` is ``ch0`` / ``ch1`` (matches result evaluation for data3 subway dual-channel).
    """
    s = str(split).lower().strip()
    if s == "all":
        return None

    if s not in VIZ_SPLIT_CHOICES:
        raise ValueError(f"split must be one of {VIZ_SPLIT_CHOICES}, got {split!r}")

    root = Path(data_root)
    # Split keys must cover pairs that actually exist under the data root; avoid strict_all_bands (data3 subway-only would drop samples).
    specs = list_pair_specs(
        root,
        reference_subdir=reference_subdir,
        noisy_subdir=noisy_subdir,
        band=str(band),
        subway_dual_channels=bool(subway_dual_channels),
        strict_all_bands=False,
    )
    if not specs:
        return set()

    dataset_tag = dataset_tag_for_path(root)
    if split_manifest_path is not None and Path(split_manifest_path).is_file():
        from data_common.pooled_data_split import holdout_sample_ids_for_dataset_tag, load_pooled_split_manifest
        from data_common.split_82 import is_single_dataset_split_manifest, load_split_manifest, sample_ids_for_split

        manifest = load_split_manifest(split_manifest_path)
        if is_single_dataset_split_manifest(manifest):
            if manifest.get("data_tag") and str(manifest.get("data_tag")) != dataset_tag:
                raise ValueError(
                    f"split manifest targets {manifest.get('data_tag')!r}, current data_root={dataset_tag!r}"
                )
            if s in ("test", "holdout"):
                chosen_sids = set(sample_ids_for_split(manifest, "test"))
            else:
                chosen_sids = set(sample_ids_for_split(manifest, "train"))
        else:
            manifest = load_pooled_split_manifest(split_manifest_path)
            if s in ("test", "holdout"):
                chosen_sids = holdout_sample_ids_for_dataset_tag(manifest, dataset_tag)
            else:
                chosen_sids = chosen_sample_ids_from_specs(
                    specs,
                    split=s,
                    train_ratio=float(train_ratio),
                    seed=int(seed),
                    shuffle_split=bool(shuffle_split),
                    cv_folds=int(cv_folds),
                    cv_fold=int(cv_fold),
                )
    else:
        chosen_sids = chosen_sample_ids_from_specs(
            specs,
            split=s,
            train_ratio=float(train_ratio),
            seed=int(seed),
            shuffle_split=bool(shuffle_split),
            cv_folds=int(cv_folds),
            cv_fold=int(cv_fold),
        )

    keys: set[tuple[str, str]] = set()
    for sp in specs:
        sid = sample_id_from_reference_path(sp.reference_path, data_root=root)
        if not sid or sid not in chosen_sids:
            continue
        ch = _channel_tag_for_pair(value_column=int(sp.value_column), noisy_path=sp.noisy_path)
        keys.add((sp.noisy_path.name, ch))
    return keys


def format_eval_split_banner(
    *,
    split: str,
    keys: set[tuple[str, str]] | None,
    train_ratio: float,
    seed: int,
    shuffle_split: bool,
    split_manifest_path: Path | None = None,
) -> str:
    if keys is None:
        return "[INFO] split=all: evaluate all txt under result dir (no train-split filter)."
    extra = ""
    if split_manifest_path is not None:
        extra = f" manifest={Path(split_manifest_path).name}"
    return (
        f"[INFO] split={split}: evaluate holdout/split segments only, {len(keys)} (noisy file, channel) pairs;"
        f"train_ratio={train_ratio}, seed={seed}, shuffle_split={shuffle_split}{extra}"
    )
