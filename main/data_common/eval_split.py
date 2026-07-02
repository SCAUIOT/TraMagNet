"""
Sample split filtering shared by ``eval_metrics`` / ``loss_eval``, training, and ``visualize_data``.

By default only the fixed **20% holdout test set** is evaluated (``split=test``), matching ``our_data_split`` / ``OurDataDataset(holdout_eval=True)``.
Splits are computed in code from ``train_ratio`` / ``seed`` / ``shuffle_split`` (no external manifest files).
"""

from __future__ import annotations

from pathlib import Path

from data_common.pair_specs import list_pair_specs
from data_common.flat_pairing import sample_id_from_reference_path
from data_common.viz_split import VIZ_SPLIT_CHOICES, chosen_sample_ids_from_specs


def channel_tag_for_eval(*, value_column: int, noisy_path: Path) -> str:
    """Matches ``ch_tag`` naming in ``eval_metrics.evaluate_methods_on_data1``."""
    return _channel_tag_for_pair(value_column=value_column, noisy_path=noisy_path)


def _channel_tag_for_pair(*, value_column: int, noisy_path: Path) -> str:
    from data_common.txt_io import subway_noisy_has_four_value_columns

    if int(value_column) == 3:
        return "ch1"
    if int(value_column) == 2 and subway_noisy_has_four_value_columns(noisy_path):
        return "ch0"
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
) -> str:
    if keys is None:
        return "[INFO] split=all: evaluate all txt under result dir (no train-split filter)."
    return (
        f"[INFO] split={split}: evaluate holdout/split segments only, {len(keys)} (noisy file, channel) pairs;"
        f"train_ratio={train_ratio}, seed={seed}, shuffle_split={shuffle_split}"
    )
