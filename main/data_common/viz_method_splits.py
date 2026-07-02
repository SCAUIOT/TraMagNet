"""Visualization helpers: DnCNN / TraMagNet test sets and their overlap."""

from __future__ import annotations

from pathlib import Path

from data_common.eval_split import build_eval_segment_keys, format_eval_split_banner


def build_dncnn_test_segment_keys(
    data_root: Path,
    *,
    split: str = "test",
    train_ratio: float = 0.8,
    seed: int = 42,
    shuffle_split: bool = True,
    band: str = "all",
    subway_dual_channels: bool = True,
) -> set[tuple[str, str]] | None:
    """Single-dataset holdout test keys (8:2 split from ``seed`` / ``train_ratio``)."""
    if str(split).lower().strip() == "all":
        return None
    return build_eval_segment_keys(
        data_root,
        split=split,
        train_ratio=train_ratio,
        seed=seed,
        shuffle_split=shuffle_split,
        band=band,
        subway_dual_channels=subway_dual_channels,
    )


def build_gan_test_segment_keys(
    data_root: Path,
    repo: Path,
    *,
    split: str = "test",
    train_ratio: float = 0.8,
    seed: int = 42,
    shuffle_split: bool = True,
    band: str = "all",
    subway_dual_channels: bool = True,
) -> set[tuple[str, str]] | None:
    """TraMagNet test keys — same inline 8:2 holdout as DnCNN for the given ``data_root``."""
    del repo
    return build_dncnn_test_segment_keys(
        data_root,
        split=split,
        train_ratio=train_ratio,
        seed=seed,
        shuffle_split=shuffle_split,
        band=band,
        subway_dual_channels=subway_dual_channels,
    )


def intersect_segment_keys(
    a: set[tuple[str, str]] | None,
    b: set[tuple[str, str]] | None,
) -> set[tuple[str, str]] | None:
    if a is None and b is None:
        return None
    if a is None:
        return set(b) if b is not None else set()
    if b is None:
        return set(a)
    return set(a) & set(b)


def list_noisy_files_for_segment_keys(
    keys: set[tuple[str, str]] | None,
    noise_dir: Path,
) -> list[str]:
    """Deduplicated noisy filenames from ``(noisy filename, channel)`` keys (must exist in noise_dir)."""
    if keys is None:
        return sorted(p.name for p in noise_dir.glob("*.txt") if p.is_file())
    names = sorted({fn for fn, _ch in keys})
    return [n for n in names if (noise_dir / n).is_file()]


def format_overlap_split_banner(
    *,
    split: str,
    dncnn_keys: set[tuple[str, str]] | None,
    gan_keys: set[tuple[str, str]] | None,
    overlap_keys: set[tuple[str, str]] | None,
    train_ratio: float,
    seed: int,
    shuffle_split: bool,
) -> str:
    nc = len(dncnn_keys) if dncnn_keys is not None else "all"
    ng = len(gan_keys) if gan_keys is not None else "all"
    no = len(overlap_keys) if overlap_keys is not None else "all"
    return (
        f"[INFO] split={split} comparison: DnCNN test {nc} segments;"
        f"TraMagNet test {ng} segments;"
        f"overlap {no} segments; plot/statistics on overlap only."
        f" train_ratio={train_ratio}, seed={seed}, shuffle_split={shuffle_split}"
    )


def print_method_test_banners(
    *,
    split: str,
    dncnn_keys: set[tuple[str, str]] | None,
    gan_keys: set[tuple[str, str]] | None,
    train_ratio: float,
    seed: int,
    shuffle_split: bool,
) -> None:
    print(
        format_eval_split_banner(
            split=split,
            keys=dncnn_keys,
            train_ratio=train_ratio,
            seed=seed,
            shuffle_split=shuffle_split,
        ).replace("[INFO]", "[INFO] DnCNN", 1),
        flush=True,
    )
    print(
        format_eval_split_banner(
            split=split,
            keys=gan_keys,
            train_ratio=train_ratio,
            seed=seed,
            shuffle_split=shuffle_split,
        ).replace("[INFO]", "[INFO] TraMagNet", 1),
        flush=True,
    )
