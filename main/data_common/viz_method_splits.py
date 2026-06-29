"""Visualization helpers: CNN / MagGAN test sets and their overlap (CNN does not use GAN data134 manifest)."""

from __future__ import annotations

from pathlib import Path

from data_common.eval_split import (
    build_eval_segment_keys,
    format_eval_split_banner,
    resolve_eval_split_manifest_path,
)


def build_cnn_test_segment_keys(
    data_root: Path,
    *,
    split: str = "test",
    train_ratio: float = 0.8,
    seed: int = 42,
    shuffle_split: bool = True,
    band: str = "all",
    subway_dual_channels: bool = True,
) -> set[tuple[str, str]] | None:
    """DnCNN baseline single-dataset holdout test keys."""
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
        split_manifest_path=None,
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
    split_manifest: Path | str | None = None,
) -> set[tuple[str, str]] | None:
    """TraMagNet pooled data134 holdout test keys (``ztest5_data134_manifest.json``)."""
    if str(split).lower().strip() == "all":
        return None
    manifest = resolve_eval_split_manifest_path(repo, split_manifest)
    return build_eval_segment_keys(
        data_root,
        split=split,
        train_ratio=train_ratio,
        seed=seed,
        shuffle_split=shuffle_split,
        band=band,
        subway_dual_channels=subway_dual_channels,
        split_manifest_path=manifest,
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
    cnn_keys: set[tuple[str, str]] | None,
    gan_keys: set[tuple[str, str]] | None,
    overlap_keys: set[tuple[str, str]] | None,
    train_ratio: float,
    seed: int,
    shuffle_split: bool,
    gan_manifest: Path | None,
) -> str:
    nc = len(cnn_keys) if cnn_keys is not None else "all"
    ng = len(gan_keys) if gan_keys is not None else "all"
    no = len(overlap_keys) if overlap_keys is not None else "all"
    mf = f" manifest={gan_manifest.name}" if gan_manifest is not None else ""
    return (
        f"[INFO] split={split} comparison: CNN test {nc} segments (single-dataset 8:2);"
        f"MagGAN test {ng} segments (data134{mf});"
        f"overlap {no} segments; plot/statistics on overlap only."
        f" train_ratio={train_ratio}, seed={seed}, shuffle_split={shuffle_split}"
    )


def print_method_test_banners(
    *,
    split: str,
    cnn_keys: set[tuple[str, str]] | None,
    gan_keys: set[tuple[str, str]] | None,
    train_ratio: float,
    seed: int,
    shuffle_split: bool,
    gan_manifest: Path | None,
) -> None:
    print(
        format_eval_split_banner(
            split=split,
            keys=cnn_keys,
            train_ratio=train_ratio,
            seed=seed,
            shuffle_split=shuffle_split,
            split_manifest_path=None,
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
            split_manifest_path=gan_manifest,
        ).replace("[INFO]", "[INFO] TraMagNet", 1),
        flush=True,
    )
