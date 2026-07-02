"""
Pooled split across data1 / data3 / data4 (and other roots): group key ``{dataset_dir}/{sample_id}`` (e.g. ``data3/42``).

Apply 8:2 holdout + K-fold on the full pool of sample ids (same ``seed`` / ``train_ratio`` as ``our_data_split``).
Manifest builders below are used for optional run-dir JSON export, not for reading ``splits/`` at startup.
"""

from __future__ import annotations

from pathlib import Path

from data_common.our_data_split import build_cv_split_manifest, write_cv_split_manifest
from data_common.pair_specs import list_pair_specs
from data_common.flat_pairing import sample_id_from_reference_path
from data_common.dataset_paths import (
    DEFAULT_POOL_ROOTS,
    DEFAULT_POOL_TAG,
    resolve_data_root_entry,
)

ZTEST5_DEFAULT_POOL_TAG = DEFAULT_POOL_TAG
ZTEST5_DEFAULT_DATA_ROOTS = DEFAULT_POOL_ROOTS


def group_key(dataset_tag: str, sample_id: str) -> str:
    return f"{str(dataset_tag).strip()}/{str(sample_id).strip()}"


def parse_group_key(gk: str) -> tuple[str, str]:
    s = str(gk).strip()
    if "/" not in s:
        return "", s
    tag, sid = s.split("/", 1)
    return tag.strip(), sid.strip()


def sort_group_keys(keys: list[str], *, tag_order: list[str] | None = None) -> list[str]:
    order = {t: i for i, t in enumerate(tag_order or [])}

    def _key(gk: str) -> tuple:
        tag, sid = parse_group_key(gk)
        try:
            n = int(sid)
        except ValueError:
            n = 0
        return (order.get(tag, 999), tag, n, gk)

    return sorted(keys, key=_key)


def parse_data_roots_arg(raw: str, *, repo: Path) -> list[tuple[str, Path]]:
    """``data1,data3,data4`` or ``../datasets/high-voltage_cable``; tag keeps CLI name (matches manifest group keys)."""
    out: list[tuple[str, Path]] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        tag, p = resolve_data_root_entry(part, repo=repo)
        out.append((tag, p))
    if not out:
        raise ValueError("data-roots is empty")
    return out


def collect_pooled_group_keys(
    root_entries: list[tuple[str, Path]],
    *,
    reference_subdir: str = "reference_signal",
    noisy_subdir: str = "noise_signal",
    band: str = "all",
    subway_dual_channels: bool = True,
    strict_all_bands: bool = True,
) -> list[str]:
    """Deduplicated, sorted group keys after merging multiple data roots."""
    tag_order = [t for t, _ in root_entries]
    seen: set[str] = set()
    keys: list[str] = []
    strict = bool(strict_all_bands) if str(band).lower() == "all" else False
    for tag, root in root_entries:
        specs = list_pair_specs(
            root,
            reference_subdir=reference_subdir,
            noisy_subdir=noisy_subdir,
            band=band,  # type: ignore[arg-type]
            subway_dual_channels=subway_dual_channels,
            strict_all_bands=strict,
        )
        for sp in specs:
            sid = sample_id_from_reference_path(sp.reference_path, data_root=root)
            if not sid:
                continue
            gk = group_key(tag, sid)
            if gk not in seen:
                seen.add(gk)
                keys.append(gk)
    return sort_group_keys(keys, tag_order=tag_order)


def build_pooled_cv_split_manifest(
    group_keys: list[str],
    *,
    pool_tag: str,
    root_entries: list[tuple[str, Path]],
    train_ratio: float = 0.8,
    seed: int = 42,
    shuffle_split: bool = True,
    split_round: bool = True,
    cv_folds: int = 5,
) -> dict:
    """Build split manifest on group keys (field names match single-dataset manifest; values are group keys)."""
    gkeys = sort_group_keys(list(group_keys))
    base = build_cv_split_manifest(
        gkeys,
        train_ratio=float(train_ratio),
        seed=int(seed),
        shuffle_split=bool(shuffle_split),
        split_round=bool(split_round),
        cv_folds=int(cv_folds),
    )
    base["pool_tag"] = str(pool_tag)
    base["group_key_format"] = "{dataset_tag}/{sample_id}"
    base["data_roots"] = {str(tag): str(tag) for tag, _ in root_entries}
    base["holdout_test_group_keys"] = list(base.get("holdout_test_sample_ids") or [])
    base["train_pool_group_keys"] = list(base.get("train_pool_sample_ids") or [])
    for fold in base.get("folds") or []:
        fold["train_group_keys"] = list(fold.get("train_sample_ids") or [])
        fold["val_group_keys"] = list(fold.get("val_sample_ids") or [])
    return base


def write_pooled_split_manifest(path: str | Path, manifest: dict) -> None:
    write_cv_split_manifest(path, manifest)

