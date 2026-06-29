"""
Reproducible 8:2 split for a single data root (same algorithm as ``our_data_split`` / ``OurDataDataset``).

- Fixed ``seed`` + ``shuffle_split`` + ``train_ratio`` reproduces the same train pool / test set.
- Manifest writes ``holdout_test_sample_ids`` for shared use by ``eval_metrics`` / ``visualize_data`` / each ``train.py``.
- Distinct from pooled manifest (``data1/507`` group keys): test ids here are plain numeric sample ids (e.g. ``"636"``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from data_common.our_data_split import build_cv_split_manifest, write_cv_split_manifest
from data_common.pair_specs import list_pair_specs
from data_common.rename_manifest import sample_id_from_reference_path

MANIFEST_KIND = "single_dataset_82"
SCHEMA_VERSION = 1

SplitName = Literal["train", "test"]


def data_tag_from_root(data_root: str | Path) -> str:
    p = Path(data_root)
    try:
        return p.expanduser().resolve().name
    except OSError:
        return p.name or "data"


def split_params_from_manifest(manifest: dict) -> dict[str, Any]:
    """CLI params aligned with training/eval (match manifest fields)."""
    return {
        "train_ratio": float(manifest.get("train_ratio", 0.8)),
        "seed": int(manifest.get("seed", 42)),
        "shuffle_split": bool(manifest.get("shuffle_split", True)),
        "split_round": bool(manifest.get("split_round", True)),
    }


def manifest_filename(
    data_tag: str,
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> str:
    pct_train = int(round(float(train_ratio) * 100))
    pct_test = 100 - pct_train
    return f"{data_tag}_split{pct_train}{pct_test}_seed{int(seed)}.json"


def default_manifest_path(
    repo: Path,
    data_root: str | Path,
    *,
    splits_dir: Path | None = None,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Path:
    tag = data_tag_from_root(data_root)
    base = splits_dir if splits_dir is not None else (repo / "splits")
    return base / manifest_filename(tag, train_ratio=train_ratio, seed=seed)


def resolve_data_root(repo: Path, data_root: str | Path) -> Path:
    p = Path(data_root).expanduser()
    if p.is_absolute():
        return p.resolve()
    cand = (Path.cwd() / p).resolve()
    if cand.exists():
        return cand
    repo_cand = (repo / p).resolve()
    if repo_cand.exists():
        return repo_cand
    return cand


def collect_unique_sample_ids(
    data_root: str | Path,
    *,
    reference_subdir: str = "reference_signal",
    noisy_subdir: str = "noise_signal",
    band: str = "all",
    strict_all_bands: bool = True,
) -> list[str]:
    root = Path(data_root)
    strict = bool(strict_all_bands) if str(band).lower() == "all" else False
    specs = list_pair_specs(
        root,
        reference_subdir=reference_subdir,
        noisy_subdir=noisy_subdir,
        band=band,
        subway_dual_channels=False,
        strict_all_bands=strict,
    )
    sids: set[str] = set()
    for sp in specs:
        sid = sample_id_from_reference_path(sp.reference_path, data_root=root)
        if sid:
            sids.add(sid)
            continue
        m = re.match(r"^sample(\d+)_([xyz])\.txt$", sp.reference_path.name, re.IGNORECASE)
        if m:
            sids.add(m.group(1))
    return sorted(sids, key=lambda s: int(s))


def build_split_manifest(
    data_root: str | Path,
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
    shuffle_split: bool = True,
    reference_subdir: str = "reference_signal",
    noisy_subdir: str = "noise_signal",
    band: str = "all",
    strict_all_bands: bool = True,
) -> dict:
    root = Path(data_root)
    sids = collect_unique_sample_ids(
        root,
        reference_subdir=reference_subdir,
        noisy_subdir=noisy_subdir,
        band=band,
        strict_all_bands=strict_all_bands,
    )
    if not sids:
        raise RuntimeError(f"No sample id found: {root}")
    manifest = build_cv_split_manifest(
        sids,
        train_ratio=float(train_ratio),
        seed=int(seed),
        shuffle_split=bool(shuffle_split),
        split_round=True,
        cv_folds=0,
    )
    manifest["manifest_kind"] = MANIFEST_KIND
    manifest["schema_version"] = SCHEMA_VERSION
    manifest["data_tag"] = data_tag_from_root(root)
    manifest["band"] = str(band)
    manifest["reference_subdir"] = reference_subdir
    manifest["noisy_subdir"] = noisy_subdir
    return manifest


def save_split_manifest(path: str | Path, manifest: dict) -> Path:
    p = Path(path)
    write_cv_split_manifest(p, manifest)
    return p.resolve()


def load_split_manifest(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Split manifest not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def is_single_dataset_split_manifest(manifest: dict) -> bool:
    if str(manifest.get("manifest_kind", "")).strip() == MANIFEST_KIND:
        return True
    ids = manifest.get("holdout_test_sample_ids") or []
    if not ids:
        return False
    return "/" not in str(ids[0])


def sample_ids_for_split(manifest: dict, split: SplitName) -> list[str]:
    if split == "test":
        return [str(x) for x in (manifest.get("holdout_test_sample_ids") or [])]
    return [str(x) for x in (manifest.get("train_pool_sample_ids") or [])]


def sid_in_split(sid: str, manifest: dict, split: SplitName) -> bool:
    return str(sid) in set(sample_ids_for_split(manifest, split))


def assert_manifest_params(
    manifest: dict,
    *,
    train_ratio: float,
    seed: int,
    shuffle_split: bool,
    strict: bool = True,
) -> None:
    """When loading an existing manifest, verify CLI params match to avoid using the wrong test set."""
    sp = split_params_from_manifest(manifest)
    mismatches: list[str] = []
    if abs(sp["train_ratio"] - float(train_ratio)) > 1e-9:
        mismatches.append(f"train_ratio manifest={sp['train_ratio']} != {train_ratio}")
    if int(sp["seed"]) != int(seed):
        mismatches.append(f"seed manifest={sp['seed']} != {seed}")
    if bool(sp["shuffle_split"]) != bool(shuffle_split):
        mismatches.append(f"shuffle_split manifest={sp['shuffle_split']} != {shuffle_split}")
    if mismatches:
        msg = (
            "Split manifest does not match current params; test set will differ from training:\n  "
            + "\n  ".join(mismatches)
            + "\nUse the same --seed/--train-ratio/--shuffle-split, or --rebuild to regenerate the manifest."
        )
        if strict:
            raise ValueError(msg)
        print(f"[WARN] {msg}", flush=True)


def verify_manifest_reproducible(
    data_root: str | Path,
    manifest: dict,
    *,
    band: str | None = None,
    strict_all_bands: bool = True,
) -> tuple[bool, list[str]]:
    """Recompute from seed etc. recorded in manifest; train/test lists should match exactly."""
    band = str(band if band is not None else manifest.get("band", "all"))
    rebuilt = build_split_manifest(
        data_root,
        train_ratio=float(manifest["train_ratio"]),
        seed=int(manifest["seed"]),
        shuffle_split=bool(manifest["shuffle_split"]),
        reference_subdir=str(manifest.get("reference_subdir", "reference_signal")),
        noisy_subdir=str(manifest.get("noisy_subdir", "noise_signal")),
        band=band,
        strict_all_bands=strict_all_bands,
    )
    diffs: list[str] = []
    for key in ("train_pool_sample_ids", "holdout_test_sample_ids"):
        a = sample_ids_for_split(manifest, "train") if key.startswith("train") else sample_ids_for_split(manifest, "test")
        b = sample_ids_for_split(rebuilt, "train") if key.startswith("train") else sample_ids_for_split(rebuilt, "test")
        if a != b:
            diffs.append(f"{key}: length {len(a)} vs {len(b)}, compare first mismatch")
    return (len(diffs) == 0, diffs)


def resolve_manifest_path(
    repo: Path,
    data_root: str | Path,
    *,
    manifest: str | Path | None = None,
    train_ratio: float = 0.8,
    seed: int = 42,
    splits_dir: Path | None = None,
) -> Path:
    if manifest is not None and str(manifest).strip():
        p = Path(str(manifest).strip()).expanduser()
        if p.is_absolute():
            return p.resolve()
        cand = (Path.cwd() / p).resolve()
        if cand.is_file():
            return cand
        return (repo / p).resolve()
    tag = data_tag_from_root(data_root)
    base = splits_dir if splits_dir is not None else (repo / "splits")
    primary = base / manifest_filename(tag, train_ratio=train_ratio, seed=seed)
    if primary.is_file():
        return primary.resolve()
    # Legacy filename (no seed suffix)
    legacy = base / f"{tag}_split{int(round(train_ratio * 100))}{100 - int(round(train_ratio * 100))}.json"
    if legacy.is_file():
        return legacy.resolve()
    return primary.resolve()


def ensure_split_manifest(
    repo: Path,
    data_root: str | Path,
    *,
    manifest: str | Path | None = None,
    train_ratio: float = 0.8,
    seed: int = 42,
    shuffle_split: bool = True,
    rebuild: bool = False,
    splits_dir: Path | None = None,
    export_repo: bool = True,
    assert_params_match: bool = True,
    **build_kw,
) -> tuple[dict, Path]:
    root = resolve_data_root(repo, data_root)
    path = resolve_manifest_path(
        repo,
        root,
        manifest=manifest,
        train_ratio=train_ratio,
        seed=seed,
        splits_dir=splits_dir,
    )
    if path.is_file() and not rebuild:
        m = load_split_manifest(path)
        if assert_params_match:
            assert_manifest_params(
                m,
                train_ratio=train_ratio,
                seed=seed,
                shuffle_split=shuffle_split,
                strict=True,
            )
        return m, path
    m = build_split_manifest(
        root,
        train_ratio=train_ratio,
        seed=seed,
        shuffle_split=shuffle_split,
        **build_kw,
    )
    save_split_manifest(path, m)
    if export_repo:
        repo_path = default_manifest_path(repo, root, train_ratio=train_ratio, seed=seed)
        if repo_path.resolve() != path.resolve():
            save_split_manifest(repo_path, m)
    return m, path


def build_eval_segment_keys(
    data_root: str | Path,
    manifest: dict | str | Path,
    *,
    split: SplitName = "test",
    band: str = "all",
    subway_dual_channels: bool = True,
    reference_subdir: str = "reference_signal",
    noisy_subdir: str = "noise_signal",
) -> set[tuple[str, str]]:
    """
    From manifest, build ``(noisy filename, channel)`` set used by ``eval_metrics`` / ``visualize_data``,
    ensuring all methods evaluate the same test set.
    """
    from data_common.eval_split import _channel_tag_for_pair

    if not isinstance(manifest, dict):
        manifest = load_split_manifest(manifest)
    root = Path(data_root)
    chosen_sids = set(sample_ids_for_split(manifest, split))
    specs = list_pair_specs(
        root,
        reference_subdir=reference_subdir,
        noisy_subdir=noisy_subdir,
        band=str(band),
        subway_dual_channels=bool(subway_dual_channels),
        strict_all_bands=False,
    )
    keys: set[tuple[str, str]] = set()
    for sp in specs:
        m = _REFERENCE_SID_RE.match(sp.reference_path.name)
        if not m or m.group(1) not in chosen_sids:
            continue
        ch = _channel_tag_for_pair(value_column=int(sp.value_column), noisy_path=sp.noisy_path)
        keys.add((sp.noisy_path.name, ch))
    return keys


def format_split_banner(manifest: dict, path: Path | None = None) -> str:
    sp = split_params_from_manifest(manifest)
    tr = len(sample_ids_for_split(manifest, "train"))
    te = len(sample_ids_for_split(manifest, "test"))
    extra = f" file={path.name}" if path else ""
    return (
        f"[split_82] seed={sp['seed']} train_ratio={sp['train_ratio']} "
        f"shuffle_split={sp['shuffle_split']} train_ids={tr} test_ids={te}{extra}"
    )
