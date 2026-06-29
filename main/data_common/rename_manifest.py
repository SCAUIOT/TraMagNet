"""Load ``rename_manifest.json`` and build pairs for flat ``sample{i}.txt`` layout under public/datasets."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_FLAT_SAMPLE = re.compile(r"^sample(\d+)\.txt$", re.IGNORECASE)


def _entry_is_noisy_meta(entry: dict[str, Any]) -> bool:
    if "band" in entry:
        return True
    src = str(entry.get("source", ""))
    return "+" in src


def _is_reference_entry(entry: dict[str, Any], *, root: Path, reference_subdir: str) -> bool:
    if _entry_is_noisy_meta(entry):
        return False
    p = root / reference_subdir / str(entry["new_name"])
    return p.is_file()


def _is_noisy_entry(entry: dict[str, Any], *, root: Path, noisy_subdir: str) -> bool:
    if not _entry_is_noisy_meta(entry) and "channel" in entry:
        src = str(entry.get("source", ""))
        if "+" not in src:
            return False
    p = root / noisy_subdir / str(entry["new_name"])
    return p.is_file()


@lru_cache(maxsize=32)
def _load_manifest_cached(manifest_path: str) -> tuple[dict[str, Any], ...] | None:
    p = Path(manifest_path)
    if not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return None
    return tuple(data)


def load_rename_manifest(data_root: Path) -> list[dict[str, Any]] | None:
    path = (Path(data_root) / "rename_manifest.json").resolve()
    cached = _load_manifest_cached(str(path))
    if cached is None:
        return None
    return list(cached)


def sample_id_from_reference_path(reference_path: Path, *, data_root: Path | None = None) -> str | None:
    """Parse split sample id from manifest ``original_id`` or legacy filename."""
    root = Path(data_root) if data_root is not None else reference_path.parent.parent
    manifest = load_rename_manifest(root)
    if manifest:
        name = reference_path.name
        for e in manifest:
            if str(e.get("new_name")) != name:
                continue
            if _entry_is_noisy_meta(e):
                continue
            if (root / "reference_signal" / name).is_file():
                return str(e["original_id"])
    m = re.match(r"^sample(\d+)_([xyz])\.txt$", reference_path.name, re.IGNORECASE)
    if m:
        return m.group(1)
    m2 = _FLAT_SAMPLE.match(reference_path.name)
    if m2:
        return m2.group(1)
    return None


def list_pair_specs_from_manifest(
    data_root: Path,
    *,
    reference_subdir: str = "reference_signal",
    noisy_subdir: str = "noise_signal",
    band: str = "low",
    subway_dual_channels: bool = False,
    strict_all_bands: bool = False,
) -> list | None:
    from .pair_specs import PairSpec

    root = Path(data_root)
    manifest = load_rename_manifest(root)
    if not manifest:
        return None

    reference_dir = root / reference_subdir
    noisy_dir = root / noisy_subdir
    if not reference_dir.is_dir() or not noisy_dir.is_dir():
        return None

    reference_entries: list[dict[str, Any]] = []
    noisy_entries: list[dict[str, Any]] = []
    for e in manifest:
        if _is_reference_entry(e, root=root, reference_subdir=reference_subdir):
            reference_entries.append(e)
        elif _is_noisy_entry(e, root=root, noisy_subdir=noisy_subdir):
            noisy_entries.append(e)

    if not reference_entries or not noisy_entries:
        return None

    def _noisy_key(e: dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(e.get("original_id", "")),
            str(e.get("axis", "")).lower(),
            str(e.get("channel", "")).lower(),
            str(e.get("band", "")).lower(),
        )

    noisy_map: dict[tuple[str, str, str, str], Path] = {}
    for e in noisy_entries:
        noisy_map[_noisy_key(e)] = noisy_dir / str(e["new_name"])

    def _reference_key(e: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(e.get("original_id", "")),
            str(e.get("axis", "")).lower(),
            str(e.get("channel", "")).lower(),
        )

    has_bands = any(k[3] for k in noisy_map)

    if band.lower() == "all" and strict_all_bands and has_bands:
        specs: list = []
        for ce in reference_entries:
            ck = _reference_key(ce)
            reference_path = reference_dir / str(ce["new_name"])
            trip: list = []
            ok = True
            for band_name in ("low", "middle", "high"):
                nk = (*ck, band_name)
                if nk not in noisy_map:
                    ok = False
                    break
                trip.append(PairSpec(reference_path=reference_path, noisy_path=noisy_map[nk], value_column=2))
            if ok:
                specs.extend(trip)
        if specs:
            return sorted(specs, key=lambda s: (s.reference_path.name, s.noisy_path.name, s.value_column))
        return []

    if not has_bands:
        specs = []
        for ce in reference_entries:
            ck = _reference_key(ce)
            reference_path = reference_dir / str(ce["new_name"])
            nk = (*ck, "")
            if nk in noisy_map:
                specs.append(PairSpec(reference_path=reference_path, noisy_path=noisy_map[nk], value_column=2))
        return sorted(specs, key=lambda s: (s.reference_path.name, s.noisy_path.name, s.value_column))

    bands = ["low", "middle", "high"] if band.lower() == "all" else [band.lower()]
    specs = []
    for ce in reference_entries:
        ck = _reference_key(ce)
        reference_path = reference_dir / str(ce["new_name"])
        matched = False
        for band_name in bands:
            nk = (*ck, band_name)
            if nk in noisy_map:
                specs.append(PairSpec(reference_path=reference_path, noisy_path=noisy_map[nk], value_column=2))
                matched = True
        if not matched:
            nk_flat = (*ck, "")
            if nk_flat in noisy_map:
                specs.append(PairSpec(reference_path=reference_path, noisy_path=noisy_map[nk_flat], value_column=2))

    if not specs:
        available = sorted({k[3] for k in noisy_map if k[3]})
        if len(available) == 1:
            only = available[0]
            for ce in reference_entries:
                ck = _reference_key(ce)
                reference_path = reference_dir / str(ce["new_name"])
                nk = (*ck, only)
                if nk in noisy_map:
                    specs.append(PairSpec(reference_path=reference_path, noisy_path=noisy_map[nk], value_column=2))

    return sorted(specs, key=lambda s: (s.reference_path.name, s.noisy_path.name, s.value_column))
