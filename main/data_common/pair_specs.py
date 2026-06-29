# -*- coding: utf-8 -*-
"""
Enumerate (reference, noisy) pairs under any data root: try data1/2 band naming, then data3 subway, then ``<base>.txt`` / ``<base>_band.txt``.
Used by ``OurDataFolderDataset`` and each ``data*/read_official.list_pair_specs``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .txt_io import subway_noisy_has_four_value_columns
from .rename_manifest import list_pair_specs_from_manifest


_REFERENCE_NAME = re.compile(r"^sample(\d+)_([xyz])\.txt$", re.IGNORECASE)
_STEM_UNDERSCORE_BAND = re.compile(r"^(.+)_(low|middle|high)\.txt$", re.IGNORECASE)


@dataclass(frozen=True)
class PairSpec:
    reference_path: Path
    noisy_path: Path
    #: Amplitude column when parsing noisy txt (0-based; matches ``txt_io.read_one_file_with_meta(..., value_column=)``)
    value_column: int = 2


def list_pair_specs(
    data_root: Path,
    *,
    reference_subdir: str = "reference_signal",
    noisy_subdir: str = "noise_signal",
    band: str = "low",
    subway_dual_channels: bool = False,
    strict_all_bands: bool = False,
) -> list[PairSpec]:
    """
    Return a sorted list of pairs.
    - If ``sample<id>+{low|middle|high}_{axis}.txt`` exists, use that layout only (data1/2).
    - Else use ``sample<id>_{axis}+subway.txt`` (data3); optionally split columns 3 and 4 into two samples.
    - If neither: try ``<base>.txt`` with ``<base>_low|middle|high.txt`` (datatmp-style flat naming without axis suffix).
    """
    root = Path(data_root)
    reference_dir = root / reference_subdir
    noisy_dir = root / noisy_subdir
    if not reference_dir.is_dir() or not noisy_dir.is_dir():
        return []

    manifest_specs = list_pair_specs_from_manifest(
        root,
        reference_subdir=reference_subdir,
        noisy_subdir=noisy_subdir,
        band=band,
        subway_dual_channels=subway_dual_channels,
        strict_all_bands=strict_all_bands,
    )
    if manifest_specs is not None:
        return manifest_specs

    if band.lower() == "all" and strict_all_bands:
        strict_specs: list[PairSpec] = []
        for reference_path in sorted(reference_dir.glob("sample*_*.txt")):
            m = _REFERENCE_NAME.match(reference_path.name)
            if not m:
                continue
            sid, axis = m.group(1), m.group(2).lower()
            trip: list[PairSpec] = []
            ok = True
            for band_name in ("low", "middle", "high"):
                noisy_path = noisy_dir / f"sample{sid}+{band_name}_{axis}.txt"
                if not noisy_path.is_file():
                    ok = False
                    break
                trip.append(PairSpec(reference_path=reference_path, noisy_path=noisy_path, value_column=2))
            if ok:
                strict_specs.extend(trip)
        if strict_specs:
            return sorted(
                strict_specs, key=lambda s: (s.reference_path.name, s.noisy_path.name, s.value_column)
            )

    bands = ["low", "middle", "high"] if band.lower() == "all" else [band.lower()]
    specs: list[PairSpec] = []

    for band_name in bands:
        for reference_path in sorted(reference_dir.glob("sample*_*.txt")):
            m = _REFERENCE_NAME.match(reference_path.name)
            if not m:
                continue
            sid, axis = m.group(1), m.group(2).lower()
            noisy_path = noisy_dir / f"sample{sid}+{band_name}_{axis}.txt"
            if noisy_path.is_file():
                specs.append(PairSpec(reference_path=reference_path, noisy_path=noisy_path, value_column=2))

    if specs:
        return sorted(specs, key=lambda s: (s.reference_path.name, s.noisy_path.name, s.value_column))

    for reference_path in sorted(reference_dir.glob("sample*_*.txt")):
        m = _REFERENCE_NAME.match(reference_path.name)
        if not m:
            continue
        sid, axis = m.group(1), m.group(2).lower()
        noisy_path = noisy_dir / f"sample{sid}_{axis}+subway.txt"
        if not noisy_path.is_file():
            continue
        if subway_dual_channels and subway_noisy_has_four_value_columns(noisy_path):
            specs.append(PairSpec(reference_path, noisy_path, 2))
            specs.append(PairSpec(reference_path, noisy_path, 3))
        else:
            specs.append(PairSpec(reference_path, noisy_path, 2))

    if specs:
        return sorted(specs, key=lambda s: (s.reference_path.name, s.noisy_path.name, s.value_column))

    # <base>.txt ↔ <base>_low|middle|high.txt (no +, no _x/_y/_z axis suffix)
    bands_flat = ["low", "middle", "high"] if band.lower() == "all" else [band.lower()]
    flat_specs: list[PairSpec] = []
    for noisy_path in sorted(noisy_dir.glob("*.txt")):
        mm = _STEM_UNDERSCORE_BAND.match(noisy_path.name)
        if not mm:
            continue
        base, bname = mm.group(1), mm.group(2).lower()
        if bname not in bands_flat:
            continue
        reference_path = reference_dir / f"{base}.txt"
        if reference_path.is_file():
            flat_specs.append(PairSpec(reference_path=reference_path, noisy_path=noisy_path, value_column=2))

    return sorted(flat_specs, key=lambda s: (s.reference_path.name, s.noisy_path.name, s.value_column))
