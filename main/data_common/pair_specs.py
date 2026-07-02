# -*- coding: utf-8 -*-
"""Enumerate (clean, noisy) pairs by matching ``sample{i}.txt`` basenames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .flat_pairing import list_pair_specs_by_matching_names


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
    Return sorted (clean, noisy) pairs where both sides use the same ``sample{i}.txt`` name.

    ``band`` / ``strict_all_bands`` are accepted for CLI compatibility but ignored — each file is one pair.
    """
    del band, strict_all_bands
    root = Path(data_root)
    if not (root / reference_subdir).is_dir() or not (root / noisy_subdir).is_dir():
        return []

    specs = list_pair_specs_by_matching_names(
        root,
        reference_subdir=reference_subdir,
        noisy_subdir=noisy_subdir,
        subway_dual_channels=subway_dual_channels,
    )
    if specs:
        return specs

    raise FileNotFoundError(
        f"{root}: no matching sample{{i}}.txt pairs under {reference_subdir}/ vs {noisy_subdir}/."
    )
