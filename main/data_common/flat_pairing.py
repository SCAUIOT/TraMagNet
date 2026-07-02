"""Pair flat ``sample{i}.txt`` files by matching filename (1:1 index layout)."""

from __future__ import annotations

import re
from pathlib import Path

_SAMPLE_TXT_RE = re.compile(r"^sample(\d+)\.txt$", re.IGNORECASE)


def sample_index_from_filename(name: str) -> str | None:
    m = _SAMPLE_TXT_RE.match(str(name).strip())
    if m is None:
        return None
    return m.group(1)


def sample_index_from_path(path: Path) -> str | None:
    return sample_index_from_filename(path.name)


def sample_id_from_reference_path(reference_path: Path, *, data_root: Path | None = None) -> str | None:
    """Split / pool sample id = numeric index in ``sample{i}.txt``."""
    del data_root
    return sample_index_from_filename(reference_path.name)


def axis_from_reference_path(reference_path: Path, *, data_root: Path | None = None) -> str:
    """Legacy field kept for pair tuples; 1:1 index layout has no axis metadata."""
    del reference_path, data_root
    return "x"


def reference_filename_for_noisy(noisy_name: str, *, data_root: Path) -> str | None:
    """Paired clean file uses the same basename when it exists."""
    reference_path = Path(data_root) / "reference_signal" / noisy_name
    if reference_path.is_file():
        return noisy_name
    return None


def sort_sample_txt_names(names: list[str]) -> list[str]:
    def _key(n: str) -> tuple[int, str]:
        idx = sample_index_from_filename(n)
        return (int(idx) if idx is not None else 10**9, n)

    return sorted(names, key=_key)


def list_pair_specs_by_matching_names(
    data_root: Path,
    *,
    reference_subdir: str = "reference_signal",
    noisy_subdir: str = "noise_signal",
    subway_dual_channels: bool = False,
) -> list:
    from .pair_specs import PairSpec
    from .txt_io import subway_noisy_has_four_value_columns

    root = Path(data_root)
    reference_dir = root / reference_subdir
    noisy_dir = root / noisy_subdir
    if not reference_dir.is_dir() or not noisy_dir.is_dir():
        return []

    reference_by_name = {p.name: p for p in reference_dir.glob("*.txt") if p.is_file()}
    specs: list[PairSpec] = []
    for noisy_name in sort_sample_txt_names([p.name for p in noisy_dir.glob("*.txt") if p.is_file()]):
        reference_path = reference_by_name.get(noisy_name)
        if reference_path is None:
            continue
        noisy_path = noisy_dir / noisy_name
        if subway_dual_channels and subway_noisy_has_four_value_columns(noisy_path):
            specs.append(PairSpec(reference_path=reference_path, noisy_path=noisy_path, value_column=2))
            specs.append(PairSpec(reference_path=reference_path, noisy_path=noisy_path, value_column=3))
        else:
            specs.append(PairSpec(reference_path=reference_path, noisy_path=noisy_path, value_column=2))
    return specs
