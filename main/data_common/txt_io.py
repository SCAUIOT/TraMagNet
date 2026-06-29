# -*- coding: utf-8 -*-
"""
Txt parsing consistent with legacy ``2/data/our_data_dataset._parse_timeseries_file_with_meta``;
shared by data1/2/3 read_official and in-repo training code, avoiding sys.path hacks to directory 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, NamedTuple, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


@dataclass(frozen=True)
class TimeSeriesSampleWithMeta:
    index: list[int]
    timestamp: list[int]
    value: list[float]


class TwoChannelSampleWithMeta(NamedTuple):
    index: list[int]
    timestamp: list[int]
    value_a: list[float]
    value_b: list[float]


def pad_or_resample_to_length(
    y: list[float],
    target_length: int,
    *,
    mode: str = "pad_edge",
) -> tuple[list[float], list[int]]:
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    n = len(y)
    if n == 0:
        return [0.0] * target_length, [0] * target_length
    if n == target_length:
        return list(map(float, y)), [1] * target_length
    if n < target_length:
        y0 = list(map(float, y))
        if mode == "pad_zero":
            pad_val = 0.0
        elif mode == "pad_edge":
            pad_val = y0[-1]
        elif mode == "resample_linear":
            mode = "resample_linear"
        else:
            raise ValueError(f"unknown mode: {mode}")
        if mode != "resample_linear":
            y_out = y0 + [float(pad_val)] * (target_length - n)
            mask = [1] * n + [0] * (target_length - n)
            return y_out, mask
    if mode != "resample_linear":
        raise ValueError(
            f"len(y)={n} > target_length={target_length}. "
            f"Cutting is disabled; use mode='resample_linear'."
        )
    y_in = list(map(float, y))
    x_in = [i / (n - 1) for i in range(n)] if n > 1 else [0.0]
    x_out = [i / (target_length - 1) for i in range(target_length)] if target_length > 1 else [0.0]
    y_out: list[float] = []
    j = 0
    for xo in x_out:
        while j + 1 < len(x_in) and x_in[j + 1] < xo:
            j += 1
        if j + 1 >= len(x_in):
            y_out.append(y_in[-1])
            continue
        x0, x1 = x_in[j], x_in[j + 1]
        y0, y1 = y_in[j], y_in[j + 1]
        if math.isclose(x1, x0):
            y_out.append(y0)
        else:
            t = (xo - x0) / (x1 - x0)
            y_out.append((1 - t) * y0 + t * y1)
    return y_out, [1] * target_length


def _iter_data_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*.txt") if p.is_file()])


def _parse_timeseries_file_with_meta(
    path: Path, *, value_column: int = 2
) -> tuple[TimeSeriesSampleWithMeta, dict]:
    idx: list[int] = []
    ts: list[int] = []
    val: list[float] = []
    stats = {
        "total_lines": 0,
        "kept_lines": 0,
        "skipped_empty": 0,
        "skipped_bad_cols": 0,
        "skipped_non_numeric": 0,
        "format_value_only": 0,
        "format_index_ts_value": 0,
    }
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            stats["total_lines"] += 1
            line = raw.strip()
            if not line:
                stats["skipped_empty"] += 1
                continue
            parts = line.replace(",", " ").split()
            low = "".join(parts[: min(3, len(parts))]).lower()
            if "nan" in low or "inf" in low:
                stats["skipped_non_numeric"] += 1
                continue
            try:
                if len(parts) == 1:
                    yv = float(parts[0])
                    i = len(val)
                    t = i
                    stats["format_value_only"] += 1
                elif len(parts) >= 3:
                    if len(parts) <= value_column:
                        stats["skipped_bad_cols"] += 1
                        continue
                    i = int(float(parts[0]))
                    t = int(float(parts[1]))
                    yv = float(parts[value_column])
                    stats["format_index_ts_value"] += 1
                else:
                    stats["skipped_bad_cols"] += 1
                    continue
            except Exception:
                stats["skipped_non_numeric"] += 1
                continue
            idx.append(i)
            ts.append(t)
            val.append(yv)
            stats["kept_lines"] += 1
    return TimeSeriesSampleWithMeta(index=idx, timestamp=ts, value=val), stats


def read_one_file_with_meta(
    path: PathLike, *, value_column: int = 2
) -> tuple[TimeSeriesSampleWithMeta, dict]:
    return _parse_timeseries_file_with_meta(Path(path), value_column=value_column)


def read_all_files_with_meta(
    dir_path: PathLike, *, drop_empty: bool = True, value_column: int = 2
) -> list[tuple[str, TimeSeriesSampleWithMeta, dict]]:
    root = Path(dir_path)
    files = _iter_data_files(root)
    if not files:
        return []
    out: list[tuple[str, TimeSeriesSampleWithMeta, dict]] = []
    for p in files:
        sample, stats = _parse_timeseries_file_with_meta(p, value_column=value_column)
        if stats["kept_lines"] == 0 and drop_empty:
            continue
        out.append((str(p), sample, stats))
    return out


def read_amplitude_np(path: PathLike, *, value_column: int = 2) -> np.ndarray:
    """Return only the amplitude column as a 1-D float32 vector (fast Dataset loading)."""
    s, st = read_one_file_with_meta(path, value_column=value_column)
    if st["kept_lines"] < 1:
        raise ValueError(f"{path}: no valid numeric rows")
    return np.asarray(s.value, dtype=np.float32)


def read_two_channel_file(path: PathLike) -> tuple[TwoChannelSampleWithMeta, dict]:
    path = Path(path)
    idx: list[int] = []
    ts: list[int] = []
    va: list[float] = []
    vb: list[float] = []
    stats = {
        "total_lines": 0,
        "kept_lines": 0,
        "skipped_empty": 0,
        "skipped_bad_cols": 0,
        "skipped_non_numeric": 0,
        "format_four_col": 0,
    }
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            stats["total_lines"] += 1
            line = raw.strip()
            if not line:
                stats["skipped_empty"] += 1
                continue
            parts = line.replace(",", " ").split()
            low = "".join(parts[: min(4, len(parts))]).lower()
            if "nan" in low or "inf" in low:
                stats["skipped_non_numeric"] += 1
                continue
            if len(parts) < 4:
                stats["skipped_bad_cols"] += 1
                continue
            try:
                i = int(float(parts[0]))
                t = int(float(parts[1]))
                a = float(parts[2])
                b = float(parts[3])
            except Exception:
                stats["skipped_non_numeric"] += 1
                continue
            idx.append(i)
            ts.append(t)
            va.append(a)
            vb.append(b)
            stats["kept_lines"] += 1
            stats["format_four_col"] += 1
    return TwoChannelSampleWithMeta(index=idx, timestamp=ts, value_a=va, value_b=vb), stats


def subway_noisy_has_four_value_columns(path: PathLike) -> bool:
    """Whether the first valid data row has at least four columns (two amplitude channels)."""
    p = Path(path)
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            return len(parts) >= 4
    return False
