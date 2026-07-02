from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import math


@dataclass(frozen=True)
class TimeSeriesSample:
    """
    One time-series sample (time-domain signal only).

    This project typically uses tab/space separated lines like:
        449    1610678462805    778

    We will still *read* 3 columns (index, timestamp, value) for validation,
    but we only *keep* the cleaned signal values, and use natural indices
    (0..N-1) after removing bad rows.
    """

    value: List[float]

    @property
    def y(self) -> List[float]:
        """Convenience alias of `value`."""
        return self.value


@dataclass(frozen=True)
class TimeSeriesSampleWithMeta:
    """Time-series sample with optional index/timestamp metadata."""

    index: List[int]
    timestamp: List[int]
    value: List[float]


PathLike = Union[str, Path]


def pad_or_resample_to_length(
    y: List[float],
    target_length: int,
    *,
    mode: str = "pad_edge",
) -> Tuple[List[float], List[int]]:
    """
    Convert a variable-length 1D sequence to a fixed length WITHOUT cutting out a sub-window.

    Returns:
      - y_out: length == target_length
      - mask:  length == target_length, 1 for valid original samples, 0 for padded samples

    mode:
      - "pad_zero": pad with 0 at the end
      - "pad_edge": pad with last value at the end (better for continuity)
      - "resample_linear": linear-resample whole sequence to target_length (changes sampling rate)
    """
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
            # For short sequences, still resample to reduce padding artifacts.
            mode = "resample_linear"
        else:
            raise ValueError(f"unknown mode: {mode}")

        if mode != "resample_linear":
            y_out = y0 + [float(pad_val)] * (target_length - n)
            mask = [1] * n + [0] * (target_length - n)
            return y_out, mask

    # n > target_length OR forced resample
    if mode != "resample_linear":
        raise ValueError(
            f"len(y)={n} > target_length={target_length}. "
            f"Cutting is disabled; use mode='resample_linear' to keep whole sequence."
        )

    # Linear resample on normalized time axis [0, 1]
    y_in = list(map(float, y))
    x_in = [i / (n - 1) for i in range(n)] if n > 1 else [0.0]
    x_out = [i / (target_length - 1) for i in range(target_length)] if target_length > 1 else [0.0]

    y_out: List[float] = []
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


def _iter_data_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*.txt") if p.is_file()])


def _parse_timeseries_file(path: Path) -> Tuple[TimeSeriesSample, dict]:
    """
    Read one txt file and robustly drop bad lines:
    - skip empty lines
    - split by whitespace or comma; require at least 3 fields (use first 3)
    - skip non-numeric / NaN / inf

    Returns: (TimeSeriesSample, stats dict)
    """
    val: List[float] = []

    stats = {
        "total_lines": 0,
        "kept_lines": 0,
        "skipped_empty": 0,
        "skipped_bad_cols": 0,
        "skipped_non_numeric": 0,
    }

    # Data files may be in ANSI/GBK/UTF-8; errors='replace' keeps the scan going.
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            stats["total_lines"] += 1
            line = raw.strip()
            if not line:
                stats["skipped_empty"] += 1
                continue

            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                stats["skipped_bad_cols"] += 1
                continue

            a, b, c = parts[0], parts[1], parts[2]
            low = (a + b + c).lower()
            if "nan" in low or "inf" in low:
                stats["skipped_non_numeric"] += 1
                continue

            try:
                # still validate index/timestamp columns are numeric
                int(float(a))
                int(float(b))
                y = float(c)
            except Exception:
                stats["skipped_non_numeric"] += 1
                continue

            val.append(y)
            stats["kept_lines"] += 1

    return TimeSeriesSample(value=val), stats


def _parse_timeseries_file_with_meta(path: Path) -> Tuple[TimeSeriesSampleWithMeta, dict]:
    """
    Read one txt file and robustly drop bad lines.

    Supported line formats (whitespace or comma separated):
    - 1 col: value
    - 3+ cols: index, timestamp, value, ...

    Returns: (TimeSeriesSampleWithMeta, stats dict)
    """
    idx: List[int] = []
    ts: List[int] = []
    val: List[float] = []

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
                    y = float(parts[0])
                    i = len(val)
                    t = i
                    stats["format_value_only"] += 1
                elif len(parts) >= 3:
                    i = int(float(parts[0]))
                    t = int(float(parts[1]))
                    y = float(parts[2])
                    stats["format_index_ts_value"] += 1
                else:
                    stats["skipped_bad_cols"] += 1
                    continue
            except Exception:
                stats["skipped_non_numeric"] += 1
                continue

            idx.append(i)
            ts.append(t)
            val.append(y)
            stats["kept_lines"] += 1

    return TimeSeriesSampleWithMeta(index=idx, timestamp=ts, value=val), stats


def read_one_file_with_meta(path: PathLike) -> Tuple[TimeSeriesSampleWithMeta, dict]:
    """Public wrapper for reading a single file with metadata."""
    return _parse_timeseries_file_with_meta(Path(path))


def random_read_one_file(
    data_root: PathLike = Path(__file__).parent / "noise_signal",
    seed: Optional[int] = None,
) -> Tuple[str, List[float]]:
    """
    Function 1:
    Randomly read ONE data file under data_root (recursive),
    return (filename, time-series sample).
    """
    root = Path(data_root)
    files = _iter_data_files(root)
    if not files:
        raise FileNotFoundError(f"No .txt data files found under: {root}")

    rng = random.Random(seed)
    chosen = rng.choice(files)

    sample, stats = _parse_timeseries_file(chosen)
    if stats["kept_lines"] == 0:
        raise ValueError(f"File has no valid data lines after cleaning: {chosen}")

    return str(chosen), sample.y


def read_all_files(
    data_root: PathLike = Path(__file__).parent / "noise_signal",
    *,
    drop_empty: bool = True,
) -> List[Tuple[str, List[float]]]:
    """
    Function 2:
    Read ALL data files under data_root (recursive),
    return [(filename, time-series sample), ...]
    """
    root = Path(data_root)
    files = _iter_data_files(root)
    if not files:
        return []

    out: List[Tuple[str, List[float]]] = []
    for p in files:
        sample, stats = _parse_timeseries_file(p)
        if stats["kept_lines"] == 0 and drop_empty:
            continue
        out.append((str(p), sample.y))
    return out


def read_all_files_with_meta(
    data_root: PathLike = Path(__file__).parent / "noise_signal",
    *,
    drop_empty: bool = True,
) -> List[Tuple[str, TimeSeriesSampleWithMeta, dict]]:
    """
    Read ALL data files under data_root (recursive), returning metadata too.

    Returns:
        [(filename, sample_with_meta, stats), ...]
    """
    root = Path(data_root)
    files = _iter_data_files(root)
    if not files:
        return []

    out: List[Tuple[str, TimeSeriesSampleWithMeta, dict]] = []
    for p in files:
        sample, stats = _parse_timeseries_file_with_meta(p)
        if stats["kept_lines"] == 0 and drop_empty:
            continue
        out.append((str(p), sample, stats))
    return out

