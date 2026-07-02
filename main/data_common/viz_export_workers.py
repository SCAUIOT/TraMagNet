"""Helpers for parallel / chunked export in visualize_data scripts."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Sequence, TypeVar

T = TypeVar("T")

_POOLED_OUTPUT_TAGS = frozenset({"data1", "data3", "data4"})


def output_run_lookup_tags(data_tag: str) -> tuple[str, ...]:
    """Map CLI pool tag to ``output/<tag>/runs`` candidates (single dataset also tries pooled data134)."""
    t = str(data_tag).strip().lower()
    if t == "data134":
        return ("data134", "data1", "data3", "data4")
    if t in _POOLED_OUTPUT_TAGS:
        return (t, "data134")
    return (t,)


def empty_viz_output_dirs(*dirs: Path) -> None:
    """Remove existing files/subdirs under each dir, then recreate empty dirs to avoid overwrite overhead."""
    for raw in dirs:
        d = Path(raw)
        if d.is_dir():
            for child in list(d.iterdir()):
                if child.is_file() or child.is_symlink():
                    child.unlink(missing_ok=True)
                elif child.is_dir():
                    shutil.rmtree(child)
        d.mkdir(parents=True, exist_ok=True)


def default_export_worker_count(*, cap: int = 8) -> int:
    """Default parallel export worker count (capped at ``cap``, at least 1)."""
    n = int(os.cpu_count() or 4)
    return max(1, min(n, int(cap)))


def checkpoint_run_candidates(*, repo: Path, data_tag: str, nn_dir: Path | None = None) -> list[Path]:
    """
    Weight dirs tried in order when ``--runs-dir`` is not set (deduped, order preserved).

    Prefer ``<nn_dir>/output/<tag>/runs`` (training output under each method subdir), then cwd
    ``output/<tag>/runs``, repo ``<repo>/output/<tag>/runs``, ``<nn_dir>/runs``, ``./runs``.
    For data1/data3/data4 also try pooled ``data134``.
    """
    raw: list[Path] = []
    for tag in output_run_lookup_tags(data_tag):
        if nn_dir is not None:
            raw.append(nn_dir / "output" / tag / "runs")
        raw.extend(
            [
                Path("output") / tag / "runs",
                repo / "output" / tag / "runs",
            ]
        )
    if nn_dir is not None:
        raw.append(nn_dir / "runs")
    raw.append(Path("runs"))
    seen: set[str] = set()
    out: list[Path] = []
    for p in raw:
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def split_into_n_chunks(items: Sequence[T], n_chunks: int) -> list[list[T]]:
    """Split ``items`` into up to ``n_chunks`` non-empty contiguous chunks (lengths differ by at most 1)."""
    xs = list(items)
    if n_chunks <= 1 or len(xs) <= 1:
        return [xs]
    n_chunks = min(n_chunks, len(xs))
    base, rem = divmod(len(xs), n_chunks)
    out: list[list[T]] = []
    i = 0
    for k in range(n_chunks):
        take = base + (1 if k < rem else 0)
        out.append(xs[i : i + take])
        i += take
    return out
