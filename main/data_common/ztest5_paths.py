"""Discover ztest5 grid dirs and checkpoint paths for visualization / evaluation."""

from __future__ import annotations

import re
from pathlib import Path

from data_common.cv_ensemble import job_dir_has_ckpt, runs_base_has_any_ckpt

ZTEST5_GRID_DIR_TRAMAGNET = "ztest5_TraMagNet_grid"

_JOB_EPOCH_RE = re.compile(r"_e(\d+)$", re.IGNORECASE)


def backend_ztest5_meta(nn_dir: Path | None) -> tuple[str, str] | None:
    """
    Infer ztest5 grid name and job name infix from the method subdirectory.

    Returns ``(grid_dir_name, tag_infix)``, e.g. ``("ztest5_TraMagNet_grid", "tm")``.
    """
    if nn_dir is None:
        return None
    key = nn_dir.name.casefold()
    if key in ("tramagnet", "TraMagNet"):
        return ZTEST5_GRID_DIR_TRAMAGNET, "tm"
    if key in ("cnn", "dncnn"):
        return "ztest5_cnn_grid", "cn"
    return None


def parse_job_epoch_suffix(job_name: str) -> int:
    m = _JOB_EPOCH_RE.search(str(job_name).strip())
    if m is None:
        return -1
    try:
        return int(m.group(1))
    except ValueError:
        return -1


_POOLED_ZTEST5_TAGS = frozenset({"data1", "data3", "data4"})


def ztest5_lookup_data_tags(data_tag: str) -> tuple[str, ...]:
    """
    Enumerate data prefixes used to match ztest5 job directory names.

    ``data1`` / ``data3`` / ``data4`` also try pooled ``data134`` jobs;
    ``data134`` tries each single-dataset prefix as well.
    """
    t = str(data_tag).strip().lower()
    if t == "data134":
        return ("data134", "data1", "data3", "data4")
    if t in _POOLED_ZTEST5_TAGS:
        return (t, "data134")
    return (t,)


def job_name_matches_dataset(job_name: str, *, data_tag: str, tag_infix: str) -> bool:
    """Match dirs like ``data3_4u_msestft_2_8_randz_e1000`` or ``data134_5g_...``."""
    name = str(job_name).casefold()
    tag = str(data_tag).casefold()
    infix = str(tag_infix).casefold()
    prefixes = (
        f"{tag}_{infix}_",
        f"{tag}-{infix}_",
        f"{tag}_{infix}",
    )
    return any(name.startswith(p) for p in prefixes)


def job_name_matches_any_dataset_tag(
    job_name: str,
    *,
    data_tags: tuple[str, ...],
    tag_infix: str,
) -> bool:
    return any(
        job_name_matches_dataset(job_name, data_tag=dt, tag_infix=tag_infix) for dt in data_tags
    )


def normalize_cv_runs_base(path: Path) -> Path:
    """
    Normalize user path to K-fold weight root (``runs`` dir with ``fold_*`` or single ``best.pt``).

    - ``…/data3_4u_…_e1000`` → ``…/data3_4u_…_e1000/runs``
    - Keep ``…/runs`` unchanged when it already contains ``fold_0``
    - When only ``runs/fold_*`` exists under job root, enter ``runs`` (do not treat job root as a single fold)
    """
    p = Path(path)
    inner = p / "runs"
    if inner.is_dir():
        if (inner / "fold_0").is_dir() or runs_base_has_any_ckpt(inner, cv_folds=0):
            return inner
    if (p / "fold_0").is_dir() or runs_base_has_any_ckpt(p, cv_folds=0):
        return p
    if inner.is_dir():
        return inner
    return p


def ztest5_grid_roots(*, repo: Path, nn_dir: Path | None, grid_dir_name: str) -> list[Path]:
    raw: list[Path] = []
    if nn_dir is not None:
        raw.append(nn_dir / "output" / grid_dir_name)
    raw.extend(
        [
            Path("output") / grid_dir_name,
            repo / "output" / grid_dir_name,
        ]
    )
    if nn_dir is not None:
        raw.append(repo / nn_dir.name / "output" / grid_dir_name)
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


def iter_ztest5_job_runs_bases(
    *,
    repo: Path,
    data_tag: str,
    nn_dir: Path | None,
    grid_dir_name: str,
    tag_infix: str,
    job_name: str | None = None,
    cv_folds: int = 5,
    ckpt_prefer: str = "last",
) -> list[Path]:
    """
    List ``…/<job>/runs`` under the ztest5 grid (dirs that contain checkpoints only).

    When ``job_name`` is set, return that job only; else all jobs matching ``{data_tag}_{tag_infix}_*``,
    sorted by ``_e<epoch>`` in the dir name descending (prefer newer training).
    """
    out: list[Path] = []
    want = str(job_name).strip() if job_name else ""
    lookup_tags = ztest5_lookup_data_tags(data_tag)
    infix_token = f"_{str(tag_infix).casefold()}_"

    def _append_job_dir(jd: Path) -> None:
        base = normalize_cv_runs_base(jd)
        if job_dir_has_ckpt(jd, cv_folds=int(cv_folds), prefer=ckpt_prefer) or runs_base_has_any_ckpt(
            base, cv_folds=int(cv_folds), prefer=ckpt_prefer
        ):
            out.append(base)

    for grid in ztest5_grid_roots(repo=repo, nn_dir=nn_dir, grid_dir_name=grid_dir_name):
        if not grid.is_dir():
            continue
        if want:
            jd = grid / want
            if jd.is_dir():
                base = normalize_cv_runs_base(jd)
                if runs_base_has_any_ckpt(base, cv_folds=int(cv_folds), prefer=ckpt_prefer):
                    out.append(base)
            continue
        for jd in grid.iterdir():
            if not jd.is_dir():
                continue
            if not job_name_matches_any_dataset_tag(
                jd.name, data_tags=lookup_tags, tag_infix=tag_infix
            ):
                continue
            _append_job_dir(jd)

        # Still no match: any job in grid with ``_{4u|5g}_`` and weights (e.g. data134 pool only)
        if not want and not out:
            for jd in grid.iterdir():
                if not jd.is_dir():
                    continue
                if infix_token not in jd.name.casefold():
                    continue
                _append_job_dir(jd)

    out.sort(key=lambda p: parse_job_epoch_suffix(p.parent.name), reverse=True)
    seen: set[str] = set()
    deduped: list[Path] = []
    for p in out:
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    return deduped


def extend_run_candidates_with_ztest5(
    candidates: list[Path],
    *,
    repo: Path,
    data_tag: str,
    nn_dir: Path | None,
    job_name: str | None = None,
    cv_folds: int = 5,
    ckpt_prefer: str = "last",
) -> list[Path]:
    """Append ztest5 grid job ``runs`` dirs after standard ``output/<tag>/runs`` candidates."""
    meta = backend_ztest5_meta(nn_dir)
    if meta is None:
        return candidates
    grid_dir_name, tag_infix = meta
    ztest5_bases = iter_ztest5_job_runs_bases(
        repo=repo,
        data_tag=data_tag,
        nn_dir=nn_dir,
        grid_dir_name=grid_dir_name,
        tag_infix=tag_infix,
        job_name=job_name,
        cv_folds=int(cv_folds),
        ckpt_prefer=str(ckpt_prefer),
    )
    seen: set[str] = set()
    out: list[Path] = []
    for p in list(candidates) + ztest5_bases:
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out
