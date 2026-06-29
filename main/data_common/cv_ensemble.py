"""Resolve K-fold training weight dirs and checkpoint lists for ensemble inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import argparse

from common_train_cli import default_train_out_dir, data_tag_from_root


def _pick_best_or_last_in(base: Path, prefer: str) -> Path | None:
    bp = base / "best.pt"
    lp = base / "last.pt"
    if prefer == "best" and bp.is_file():
        return bp
    if lp.is_file():
        return lp
    if bp.is_file():
        return bp
    return None


def pick_ckpt_in_dir(run_dir: Path, prefer: str = "best") -> Path:
    """
    Pick ``best.pt`` / ``last.pt`` under ``run_dir``, ``run_dir/runs``, or K-fold layout ``…/runs/fold_k/``.
    """
    prefer = str(prefer).strip().lower()
    bases: list[Path] = [run_dir]
    inner = run_dir / "runs"
    if inner.is_dir():
        bases.insert(0, inner)

    for base in bases:
        hit = _pick_best_or_last_in(base, prefer)
        if hit is not None:
            return hit

    for base in bases:
        avail = list_available_fold_ckpt_paths(base, cv_folds=32, prefer=prefer)
        if avail:
            return avail[0]

    raise FileNotFoundError(
        f"No best.pt / last.pt: {run_dir} (tried runs/ and runs/fold_* subdirs)"
    )


def list_available_fold_ckpt_paths(
    base_out: Path,
    *,
    cv_folds: int,
    prefer: str = "best",
) -> list[Path] | None:
    """
    Collect checkpoints from ``fold_0`` … ``fold_{cv_folds-1}`` that **already** have weights (partial K-fold OK).
    """
    nf = int(cv_folds)
    if nf < 1:
        return None
    prefer = str(prefer).strip().lower()
    paths: list[Path] = []
    base = Path(base_out)
    for k in range(nf):
        fold_dir = base / f"fold_{k}"
        if not fold_dir.is_dir():
            continue
        hit = _pick_best_or_last_in(fold_dir, prefer)
        if hit is not None:
            paths.append(hit)
    return paths if paths else None


def list_fold_ckpt_paths(
    base_out: Path,
    *,
    cv_folds: int,
    prefer: str = "best",
) -> list[Path] | None:
    """
    If ``base_out/fold_0`` … ``fold_{cv_folds-1}`` all have weights, return per-fold ckpt paths;
    else ``None`` (full ensemble unavailable; try ``list_available_fold_ckpt_paths``).
    """
    nf = int(cv_folds)
    if nf < 2:
        return None
    paths: list[Path] = []
    base = Path(base_out)
    for k in range(nf):
        fold_dir = base / f"fold_{k}"
        if not fold_dir.is_dir():
            return None
        hit = _pick_best_or_last_in(fold_dir, str(prefer).strip().lower())
        if hit is None:
            return None
        paths.append(hit)
    return paths


def runs_base_has_any_ckpt(
    runs_base: Path,
    *,
    cv_folds: int = 5,
    prefer: str = "best",
) -> bool:
    """Whether any usable checkpoint exists under ``runs`` root or ``fold_k`` (including partial folds when K-fold training incomplete)."""
    base = Path(runs_base)
    if not base.is_dir():
        return False
    try:
        pick_ckpt_in_dir(base, prefer)
        return True
    except FileNotFoundError:
        pass
    nf = int(cv_folds)
    if nf >= 2:
        if list_fold_ckpt_paths(base, cv_folds=nf, prefer=prefer) is not None:
            return True
        for k in range(nf):
            fold_dir = base / f"fold_{k}"
            if not fold_dir.is_dir():
                continue
            try:
                pick_ckpt_in_dir(fold_dir, prefer)
                return True
            except FileNotFoundError:
                pass
    return False


def job_dir_has_ckpt(
    job_dir: Path,
    *,
    cv_folds: int = 5,
    prefer: str = "best",
) -> bool:
    """Whether a ztest5 job dir (``…/job_name/runs/fold_*``) already has weights."""
    jd = Path(job_dir)
    if not jd.is_dir():
        return False
    for sub in (jd, jd / "runs"):
        if sub.is_dir() and runs_base_has_any_ckpt(sub, cv_folds=cv_folds, prefer=prefer):
            return True
    return False


def fold_dir_for_config(base_out: Path, *, cv_fold: int = 0) -> Path:
    """Prefer ``fold_0`` (or ``fold_k/runs``) when reading ``config.txt``."""
    base = Path(base_out)
    fold = base / f"fold_{int(cv_fold)}"
    inner = fold / "runs"
    return inner if inner.is_dir() else fold


def default_cv_train_base_out(data_root: str) -> Path:
    """Same as training default: ``output/<dataset dir name>/runs``."""
    return Path(default_train_out_dir(data_root))


@dataclass(frozen=True)
class InferenceCkptPlan:
    mode: Literal["single", "ensemble"]
    ckpt_paths: tuple[Path, ...]
    config_dir: Path
    label: str


def resolve_inference_ckpts(
    *,
    repo: Path,
    data_root: str,
    runs_dir: Path | None,
    run_candidates: list[Path],
    ckpt_prefer: str = "best",
    cv_folds: int = 5,
    use_ensemble: bool = True,
) -> InferenceCkptPlan:
    """
    Resolve inference checkpoints: return ensemble when full K-fold weights exist, else single model.
    """
    nf = int(cv_folds)
    prefer = str(ckpt_prefer)

    if runs_dir is not None:
        rd = Path(runs_dir)
        if use_ensemble and nf >= 2:
            ens = list_fold_ckpt_paths(rd, cv_folds=nf, prefer=prefer)
            if ens is None:
                ens = list_available_fold_ckpt_paths(rd, cv_folds=nf, prefer=prefer)
            if ens is not None and len(ens) >= 1:
                label = f"cv{len(ens)}_ensemble" if len(ens) < nf else f"cv{nf}_ensemble"
                return InferenceCkptPlan(
                    mode="ensemble",
                    ckpt_paths=tuple(ens),
                    config_dir=fold_dir_for_config(rd, cv_fold=0),
                    label=label,
                )
        ckpt = pick_ckpt_in_dir(rd, prefer)
        cfg = rd / "runs" if (rd / "runs").is_dir() else rd
        return InferenceCkptPlan(
            mode="single",
            ckpt_paths=(ckpt,),
            config_dir=cfg,
            label=rd.name,
        )

    if use_ensemble and nf >= 2:
        for cand in run_candidates:
            ens = list_fold_ckpt_paths(cand, cv_folds=nf, prefer=prefer)
            if ens is None:
                ens = list_available_fold_ckpt_paths(cand, cv_folds=nf, prefer=prefer)
            if ens is not None and len(ens) >= 1:
                tag = data_tag_from_root(data_root)
                job_label = cand.parent.name if cand.name == "runs" else cand.name
                return InferenceCkptPlan(
                    mode="ensemble",
                    ckpt_paths=tuple(ens),
                    config_dir=fold_dir_for_config(cand, cv_fold=0),
                    label=f"{job_label}_cv{nf}_ensemble",
                )

    last_err: Exception | None = None
    for cand in run_candidates:
        if not cand.is_dir():
            continue
        if nf >= 2:
            fold0 = cand / "fold_0"
            if fold0.is_dir():
                try:
                    ckpt = pick_ckpt_in_dir(fold0, prefer)
                    return InferenceCkptPlan(
                        mode="single",
                        ckpt_paths=(ckpt,),
                        config_dir=fold_dir_for_config(cand, cv_fold=0),
                        label=f"{cand.name}_fold_0",
                    )
                except FileNotFoundError as e:
                    last_err = e
                    continue
        try:
            ckpt = pick_ckpt_in_dir(cand, prefer)
            cfg = cand / "runs" if (cand / "runs").is_dir() else cand
            return InferenceCkptPlan(
                mode="single",
                ckpt_paths=(ckpt,),
                config_dir=cfg,
                label=cand.name,
            )
        except FileNotFoundError as e:
            last_err = e
    tried = ", ".join(str(p) for p in run_candidates)
    raise FileNotFoundError(
        f"No usable checkpoint (best.pt / last.pt, prefer={prefer!r}). Tried: {tried}."
        " If weights are under a ztest5 grid, check TraMagNet/output/ztest5_TraMagNet_grid for task dirs."
        " Or use --job-name data3_4u_msestft_2_8_randz_e1000 and --runs-dir pointing at that job's runs/."
    ) from last_err


def add_cv_ensemble_arguments(parser: argparse.ArgumentParser) -> None:
    """Visualization / loss_eval: arithmetic-mean ensemble over full K-fold weights by default."""
    g = parser.add_argument_group("K-fold ensemble inference")
    g.add_argument(
        "--cv-ensemble-folds",
        type=int,
        default=5,
        dest="cv_ensemble_folds",
        help="Folds required for ensemble; enabled only when fold_0..fold_{K-1} under base_out all have weights.",
    )
    g.add_argument(
        "--no-cv-ensemble",
        action="store_true",
        dest="no_cv_ensemble",
        help="Disable K-fold averaging; use a single checkpoint (legacy behavior).",
    )


def cv_ensemble_folds_from_args(args, *, default: int = 5) -> int:
    if bool(getattr(args, "no_cv_ensemble", False)):
        return 0
    return max(2, int(getattr(args, "cv_ensemble_folds", default)))
