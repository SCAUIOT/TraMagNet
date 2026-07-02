"""Shared checkpoint / K-fold ensemble resolution for viz runners."""

from __future__ import annotations

from pathlib import Path

from data_common.cv_ensemble import (
    InferenceCkptPlan,
    cv_ensemble_folds_from_args,
    resolve_inference_ckpts,
    runs_base_has_any_ckpt,
)
from data_common.viz_export_workers import checkpoint_run_candidates


def normalize_cv_runs_base(path: Path) -> Path:
    """
    Normalize user path to K-fold weight root (``runs`` dir with ``fold_*`` or single ``best.pt``).

    - ``…/<pool>/runs`` → unchanged when it already contains ``fold_0``
    - ``…/<pool>`` → ``…/<pool>/runs`` when ``runs/fold_0`` exists
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


def resolve_viz_inference_plan(
    args,
    *,
    repo: Path,
    data_root: str,
    data_tag: str,
    nn_dir: Path | None,
) -> InferenceCkptPlan:
    runs_dir_arg = getattr(args, "runs_dir", None)
    runs_dir: Path | None = None
    ckpt_prefer = str(getattr(args, "ckpt", "last"))

    if runs_dir_arg:
        runs_dir = Path(str(runs_dir_arg))
        if not runs_dir.is_absolute():
            trial = (repo / runs_dir).resolve()
            runs_dir = trial if trial.exists() else runs_dir.resolve()
        runs_dir = normalize_cv_runs_base(runs_dir)

    nf = cv_ensemble_folds_from_args(args)
    use_ensemble = nf >= 2
    cv_folds = nf if use_ensemble else 5

    candidates = checkpoint_run_candidates(repo=repo, data_tag=data_tag, nn_dir=nn_dir)
    return resolve_inference_ckpts(
        repo=repo,
        data_root=data_root,
        runs_dir=runs_dir,
        run_candidates=candidates,
        ckpt_prefer=ckpt_prefer,
        cv_folds=cv_folds,
        use_ensemble=use_ensemble,
    )
