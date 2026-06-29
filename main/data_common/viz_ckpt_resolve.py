"""Shared checkpoint / K-fold ensemble resolution for viz runners."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_common.cv_ensemble import (
    InferenceCkptPlan,
    cv_ensemble_folds_from_args,
    resolve_inference_ckpts,
)
from data_common.viz_export_workers import checkpoint_run_candidates
from data_common.ztest5_paths import extend_run_candidates_with_ztest5, normalize_cv_runs_base


def resolve_viz_inference_plan(
    args: argparse.Namespace,
    *,
    repo: Path,
    data_root: str,
    data_tag: str,
    nn_dir: Path | None,
) -> InferenceCkptPlan:
    runs_dir_arg = getattr(args, "runs_dir", None)
    runs_dir: Path | None = None
    ckpt_prefer = str(getattr(args, "ckpt", "last"))
    job_name = getattr(args, "job_name", None)
    if job_name is not None:
        job_name = str(job_name).strip() or None

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
    candidates = extend_run_candidates_with_ztest5(
        candidates,
        repo=repo,
        data_tag=data_tag,
        nn_dir=nn_dir,
        job_name=job_name,
        cv_folds=cv_folds,
        ckpt_prefer=ckpt_prefer,
    )
    return resolve_inference_ckpts(
        repo=repo,
        data_root=data_root,
        runs_dir=runs_dir,
        run_candidates=candidates,
        ckpt_prefer=ckpt_prefer,
        cv_folds=cv_folds,
        use_ensemble=use_ensemble,
    )


def add_viz_job_arguments(parser: argparse.ArgumentParser) -> None:
    """ztest5 grid job dirs (``data3_4u_msestft_2_8_randz_e1000``, etc.)."""
    g = parser.add_argument_group("ztest5 / job directory")
    g.add_argument(
        "--job-name",
        type=str,
        default=None,
        dest="job_name",
        metavar="DIR",
        help="ztest5 job dir name (e.g. data3_4u_msestft_2_8_randz_e1000); if omitted, auto-pick weighted job with largest _e in grid.",
    )
