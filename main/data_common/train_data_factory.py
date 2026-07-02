"""Shared training helpers: single-root ``OurDataDataset`` or multi-root ``PooledOurDataDataset``."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from data_common.pooled_data_split import parse_data_roots_arg
from data_common.pooled_our_data_dataset import PooledOurDataConfig, PooledOurDataDataset


def training_uses_pooled_data(args: object) -> bool:
    dr = getattr(args, "data_roots", None)
    return bool(dr and str(dr).strip())


def make_supervised_dataset(
    args: argparse.Namespace,
    cfg: Any,
    *,
    repo: Path,
    train: bool,
    our_data_config_cls: type,
    our_data_dataset_cls: type,
) -> PooledOurDataDataset | Any:
    """``cfg`` must include reference_subdir, band, cv_folds, etc., same fields as ``TrainCfg`` / ``Gan5Cfg``."""
    if not training_uses_pooled_data(args):
        return our_data_dataset_cls(
            our_data_config_cls(
                root=cfg.root,
                reference_subdir=cfg.reference_subdir,
                noisy_subdir=cfg.noisy_subdir,
                band=cfg.band,
                segment_length=int(cfg.segment_length),
                train=bool(train),
                train_ratio=float(cfg.train_ratio),
                seed=int(cfg.seed),
                shuffle_split=bool(cfg.shuffle_split),
                cv_folds=int(cfg.cv_folds),
                cv_fold=int(cfg.cv_fold),
                split_round=True,
                resample_mode=cfg.resample_mode,
                strict_all_bands=bool(cfg.strict_all_bands),
                subway_dual_channels=bool(cfg.subway_dual_channels),
                match_noisy_scale_to_reference=bool(cfg.match_noisy_scale),
                zscore_using_reference=bool(cfg.zscore_using_reference),
            )
        )

    entries = parse_data_roots_arg(str(args.data_roots), repo=repo)
    pcfg = PooledOurDataConfig(
        root_entries=tuple(entries),
        reference_subdir=cfg.reference_subdir,
        noisy_subdir=cfg.noisy_subdir,
        band=cfg.band,
        segment_length=int(cfg.segment_length),
        train=bool(train),
        train_ratio=float(cfg.train_ratio),
        seed=int(cfg.seed),
        shuffle_split=bool(cfg.shuffle_split),
        cv_folds=int(cfg.cv_folds),
        cv_fold=int(cfg.cv_fold),
        split_round=True,
        resample_mode=cfg.resample_mode,
        strict_all_bands=bool(cfg.strict_all_bands),
        subway_dual_channels=bool(cfg.subway_dual_channels),
        match_noisy_scale_to_reference=bool(cfg.match_noisy_scale),
        zscore_using_reference=bool(cfg.zscore_using_reference),
    )
    return PooledOurDataDataset(pcfg)
