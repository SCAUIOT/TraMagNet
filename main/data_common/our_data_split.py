"""
Shared our_data train/test split for all models: partition by sample id, same convention as ``OurDataDataset``.

- Fixed **8:2** (``train_ratio``) yields train pool / holdout test set (reproducible via ``seed`` + ``shuffle_split``).
- When ``cv_folds > 0``, K-fold on the train pool: ``train=True`` → current fold train, ``train=False`` → current fold val;
  holdout test set via ``holdout_eval=True`` or ``split`` of ``holdout`` / ``test_holdout``.

Shuffling uses ``torch.Generator`` + ``torch.randperm`` (same as training datasets).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch

SplitSide = Literal["train", "test"]
DataSplitRole = Literal["train", "test", "holdout_test", "cv_train", "cv_val"]


def shuffle_group_indices(n_groups: int, *, shuffle_split: bool, seed: int) -> list[int]:
    """Return a length-``n_groups`` group index order: ``0..n-1`` when not shuffled, else ``torch.randperm``."""
    if n_groups <= 0:
        return []
    if not shuffle_split:
        return list(range(n_groups))
    g = torch.Generator()
    g.manual_seed(int(seed))
    return torch.randperm(n_groups, generator=g).tolist()


def train_group_split_index(n_groups: int, train_ratio: float, *, split_round: bool = True) -> int:
    """Number of groups ``k`` in the train pool, with ``1 <= k <= n_groups - 1`` when ``n_groups >= 2``."""
    if n_groups <= 1:
        return 0
    raw = n_groups * float(train_ratio)
    k = int(round(raw)) if split_round else int(raw)
    return max(1, min(n_groups - 1, k))


def infer_split_role(
    *,
    train: bool,
    cv_folds: int = 0,
    holdout_eval: bool = False,
) -> DataSplitRole:
    """Infer split role from ``train`` / ``cv_folds`` / ``holdout_eval``."""
    if holdout_eval:
        return "holdout_test"
    if int(cv_folds) <= 0:
        return "train" if train else "test"
    return "cv_train" if train else "cv_val"


def partition_group_indices_for_role(
    n_groups: int,
    *,
    role: DataSplitRole,
    train_ratio: float = 0.8,
    seed: int = 42,
    shuffle_split: bool = True,
    split_round: bool = True,
    cv_folds: int = 0,
    cv_fold: int = 0,
) -> list[int]:
    """
    Partition in group index space ``0 .. n_groups-1``; return **original group indices** for this role.

    When ``role`` is ``train``/``test`` and ``cv_folds==0``, equivalent to legacy full 8:2 split;
    when ``cv_folds>0``, ``cv_train``/``cv_val`` are K-fold subsets within the train pool; ``holdout_test`` is the fixed holdout set.
    """
    if n_groups <= 0:
        return []
    if n_groups == 1:
        if role in ("train", "cv_train"):
            return [0]
        return []

    order = shuffle_group_indices(n_groups, shuffle_split=shuffle_split, seed=int(seed))
    k_pool = train_group_split_index(n_groups, train_ratio, split_round=split_round)
    pool_pos = list(range(k_pool))
    holdout_pos = list(range(k_pool, n_groups))

    def _groups_at(positions: list[int]) -> list[int]:
        return [int(order[i]) for i in positions]

    pool_groups = _groups_at(pool_pos)
    holdout_groups = _groups_at(holdout_pos)

    if role in ("test", "holdout_test"):
        return holdout_groups
    if role == "train" and int(cv_folds) <= 0:
        return pool_groups

    n_folds = int(cv_folds)
    if n_folds < 2:
        raise ValueError(f"cv_folds must be >= 2 when using CV roles, got {cv_folds}")
    fold_i = int(cv_fold) % n_folds
    m = len(pool_groups)
    if m == 0:
        return []
    perm = shuffle_group_indices(m, shuffle_split=shuffle_split, seed=int(seed))
    val_groups: list[int] = []
    tr_groups: list[int] = []
    for pos in range(m):
        gi = pool_groups[int(perm[pos])]
        if pos % n_folds == fold_i:
            val_groups.append(gi)
        else:
            tr_groups.append(gi)
    if role == "cv_val":
        return val_groups
    if role == "cv_train":
        return tr_groups
    raise ValueError(f"unsupported split role: {role!r}")


def sample_ids_for_data_split(
    sids: list[str],
    *,
    role: DataSplitRole,
    train_ratio: float = 0.8,
    seed: int = 42,
    shuffle_split: bool = True,
    split_round: bool = True,
    cv_folds: int = 0,
    cv_fold: int = 0,
) -> list[str]:
    """``sids`` must be sorted per training rules; return sample ids on the ``role`` side."""
    if not sids:
        return []
    n = len(sids)
    if n == 1:
        return list(sids) if role in ("train", "cv_train") else []

    chosen_idx = partition_group_indices_for_role(
        n,
        role=role,
        train_ratio=train_ratio,
        seed=seed,
        shuffle_split=shuffle_split,
        split_round=split_round,
        cv_folds=cv_folds,
        cv_fold=cv_fold,
    )
    return [sids[i] for i in chosen_idx]


def ordered_sample_ids_for_train_test_split(
    sids: list[str],
    *,
    split: SplitSide,
    train_ratio: float,
    seed: int,
    shuffle_split: bool,
    split_round: bool = True,
    cv_folds: int = 0,
    cv_fold: int = 0,
) -> list[str]:
    """
    Legacy API: ``split='train'|'test'``.

    - ``test`` is always the fixed holdout set (20%), regardless of CV (for visualization / final eval).
    - ``train`` with ``cv_folds>0`` is the current CV train subset; otherwise the full 80% train pool.
    """
    if split == "test":
        role: DataSplitRole = "holdout_test"
    else:
        role = "cv_train" if int(cv_folds) > 0 else "train"
    return sample_ids_for_data_split(
        sids,
        role=role,
        train_ratio=train_ratio,
        seed=seed,
        shuffle_split=shuffle_split,
        split_round=split_round,
        cv_folds=cv_folds,
        cv_fold=cv_fold,
    )


def build_cv_split_manifest(
    sids: list[str],
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
    shuffle_split: bool = True,
    split_round: bool = True,
    cv_folds: int = 5,
) -> dict:
    """Build a JSON-serializable split manifest (sample id level)."""
    sids = list(sids)
    n = len(sids)
    order = shuffle_group_indices(n, shuffle_split=shuffle_split, seed=int(seed))
    k_pool = train_group_split_index(n, train_ratio, split_round=split_round) if n >= 2 else n
    pool_ids = [sids[order[i]] for i in range(k_pool)]
    holdout_ids = [sids[order[i]] for i in range(k_pool, n)]
    folds: list[dict] = []
    nf = max(0, int(cv_folds))
    for fi in range(nf):
        tr = sample_ids_for_data_split(
            sids,
            role="cv_train",
            train_ratio=train_ratio,
            seed=seed,
            shuffle_split=shuffle_split,
            split_round=split_round,
            cv_folds=nf,
            cv_fold=fi,
        )
        va = sample_ids_for_data_split(
            sids,
            role="cv_val",
            train_ratio=train_ratio,
            seed=seed,
            shuffle_split=shuffle_split,
            split_round=split_round,
            cv_folds=nf,
            cv_fold=fi,
        )
        folds.append({"fold": fi, "train_sample_ids": tr, "val_sample_ids": va})
    return {
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "shuffle_split": bool(shuffle_split),
        "split_round": bool(split_round),
        "cv_folds": nf,
        "n_unique_samples": n,
        "train_pool_sample_ids": pool_ids,
        "holdout_test_sample_ids": holdout_ids,
        "folds": folds,
    }


def write_cv_split_manifest(path: str | Path, manifest: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
