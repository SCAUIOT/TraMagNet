"""OurData Dataset merged across multiple ``data-root`` entries (split from pooled manifest or inline group-key list)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset

from data_common.our_data_split import infer_split_role, sample_ids_for_data_split
from data_common.pair_specs import list_pair_specs
from data_common.flat_pairing import axis_from_reference_path, sample_id_from_reference_path
from data_common.pooled_data_split import (
    group_key,
    parse_group_key,
    sort_group_keys,
)
from data_common.txt_io import pad_or_resample_to_length, read_one_file_with_meta


def _mean_std(x: torch.Tensor, *, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    mu = x.mean()
    sig = x.std(unbiased=False).clamp_min(eps)
    return mu, sig


def _affine_match_mean_std(noisy: torch.Tensor, clean: torch.Tensor, *, eps: float) -> torch.Tensor:
    mu_c, sig_c = _mean_std(clean, eps=eps)
    mu_n, sig_n = _mean_std(noisy, eps=eps)
    return (noisy - mu_n) * (sig_c / sig_n) + mu_c


def _zscore(x: torch.Tensor, mu: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
    return (x - mu) / sig


@dataclass(frozen=True)
class PooledOurDataConfig:
    """``root_entries``: ``[(tag, path), ...]``; split computed inline from ``seed`` / ``train_ratio``."""

    root_entries: tuple[tuple[str, Path], ...]

    reference_subdir: str = "reference_signal"
    noisy_subdir: str = "noise_signal"
    band: Literal["low", "middle", "high", "all"] = "all"
    strict_all_bands: bool = True

    train: bool = True
    train_ratio: float = 0.8
    seed: int = 42
    shuffle_split: bool = False
    split_round: bool = True
    cv_folds: int = 0
    cv_fold: int = 0
    holdout_eval: bool = False

    resample_mode: Literal["resample_linear", "pad_edge", "pad_zero"] = "resample_linear"
    match_noisy_scale_to_reference: bool = False
    zscore_using_reference: bool = False
    eps: float = 1e-6
    subway_dual_channels: bool = True
    segment_length: int = 1024


class PooledOurDataDataset(Dataset):
    """Merge multiple data roots; group keys ``{tag}/{sid}`` match ``pooled_data_split`` manifest."""

    def __init__(self, cfg: PooledOurDataConfig) -> None:
        super().__init__()
        self.cfg = cfg
        tag_order = [t for t, _ in cfg.root_entries]

        pairs: list[tuple[str, str, str, str, str, int]] = []
        for tag, root in cfg.root_entries:
            root = Path(root)
            strict = bool(cfg.strict_all_bands) if cfg.band == "all" else False
            specs = list_pair_specs(
                root,
                reference_subdir=cfg.reference_subdir,
                noisy_subdir=cfg.noisy_subdir,
                band=cfg.band,
                subway_dual_channels=cfg.subway_dual_channels,
                strict_all_bands=strict,
            )
            for sp in specs:
                sid = sample_id_from_reference_path(sp.reference_path, data_root=root)
                if not sid:
                    continue
                axis = axis_from_reference_path(sp.reference_path, data_root=root)
                pairs.append((tag, sid, axis, str(sp.reference_path), str(sp.noisy_path), int(sp.value_column)))

        if not pairs:
            raise RuntimeError("PooledOurDataDataset: no paired samples across roots")

        by_gk: dict[str, list[int]] = {}
        for i, (tag, sid, *_rest) in enumerate(pairs):
            by_gk.setdefault(group_key(tag, sid), []).append(i)

        gkeys_all = sort_group_keys(list(by_gk.keys()), tag_order=tag_order)

        role = infer_split_role(
            train=bool(cfg.train),
            cv_folds=int(cfg.cv_folds),
            holdout_eval=bool(cfg.holdout_eval),
        )

        chosen = set(
            sample_ids_for_data_split(
                gkeys_all,
                role=role,
                train_ratio=float(cfg.train_ratio),
                seed=int(cfg.seed),
                shuffle_split=bool(cfg.shuffle_split),
                split_round=bool(cfg.split_round),
                cv_folds=int(cfg.cv_folds),
                cv_fold=int(cfg.cv_fold),
            )
        )

        flat_indices: list[int] = []
        for gk in sort_group_keys([g for g in chosen if g in by_gk], tag_order=tag_order):
            flat_indices.extend(sorted(by_gk[gk]))
        if not flat_indices:
            raise RuntimeError("PooledOurDataDataset: split produced 0 samples")

        self._pairs = pairs
        self._indices = flat_indices
        self.n_total_pairs = len(self._pairs)
        self.n_split_pairs = len(self._indices)
        self.n_unique_groups = len(by_gk)
        # Compatible with ``OurDataDataset`` log fields (cross-dataset group-key count here)
        self.n_unique_samples = int(self.n_unique_groups)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        tag, sid, axis, c_fn, n_fn, vcol = self._pairs[self._indices[int(idx)]]

        c_s, _ = read_one_file_with_meta(c_fn, value_column=vcol)
        n_s, _ = read_one_file_with_meta(n_fn, value_column=vcol)
        c = c_s.value
        n = n_s.value

        L = min(len(c), len(n))
        if L < 2:
            raise RuntimeError(f"Too-short series: {c_fn} / {n_fn}")
        c = c[:L]
        n = n[:L]

        def _fix(y: list[float]) -> tuple[list[float], list[int]]:
            mode = self.cfg.resample_mode
            if len(y) > int(self.cfg.segment_length) and mode in ("pad_edge", "pad_zero"):
                mode = "resample_linear"
            return pad_or_resample_to_length(y, int(self.cfg.segment_length), mode=mode)

        c_r, c_mask = _fix(c)
        n_r, n_mask = _fix(n)

        clean = torch.tensor(c_r, dtype=torch.float32).unsqueeze(0)
        noisy = torch.tensor(n_r, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor([1 if (a and b) else 0 for a, b in zip(c_mask, n_mask)], dtype=torch.int64)

        if self.cfg.match_noisy_scale_to_reference:
            noisy = _affine_match_mean_std(noisy, clean, eps=float(self.cfg.eps))

        reference_phys = clean.clone()

        if self.cfg.zscore_using_reference:
            mu, sig = _mean_std(clean, eps=float(self.cfg.eps))
            clean = _zscore(clean, mu, sig)
            noisy = _zscore(noisy, mu, sig)

        clean = torch.nan_to_num(clean, nan=0.0, posinf=0.0, neginf=0.0)
        noisy = torch.nan_to_num(noisy, nan=0.0, posinf=0.0, neginf=0.0)
        reference_phys = torch.nan_to_num(reference_phys, nan=0.0, posinf=0.0, neginf=0.0)

        key = f"{tag}/{Path(n_fn).stem}"
        if vcol != 2:
            key = f"{key}__vcol{vcol}"
        return {"reference": clean, "noisy": noisy, "reference_phys": reference_phys, "key": key, "mask": mask}
