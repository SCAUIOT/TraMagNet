from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_common.our_data_split import infer_split_role, sample_ids_for_data_split
from data_common.pair_specs import list_pair_specs
from data_common.normalization import config_from_dataset_flags, normalize_pair
from data_common.flat_pairing import axis_from_reference_path, sample_id_from_reference_path
from data_common.txt_io import pad_or_resample_to_length, read_one_file_with_meta


@dataclass(frozen=True)
class OurDataConfig:
    """
    Data directory under ``public/datasets/``; pairing via matching ``sample{i}.txt`` names.
    """

    root: str | Path = "."
    reference_subdir: str = "reference_signal"
    noisy_subdir: str = "noisy_signal"

    band: Literal["low", "middle", "high", "all"] = "low"
    segment_length: int = 1500
    # When band="all": require low/middle/high all exist for each (sample_id, axis).
    strict_all_bands: bool = True

    # split
    train: bool = True
    train_ratio: float = 0.8
    seed: int = 42
    shuffle_split: bool = False
    split_round: bool = True
    cv_folds: int = 0
    cv_fold: int = 0
    holdout_eval: bool = False

    # preprocess
    resample_mode: Literal["resample_linear", "pad_edge", "pad_zero"] = "resample_linear"
    # Affine: align segment mean/std of noisy to reference (remove global gain/bias); does not guarantee pointwise noisy≈reference.
    match_noisy_scale_to_reference: bool = False
    zscore_using_reference: bool = False
    normalization: Literal["noisy_sample", "none"] = "noisy_sample"
    eps: float = 1e-6
    subway_dual_channels: bool = True


def _mean_std(x: torch.Tensor, *, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    mu = x.mean()
    sig = x.std(unbiased=False).clamp_min(eps)
    return mu, sig


def _affine_match_mean_std(noisy: torch.Tensor, reference: torch.Tensor, *, eps: float) -> torch.Tensor:
    mu_c, sig_c = _mean_std(reference, eps=eps)
    mu_n, sig_n = _mean_std(noisy, eps=eps)
    return (noisy - mu_n) * (sig_c / sig_n) + mu_c


def _zscore(x: torch.Tensor, mu: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
    return (x - mu) / sig


class OurDataDataset(Dataset):
    """
    Output item:
      - dict(reference=(1,T), noisy=(1,T), key=str, mask=(T,))
    """

    def __init__(self, cfg: OurDataConfig) -> None:
        super().__init__()
        self.cfg = cfg
        root = Path(cfg.root)
        reference_dir = root / cfg.reference_subdir
        noisy_dir = root / cfg.noisy_subdir
        if not reference_dir.is_dir():
            raise FileNotFoundError(f"reference_dir not found: {reference_dir}")
        if not noisy_dir.is_dir():
            raise FileNotFoundError(f"noisy_dir not found: {noisy_dir}")

        strict = bool(cfg.strict_all_bands) if cfg.band == "all" else False
        specs = list_pair_specs(
            root,
            reference_subdir=cfg.reference_subdir,
            noisy_subdir=cfg.noisy_subdir,
            band=cfg.band,
            subway_dual_channels=cfg.subway_dual_channels,
            strict_all_bands=strict,
        )
        pairs: list[tuple[str, str, str, str, int]] = []
        for sp in specs:
            sid = sample_id_from_reference_path(sp.reference_path, data_root=root)
            if not sid:
                continue
            axis = axis_from_reference_path(sp.reference_path, data_root=root)
            pairs.append((sid, axis, str(sp.reference_path), str(sp.noisy_path), int(sp.value_column)))

        if not pairs:
            raise RuntimeError(
                f"No paired samples under {reference_dir} / {noisy_dir} for band={cfg.band!r}. "
                f"Ensure matching sample{{i}}.txt exist in both reference_signal/ and noise_signal/."
            )

        by_sid: dict[str, list[int]] = {}
        for i, (sid, _axis, _c, _n, _v) in enumerate(pairs):
            by_sid.setdefault(sid, []).append(i)
        sids = sorted(by_sid.keys(), key=lambda s: int(s))

        role = infer_split_role(
            train=bool(cfg.train),
            cv_folds=int(cfg.cv_folds),
            holdout_eval=bool(cfg.holdout_eval),
        )
        chosen = sample_ids_for_data_split(
            sids,
            role=role,
            train_ratio=float(cfg.train_ratio),
            seed=int(cfg.seed),
            shuffle_split=bool(cfg.shuffle_split),
            split_round=bool(cfg.split_round),
            cv_folds=int(cfg.cv_folds),
            cv_fold=int(cfg.cv_fold),
        )

        flat_indices: list[int] = []
        for sid in chosen:
            flat_indices.extend(sorted(by_sid[sid]))
        if not flat_indices:
            raise RuntimeError("Split produced 0 samples; adjust train_ratio or provide more data.")

        self._pairs = pairs
        self._indices = flat_indices

        self.n_total_pairs = len(self._pairs)
        self.n_split_pairs = len(self._indices)
        self.n_unique_samples = len(sids)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sid, axis, c_fn, n_fn, vcol = self._pairs[self._indices[int(idx)]]

        c_s, _ = read_one_file_with_meta(c_fn, value_column=vcol)
        n_s, _ = read_one_file_with_meta(n_fn, value_column=vcol)
        c = c_s.value
        n = n_s.value

        L = min(len(c), len(n))
        if L < 2:
            raise RuntimeError(f"Too-short series for interpolation: {c_fn} / {n_fn}")
        c = c[:L]
        n = n[:L]

        def _fix(y: list[float]) -> tuple[list[float], list[int]]:
            mode = self.cfg.resample_mode
            if len(y) > int(self.cfg.segment_length) and mode in ("pad_edge", "pad_zero"):
                mode = "resample_linear"
            return pad_or_resample_to_length(y, int(self.cfg.segment_length), mode=mode)

        c_r, c_mask = _fix(c)
        n_r, n_mask = _fix(n)

        reference = torch.tensor(c_r, dtype=torch.float32).unsqueeze(0)
        noisy = torch.tensor(n_r, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor([1 if (a and b) else 0 for a, b in zip(c_mask, n_mask)], dtype=torch.int64)

        norm_cfg = config_from_dataset_flags(
            match_noisy_scale_to_reference=bool(self.cfg.match_noisy_scale_to_reference),
            zscore_using_reference=bool(self.cfg.zscore_using_reference),
            normalization=self.cfg.normalization,
            eps=float(self.cfg.eps),
        )
        reference, noisy = normalize_pair(reference, noisy, norm_cfg)

        key = Path(n_fn).stem
        if vcol != 2:
            key = f"{key}__vcol{vcol}"
        return {"reference": reference, "noisy": noisy, "key": key, "mask": mask}
