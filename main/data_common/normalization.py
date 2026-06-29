"""Signal preprocessing without reference-statistics leakage on noisy input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

NormalizationMode = Literal["noisy_sample", "none"]


@dataclass(frozen=True)
class NormalizationConfig:
    match_noisy_scale_to_reference: bool = False
    zscore_using_reference: bool = False
    normalization: NormalizationMode = "noisy_sample"
    eps: float = 1e-6


def mean_std(x: torch.Tensor, *, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    mu = x.mean()
    sig = x.std(unbiased=False).clamp_min(eps)
    return mu, sig


def affine_match_noisy_to_reference(
    noisy: torch.Tensor, reference: torch.Tensor, *, eps: float
) -> torch.Tensor:
    mu_c, sig_c = mean_std(reference, eps=eps)
    mu_n, sig_n = mean_std(noisy, eps=eps)
    return (noisy - mu_n) * (sig_c / sig_n) + mu_c


def zscore(x: torch.Tensor, mu: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
    return (x - mu) / sig


def normalize_pair(
    reference: torch.Tensor,
    noisy: torch.Tensor,
    cfg: NormalizationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply training/eval preprocessing. Default uses only noisy statistics."""
    reference_out = reference
    noisy_out = noisy

    if cfg.match_noisy_scale_to_reference:
        noisy_out = affine_match_noisy_to_reference(noisy_out, reference_out, eps=cfg.eps)

    if cfg.normalization == "noisy_sample":
        mu, sig = mean_std(noisy_out, eps=cfg.eps)
        reference_out = zscore(reference_out, mu, sig)
        noisy_out = zscore(noisy_out, mu, sig)
    elif cfg.zscore_using_reference:
        mu, sig = mean_std(reference_out, eps=cfg.eps)
        reference_out = zscore(reference_out, mu, sig)
        noisy_out = zscore(noisy_out, mu, sig)

    reference_out = torch.nan_to_num(reference_out, nan=0.0, posinf=0.0, neginf=0.0)
    noisy_out = torch.nan_to_num(noisy_out, nan=0.0, posinf=0.0, neginf=0.0)
    return reference_out, noisy_out


def config_from_dataset_flags(
    *,
    match_noisy_scale_to_reference: bool = False,
    zscore_using_reference: bool = False,
    normalization: NormalizationMode | str = "noisy_sample",
    eps: float = 1e-6,
) -> NormalizationConfig:
    mode: NormalizationMode = "noisy_sample"
    if str(normalization).strip().lower() in ("none", "off", "identity"):
        mode = "none"
    return NormalizationConfig(
        match_noisy_scale_to_reference=bool(match_noisy_scale_to_reference),
        zscore_using_reference=bool(zscore_using_reference),
        normalization=mode,
        eps=float(eps),
    )
