"""
TraMagNet supervised losses:

- ``mse_time`` / ``time``: divide MSE, masked L1, and MR-STFT by **fixed feature scales** (unified units, batch-independent),
  then weighted sum using normalized ``loss_*_weight`` proportions.
- No longer uses ``L / stop_grad(L)`` (each term ≈1, total ≈1, logs cannot reflect quality).

Feature scales target z-score space, segment length ~1024, typical our_data magnitudes; adjust
``LOSS_TERM_SCALE_*`` constants if the data distribution changes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# (n_fft, win_length, hop_length) — window lengths 64/128/256, balancing local high-frequency detail and longer-window spectral resolution
MR_STFT_SCALES: tuple[tuple[int, int, int], ...] = (
    (64, 64, 16),
    (128, 128, 32),
    (256, 256, 64),
)

_LOSS_UNIT_EPS = 1e-8

# Divide raw scalar losses by these scales to get O(1), then multiply by mix weights (typical data3 / z-score training values)
LOSS_TERM_SCALE_MSE: float = 0.07
LOSS_TERM_SCALE_L1: float = 0.20
LOSS_TERM_SCALE_STFT: float = 0.33


def masked_l1_mean(pred: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Pointwise absolute error in time domain; if ``mask`` is present (consistent with ``OurDataDataset``), weighted mean over valid positions."""
    diff = (pred - reference).abs()
    if mask is None:
        return diff.mean()
    m = mask.to(pred.device).float()
    if m.dim() == 2:
        m = m.unsqueeze(1)
    return (diff * m).sum() / m.sum().clamp_min(_LOSS_UNIT_EPS)


def multi_scale_stft_mag_l1(pred: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    ``pred`` / ``reference``: (B, 1, T). At each scale, per-bin L1 on ``|STFT(x)|`` vs ``|STFT(y)|``, then averaged over scales.
    """
    x = pred.squeeze(1)
    y = reference.squeeze(1)
    acc: torch.Tensor | None = None
    for n_fft, win_length, hop in MR_STFT_SCALES:
        w = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
        X = torch.stft(
            x,
            n_fft,
            hop_length=hop,
            win_length=win_length,
            window=w,
            center=True,
            return_complex=True,
        )
        Y = torch.stft(
            y,
            n_fft,
            hop_length=hop,
            win_length=win_length,
            window=w,
            center=True,
            return_complex=True,
        )
        term = (X.abs() - Y.abs()).abs().mean()
        acc = term if acc is None else acc + term
    assert acc is not None
    return acc / float(len(MR_STFT_SCALES))


def loss_to_unit_scale(loss: torch.Tensor, *, eps: float = _LOSS_UNIT_EPS) -> torch.Tensor:
    """
    Legacy ``L / stop_grad(L)`` (deprecated for mixed losses; kept for compatibility with old scripts).

    For new training, use ``scale_loss_term`` + ``combine_normalized_losses``.
    """
    return loss / loss.detach().clamp_min(eps)


def scale_loss_term(loss: torch.Tensor, kind: str) -> torch.Tensor:
    """Divide a raw loss term by a fixed feature scale so MSE / L1 / STFT share comparable units."""
    kind = str(kind).lower().strip()
    if kind == "mse":
        s = LOSS_TERM_SCALE_MSE
    elif kind == "l1":
        s = LOSS_TERM_SCALE_L1
    elif kind == "stft":
        s = LOSS_TERM_SCALE_STFT
    else:
        raise ValueError(f"unknown loss term kind: {kind!r}")
    return loss / float(s)


def resolve_mix_weights(
    w_mse: float,
    w_l1: float,
    w_stft: float,
    *,
    eps: float = 1e-12,
) -> tuple[float, float, float]:
    """
    Normalize non-negative mix weights to sum to 1; raise ``ValueError`` if all zero.
    Weights need not already sum to 1 (e.g. ``0.4, 0.6`` is equivalent to ``4, 6``).
    """
    a = max(0.0, float(w_mse))
    b = max(0.0, float(w_l1))
    c = max(0.0, float(w_stft))
    s = a + b + c
    if s <= eps:
        raise ValueError("At least one loss mix weight must be > 0 (--loss-mse-weight / --loss-l1-weight / --loss-stft-weight)")
    return a / s, b / s, c / s


def mix_score_from_raw(
    l_mse: float,
    l_l1: float,
    l_stft: float,
    *,
    w_mse: float,
    w_l1: float,
    w_stft: float,
) -> float:
    """Same formula as ``combine_normalized_losses``, for logging / best selection (scalar, no grad)."""
    wm, wl, ws = resolve_mix_weights(w_mse, w_l1, w_stft)
    score = 0.0
    if wm > 0.0:
        score += wm * float(l_mse) / LOSS_TERM_SCALE_MSE
    if wl > 0.0:
        score += wl * float(l_l1) / LOSS_TERM_SCALE_L1
    if ws > 0.0:
        score += ws * float(l_stft) / LOSS_TERM_SCALE_STFT
    return float(score)


def combine_normalized_losses(
    *,
    l_mse: torch.Tensor | None,
    l_l1: torch.Tensor | None,
    l_stft: torch.Tensor | None,
    w_mse: float,
    w_l1: float,
    w_stft: float,
) -> torch.Tensor:
    """After fixed-scale unit alignment, weighted sum using normalized weights (backprop-friendly)."""
    wm, wl, ws = resolve_mix_weights(w_mse, w_l1, w_stft)
    ref = l_mse if l_mse is not None else (l_l1 if l_l1 is not None else l_stft)
    if ref is None:
        raise ValueError("combine_normalized_losses: no loss terms")
    total = ref.new_zeros(())
    if l_mse is not None and wm > 0.0:
        total = total + wm * scale_loss_term(l_mse, "mse")
    if l_l1 is not None and wl > 0.0:
        total = total + wl * scale_loss_term(l_l1, "l1")
    if l_stft is not None and ws > 0.0:
        total = total + ws * scale_loss_term(l_stft, "stft")
    return total


def supervised_unet_loss(
    pred: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    loss_l1_weight: float,
    loss_stft_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns ``(total, l_l1, l_stft)`` (``l_*`` are **unnormalized** raw values for logging).
    STFT is skipped when ``loss_stft_weight == 0``.
    """
    _, _, ws = resolve_mix_weights(0.0, loss_l1_weight, loss_stft_weight)
    l_l1 = masked_l1_mean(pred, reference, mask)
    if ws > 0.0:
        l_stft = multi_scale_stft_mag_l1(pred, reference)
        total = combine_normalized_losses(
            l_mse=None,
            l_l1=l_l1,
            l_stft=l_stft,
            w_mse=0.0,
            w_l1=loss_l1_weight,
            w_stft=loss_stft_weight,
        )
    else:
        l_stft = l_l1.new_zeros(())
        total = combine_normalized_losses(
            l_mse=None,
            l_l1=l_l1,
            l_stft=None,
            w_mse=0.0,
            w_l1=loss_l1_weight,
            w_stft=0.0,
        )
    return total, l_l1, l_stft


def mse_time_frequency_loss(
    pred: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    loss_mse_weight: float,
    loss_l1_weight: float,
    loss_stft_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns ``(total, l_mse, l_l1, l_stft)``; ``l_*`` are unnormalized raw values.
    ``total`` = fixed-scale aligned weighted mix (weights as 0~1 proportions).
    """
    wm, wl, ws = resolve_mix_weights(loss_mse_weight, loss_l1_weight, loss_stft_weight)
    l_mse = F.mse_loss(pred, reference)
    l_l1 = masked_l1_mean(pred, reference, mask)
    if ws > 0.0:
        l_stft = multi_scale_stft_mag_l1(pred, reference)
    else:
        l_stft = l_l1.new_zeros(())
    total = combine_normalized_losses(
        l_mse=l_mse if wm > 0.0 else None,
        l_l1=l_l1 if wl > 0.0 else None,
        l_stft=l_stft if ws > 0.0 else None,
        w_mse=loss_mse_weight,
        w_l1=loss_l1_weight,
        w_stft=loss_stft_weight,
    )
    return total, l_mse, l_l1, l_stft
