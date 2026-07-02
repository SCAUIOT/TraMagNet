from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DnCNN1DConfig:
    in_channels: int = 1
    out_channels: int = 1
    features: int = 64
    kernel_size: int = 3
    padding: int = 1
    use_bn: bool = True
    predict_residual: bool = True
    middle_depth: int = 10
    num_residual_blocks: int = 5
    # Off by default: raw x*sigmoid gating can squash features toward 0, yielding pred_noise≈0 and output≈noisy; use --use-attention when needed
    use_attention: bool = False
    attention_reduction: int = 8


class ResidualBlock1D(nn.Module):
    """Conv → BN → ReLU → Conv → BN, add input, then ReLU."""

    def __init__(
        self,
        features: int,
        *,
        kernel_size: int,
        padding: int,
        use_bn: bool,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            features, features, kernel_size=kernel_size, stride=1, padding=padding, bias=not use_bn
        )
        self.bn1 = nn.BatchNorm1d(features) if use_bn else nn.Identity()
        self.conv2 = nn.Conv1d(
            features, features, kernel_size=kernel_size, stride=1, padding=padding, bias=not use_bn
        )
        self.bn2 = nn.BatchNorm1d(features) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(x + out)


class ChannelAttention1D(nn.Module):
    """Channel attention: global average pool over time, then MLP per-channel weights.

    Gating is ``0.5 + 0.5 * sigmoid(·)`` ∈ [0.5, 1], avoiding pure ``x * sigmoid`` that drives
    channels toward 0, leaving later convs with almost no signal and pred_noise stuck at 0
    (denoised output degenerates to the noisy input).
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        s = x.mean(dim=2)
        g = self.fc(s).view(b, c, 1)
        scale = 0.5 + 0.5 * g
        return x * scale


class TemporalAttention1D(nn.Module):
    """Temporal attention: 1×1 conv per time step; gating likewise clamped to [0.5, 1]."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(1, channels // 4)
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.net(x)
        scale = 0.5 + 0.5 * g
        return x * scale


class ChannelTemporalAttention1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.channel_att = ChannelAttention1D(channels, reduction)
        self.temporal_att = TemporalAttention1D(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.temporal_att(x)
        return x


class DnCNN1D(nn.Module):
    """
    1D denoising network: Head Conv+ReLU → [Conv+BN+ReLU]×middle_depth → Residual×num_residual_blocks
    → channel+temporal Attention → final Conv predicts noise; forward ``clean = x - noise``.
    """

    def __init__(self, cfg: DnCNN1DConfig = DnCNN1DConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        k = int(cfg.kernel_size)
        p = int(cfg.padding)
        f = int(cfg.features)
        self.net = None
        self.head = nn.Sequential(
            nn.Conv1d(cfg.in_channels, f, kernel_size=k, stride=1, padding=p, bias=True),
            nn.ReLU(inplace=True),
        )
        mid: list[nn.Module] = []
        for _ in range(max(1, int(cfg.middle_depth))):
            mid.append(nn.Conv1d(f, f, kernel_size=k, stride=1, padding=p, bias=not cfg.use_bn))
            if cfg.use_bn:
                mid.append(nn.BatchNorm1d(f))
            mid.append(nn.ReLU(inplace=True))
        self.middle = nn.Sequential(*mid)
        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock1D(f, kernel_size=k, padding=p, use_bn=cfg.use_bn)
                for _ in range(max(1, int(cfg.num_residual_blocks)))
            ]
        )
        self.attn = (
            ChannelTemporalAttention1D(f, int(cfg.attention_reduction))
            if cfg.use_attention
            else nn.Identity()
        )
        self.tail = nn.Conv1d(f, cfg.out_channels, kernel_size=k, stride=1, padding=p, bias=True)

    def predict_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Predict additive noise (residual); same quantity subtracted in ``forward``."""
        assert self.head is not None and self.tail is not None
        feat = self.head(x)
        feat = self.middle(feat)
        feat = self.res_blocks(feat)
        feat = self.attn(feat)
        return self.tail(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.predict_noise(x)
        if self.cfg.predict_residual:
            return x[:, : self.cfg.out_channels, :] - pred
        return pred


def dncnn_config_from_argparse(args: argparse.Namespace) -> DnCNN1DConfig:
    """Shared by train / infer / viz: requires features, middle_depth, num_residual, attention, attention_reduction."""
    ua = bool(getattr(args, "use_attention", False))
    na = bool(getattr(args, "no_attention", False))
    if ua and na:
        raise ValueError("Cannot specify both --use-attention and --no-attention")
    use_attention = ua and not na
    return DnCNN1DConfig(
        features=int(args.features),
        middle_depth=int(getattr(args, "middle_depth", 10)),
        num_residual_blocks=int(getattr(args, "num_residual", 5)),
        use_attention=use_attention,
        attention_reduction=int(getattr(args, "attention_reduction", 8)),
    )


def _mask_like_pred(mask: torch.Tensor | None, pred: torch.Tensor, *, eps: float) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    m = mask.to(dtype=pred.dtype, device=pred.device).unsqueeze(1)
    return m


def masked_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, *, eps: float = 1e-8
) -> torch.Tensor:
    """
    pred/target: (B, C, T)
    mask: (B, T) or (T,) with 1 for valid, 0 for padded.
    """
    if mask is None:
        return torch.mean((pred - target) ** 2)

    m = _mask_like_pred(mask, pred, eps=eps)
    err = (pred - target) ** 2
    num = (err * m).sum()
    den = m.sum().clamp_min(eps)
    return num / den


def masked_l1(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, *, eps: float = 1e-8
) -> torch.Tensor:
    err = (pred - target).abs()
    if mask is None:
        return err.mean()
    m = _mask_like_pred(mask, pred, eps=eps)
    return (err * m).sum() / m.sum().clamp_min(eps)


def masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    beta: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Smooth L1 (Huber); smaller beta is closer to L1 and less prone to squaring large errors vs pure MSE."""
    err = F.smooth_l1_loss(pred, target, beta=float(beta), reduction="none")
    if mask is None:
        return err.mean()
    m = _mask_like_pred(mask, pred, eps=eps)
    return (err * m).sum() / m.sum().clamp_min(eps)


def masked_temporal_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """First-order temporal difference MSE for aligning peaks/slopes (pred/target shape (B,C,T))."""
    dp = pred[..., 1:] - pred[..., :-1]
    dt = target[..., 1:] - target[..., :-1]
    if mask is None:
        return torch.mean((dp - dt) ** 2)
    mb = mask
    if mb.ndim == 1:
        mb = mb.unsqueeze(0)
    mb = mb.to(dtype=pred.dtype, device=pred.device)
    m = (mb[:, 1:] * mb[:, :-1]).unsqueeze(1)
    err = (dp - dt) ** 2
    return (err * m).sum() / m.sum().clamp_min(eps)


def masked_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    kind: str = "mse",
    huber_beta: float = 0.1,
    l1_aux_weight: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    kind:
      - ``mse``: pure MSE; if ``l1_aux_weight>0``, ``MSE + w * L1`` (emphasizes MAE, often stricter).
      - ``l1``: pure MAE.
      - ``huber``: SmoothL1.
    """
    k = kind.lower().strip()
    if k == "mse":
        out = masked_mse(pred, target, mask, eps=eps)
        if l1_aux_weight > 0:
            out = out + float(l1_aux_weight) * masked_l1(pred, target, mask, eps=eps)
        return out
    if k == "l1":
        return masked_l1(pred, target, mask, eps=eps)
    if k == "huber":
        return masked_smooth_l1(pred, target, mask, beta=huber_beta, eps=eps)
    raise ValueError(f"unknown loss kind: {kind!r}")
