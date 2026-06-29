"""
DnCNN-only ablation: coarse DnCNN denoising branch from TraMagNet, trained with standalone supervision.

Forward matches the DnCNN submodule in TraMagNet:
``pred = noisy - dncnn_subtract_scale * DnCNN(noisy)``。
"""

from __future__ import annotations

import torch
import torch.nn as nn

DNCNN_INPUT_LENGTH = 1024


def _conv1d(in_ch: int, out_ch: int, *, k: int = 3, dilation: int = 1) -> nn.Conv1d:
    pad = (k // 2) * dilation
    return nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, dilation=dilation)


class DnCNN1D(nn.Module):
    """Same structure as ``DnCNN1D`` in TraMagNet ``models/unet.py`` (depth=17, features=64)."""

    def __init__(self, *, in_ch: int = 1, depth: int = 17, features: int = 64) -> None:
        super().__init__()
        if depth < 3:
            raise ValueError("depth must be >= 3")

        layers: list[nn.Module] = [_conv1d(in_ch, features, k=3), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers.extend(
                [
                    _conv1d(features, features, k=3),
                    nn.BatchNorm1d(features),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(_conv1d(features, in_ch, k=3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DnCNNDenoiser(nn.Module):
    """
    Standalone DnCNN denoiser (no UNet, no latent z, no GAN).

    - Input: noisy (B, 1, T), T = ``DNCNN_INPUT_LENGTH``
    - Output: denoised (B, 1, T)
    """

    def __init__(self) -> None:
        super().__init__()
        self.dncnn = DnCNN1D(in_ch=1, depth=17, features=64)
        self.register_buffer("dncnn_subtract_scale", torch.tensor(0.85))

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        if noisy.dim() != 3 or noisy.size(1) != 1 or noisy.size(2) != DNCNN_INPUT_LENGTH:
            raise ValueError(
                f"noisy must be (B,1,{DNCNN_INPUT_LENGTH}), got {tuple(noisy.shape)}"
            )
        v = self.dncnn(noisy)
        return noisy - self.dncnn_subtract_scale * v
