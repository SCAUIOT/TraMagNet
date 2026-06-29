"""
unet_single: single-channel noisy-input U-Net ablation (no DnCNN, no latent z).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import UNET_INPUT_LENGTH, UNetBackbone1D


class UNetSingle(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unet = UNetBackbone1D(
            in_ch=1,
            out_ch=1,
            base_ch=64,
            depth=2,
            use_dilation=True,
        )

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        if noisy.dim() != 3 or noisy.size(1) != 1 or noisy.size(2) != UNET_INPUT_LENGTH:
            raise ValueError(
                f"noisy must be (B,1,{UNET_INPUT_LENGTH}), got {tuple(noisy.shape)}"
            )
        return self.unet(noisy)
