from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from .unet import UNET_INPUT_LENGTH


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        k: int = 4,
        s: int = 2,
        p: int = 1,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            spectral_norm(
                nn.Conv1d(
                    in_ch,
                    out_ch,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Discriminator(nn.Module):
    """
    PatchGAN discriminator for 1D denoising.

    Inputs:
        clean/fake : (B,1,T)
        noisy      : (B,1,T)

    Outputs:
        logits : (B,1,T')
        feats  : feature maps for feature matching
    """

    def __init__(self) -> None:
        super().__init__()

        self.block1 = ConvBlock(2, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 256)

        self.final = spectral_norm(
            nn.Conv1d(
                256,
                1,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

    def forward(
        self,
        x_or_xhat: torch.Tensor,
        noisy: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if (
            x_or_xhat.shape[-1] != UNET_INPUT_LENGTH
            or noisy.shape != x_or_xhat.shape
        ):
            raise ValueError(
                f"input shape mismatch: "
                f"{tuple(x_or_xhat.shape)} / {tuple(noisy.shape)}"
            )

        x = torch.cat([x_or_xhat, noisy], dim=1)

        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)

        logits = self.final(f4)

        return logits, [f1, f2, f3, f4]
