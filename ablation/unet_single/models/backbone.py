"""1D UNet backbone (from TraMagNet; for unet_single ablation)."""

from __future__ import annotations

import torch
import torch.nn as nn

UNET_INPUT_LENGTH = 1024


def _conv1d(in_ch: int, out_ch: int, *, k: int = 3, dilation: int = 1) -> nn.Conv1d:
    pad = (k // 2) * dilation
    return nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, dilation=dilation)


class DoubleConv1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, mid_ch: int | None = None, dilation: int = 1) -> None:
        super().__init__()
        m = mid_ch if mid_ch is not None else out_ch
        self.block = nn.Sequential(
            _conv1d(in_ch, m, k=3, dilation=dilation),
            nn.BatchNorm1d(m),
            nn.ReLU(inplace=True),
            _conv1d(m, out_ch, k=3, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, dilation: int = 1) -> None:
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv = DoubleConv1D(in_ch, out_ch, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up1D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, *, dilation: int = 1) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.reduce = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.conv = DoubleConv1D(out_ch + skip_ch, out_ch, dilation=dilation)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.reduce(x)
        if skip.size(-1) != x.size(-1):
            diff = skip.size(-1) - x.size(-1)
            if diff > 0:
                skip = skip[..., diff // 2 : diff // 2 + x.size(-1)]
            else:
                x = x[..., (-diff) // 2 : (-diff) // 2 + skip.size(-1)]
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetBackbone1D(nn.Module):
    def __init__(
        self,
        *,
        in_ch: int,
        out_ch: int,
        base_ch: int = 64,
        depth: int = 2,
        use_dilation: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        dil = (1, 2, 4, 8, 16)
        d0 = dil[0] if use_dilation else 1
        self.inc = DoubleConv1D(in_ch, base_ch, dilation=d0)

        downs: list[nn.Module] = []
        skips_ch: list[int] = [base_ch]
        ch = base_ch
        for di in range(depth):
            d = dil[min(di + 1, len(dil) - 1)] if use_dilation else 1
            downs.append(Down1D(ch, ch * 2, dilation=d))
            ch *= 2
            skips_ch.append(ch)
        self.downs = nn.ModuleList(downs)

        d_b = dil[min(depth + 1, len(dil) - 1)] if use_dilation else 1
        self.bottleneck = DoubleConv1D(ch, ch, dilation=d_b)

        ups: list[nn.Module] = []
        for ui in range(depth):
            skip_channels = skips_ch[-(ui + 2)]
            d = dil[min(depth - ui, len(dil) - 1)] if use_dilation else 1
            ups.append(Up1D(ch, skip_channels, ch // 2, dilation=d))
            ch //= 2
        self.ups = nn.ModuleList(ups)
        self.outc = nn.Conv1d(base_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        x = self.inc(x)
        skips.append(x)
        for d in self.downs:
            x = d(x)
            skips.append(x)
        x = self.bottleneck(x)
        for ui, u in enumerate(self.ups):
            skip = skips[-(ui + 2)]
            x = u(x, skip)
        return self.outc(x)
