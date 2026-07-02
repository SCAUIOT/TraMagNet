"""
TraMagNet main network ``UNet``: DnCNN coarse estimate + conditional latent + U-shaped encoder–decoder backbone (1D).

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Latent z aligned with bottleneck: (B, 512, 4)
UNET_LATENT_CHANNELS = 512
UNET_LATENT_LENGTH = 4
UNET_INPUT_LENGTH = 1024


def sample_latent(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """z ~ N(0,1), shape (B, UNET_LATENT_CHANNELS, UNET_LATENT_LENGTH)."""
    if generator is None:
        return torch.randn(
            batch_size,
            UNET_LATENT_CHANNELS,
            UNET_LATENT_LENGTH,
            device=device,
            dtype=dtype,
        )
    return torch.randn(
        batch_size,
        UNET_LATENT_CHANNELS,
        UNET_LATENT_LENGTH,
        device=device,
        dtype=dtype,
        generator=generator,
    )


def complete_unet_state_dict(model: "UNet", ckpt_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Merge checkpoint with current model defaults (including ``output_residual_scale``)."""
    m = dict(ckpt_state)
    m.pop("res_scale", None)
    cur = model.state_dict()
    return {k: m[k] if k in m else cur[k] for k in cur.keys()}


class UNet(nn.Module):
    """
    1D denoising network: DnCNN coarse features + latent z → U-Net decoder.

    - Inputs: noisy (B,1,T), z (B, UNET_LATENT_CHANNELS, UNET_LATENT_LENGTH), T = ``UNET_INPUT_LENGTH``
    - Concat ``[noisy, x1, z_feat]`` (``x1 = noisy - dncnn_subtract_scale * DnCNN(noisy)``) → backbone
    - Decoder directly outputs denoised signal
    """

    def __init__(self) -> None:
        super().__init__()

        self.dncnn = DnCNN1D(in_ch=1, depth=17, features=64)
        self.z_proj = nn.Conv1d(UNET_LATENT_CHANNELS, 1, kernel_size=1)
        self.unet = UNetBackbone1D(
            in_ch=3,
            out_ch=1,
            base_ch=64,
            depth=2,
            use_dilation=True,
        )
        self.register_buffer("dncnn_subtract_scale", torch.tensor(0.85))
        self.register_buffer("output_residual_scale", torch.tensor(0.9))

    def forward(self, noisy: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if noisy.dim() != 3 or noisy.size(1) != 1 or noisy.size(2) != UNET_INPUT_LENGTH:
            raise ValueError(
                f"noisy must be (B,1,{UNET_INPUT_LENGTH}), got {tuple(noisy.shape)}"
            )
        B = noisy.size(0)
        if z.shape != (B, UNET_LATENT_CHANNELS, UNET_LATENT_LENGTH):
            raise ValueError(
                f"z must be (B,{UNET_LATENT_CHANNELS},{UNET_LATENT_LENGTH}), got {tuple(z.shape)}"
            )

        v = self.dncnn(noisy)
        x1 = noisy - self.dncnn_subtract_scale * v

        z_up = F.interpolate(z, size=UNET_INPUT_LENGTH, mode="linear", align_corners=False)
        z_feat = self.z_proj(z_up)
        u = torch.cat([noisy, x1, z_feat], dim=1)
        return self.unet(u)


def _conv1d(in_ch: int, out_ch: int, *, k: int = 3, dilation: int = 1) -> nn.Conv1d:
    pad = (k // 2) * dilation
    return nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, dilation=dilation)


class DnCNN1D(nn.Module):
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
    """U-shaped encoder–decoder backbone (formerly ``UNet1D`` in the original implementation)."""

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
