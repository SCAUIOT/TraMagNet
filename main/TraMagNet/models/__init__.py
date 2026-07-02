"""TraMagNet networks: discriminator + UNet generator (formerly embedded TraMagNet backbone)."""

from __future__ import annotations

from .discriminator import Discriminator
from .unet import (
    UNET_INPUT_LENGTH,
    UNET_LATENT_CHANNELS,
    UNET_LATENT_LENGTH,
    UNet,
    complete_unet_state_dict,
    sample_latent,
)
from .unet_loss import (
    MR_STFT_SCALES,
    masked_l1_mean,
    mse_time_frequency_loss,
    multi_scale_stft_mag_l1,
    supervised_unet_loss,
)

__all__ = [
    "Discriminator",
    "MR_STFT_SCALES",
    "UNET_INPUT_LENGTH",
    "UNET_LATENT_CHANNELS",
    "UNET_LATENT_LENGTH",
    "UNet",
    "complete_unet_state_dict",
    "masked_l1_mean",
    "mse_time_frequency_loss",
    "multi_scale_stft_mag_l1",
    "sample_latent",
    "supervised_unet_loss",
]
