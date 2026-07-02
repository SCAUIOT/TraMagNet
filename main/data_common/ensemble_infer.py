"""Multi-fold checkpoint ensemble forward pass: arithmetic mean of denoised outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class UNetMember:
    model: torch.nn.Module


def _load_payload(ckpt_path: Path, device: torch.device) -> tuple[dict[str, Any] | Any, dict[str, Any]]:
    try:
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(ckpt_path, map_location=device)
    extra: dict[str, Any] = {}
    if isinstance(payload, dict):
        extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    return payload, extra


def load_unet_ensemble(
    ckpt_paths: list[Path] | tuple[Path, ...],
    device: torch.device,
    *,
    gan_generator: bool = False,
) -> list[UNetMember]:
    """Load TraMagNet UNet ensemble (``gan_generator=True`` reads ``generator`` key)."""
    from models.unet import UNet, complete_unet_state_dict

    members: list[UNetMember] = []
    for p in ckpt_paths:
        payload, extra = _load_payload(p, device)
        if gan_generator:
            if not isinstance(payload, dict) or "generator" not in payload:
                raise ValueError(f"GAN checkpoint missing generator: {p}")
            sd = payload["generator"]
        elif isinstance(payload, dict) and "model" in payload:
            sd = payload["model"]
        elif isinstance(payload, dict):
            raise ValueError(f"Unsupported checkpoint keys: {p} keys={list(payload.keys())}")
        else:
            sd = payload
        model = UNet().to(device)
        model.load_state_dict(complete_unet_state_dict(model, sd), strict=True)
        model.eval()
        members.append(UNetMember(model=model))
    return members


def unet_make_z(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    z_mode: str,
) -> torch.Tensor:
    from models.unet import UNET_LATENT_CHANNELS, UNET_LATENT_LENGTH, sample_latent

    mode = str(z_mode).strip().lower()
    if mode == "zero":
        return torch.zeros(
            batch_size,
            UNET_LATENT_CHANNELS,
            UNET_LATENT_LENGTH,
            device=device,
            dtype=dtype,
        )
    return sample_latent(batch_size, device=device, dtype=dtype)


@torch.no_grad()
def unet_ensemble_forward(
    members: list[UNetMember],
    noisy: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    acc: torch.Tensor | None = None
    n = max(1, len(members))
    for m in members:
        den = m.model(noisy, z)
        acc = den if acc is None else acc + den
    assert acc is not None
    return acc / float(n)


def load_DnCNN_ensemble(ckpt_paths: list[Path] | tuple[Path, ...], device: torch.device, model_args) -> list[torch.nn.Module]:
    from models.DnCNN_1d import DnCNN1D, DnCNN_config_from_argparse

    models: list[torch.nn.Module] = []
    for p in ckpt_paths:
        payload, _ = _load_payload(p, device)
        if isinstance(payload, dict) and "model" in payload:
            sd = payload["model"]
        else:
            sd = payload
        model = DnCNN1D(DnCNN_config_from_argparse(model_args)).to(device)
        model.load_state_dict(sd, strict=True)
        model.eval()
        models.append(model)
    return models


@torch.no_grad()
def tensor_ensemble_forward(models: list[torch.nn.Module], x: torch.Tensor) -> torch.Tensor:
    acc: torch.Tensor | None = None
    n = max(1, len(models))
    for m in models:
        y = m(x)
        acc = y if acc is None else acc + y
    assert acc is not None
    return acc / float(n)


def load_improved_unet_ensemble(ckpt_paths: list[Path] | tuple[Path, ...], device: torch.device) -> list[torch.nn.Module]:
    from models.improved_unet_1d import ImprovedUNet1D, ImprovedUNet1DConfig

    models: list[torch.nn.Module] = []
    for p in ckpt_paths:
        payload, _ = _load_payload(p, device)
        sd = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        model = ImprovedUNet1D(ImprovedUNet1DConfig()).to(device)
        model.load_state_dict(sd, strict=True)
        model.eval()
        models.append(model)
    return models
