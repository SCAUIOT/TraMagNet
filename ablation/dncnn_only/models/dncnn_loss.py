"""Bridge to ``main/TraMagNet/models/unet_loss.py``; same supervised loss as TraMagNet."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_ABLATION = Path(__file__).resolve().parents[2]
_MAIN = _ABLATION.parent / "main"
_LOSS_FILE = _MAIN / "TraMagNet" / "models" / "unet_loss.py"
_spec = importlib.util.spec_from_file_location("tramagnet_unet_loss", _LOSS_FILE)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load TraMagNet unet_loss from {_LOSS_FILE}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

mse_time_frequency_loss = _mod.mse_time_frequency_loss
supervised_unet_loss = _mod.supervised_unet_loss
resolve_mix_weights = _mod.resolve_mix_weights

__all__ = ["mse_time_frequency_loss", "supervised_unet_loss", "resolve_mix_weights"]
