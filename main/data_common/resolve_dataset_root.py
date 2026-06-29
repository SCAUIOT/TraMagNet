"""Shared ``--data-root`` resolution for training and visualization."""

from __future__ import annotations

from pathlib import Path

from .dataset_paths import resolve_dataset_root

__all__ = ["resolve_dataset_root"]
