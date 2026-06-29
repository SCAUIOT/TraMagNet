# -*- coding: utf-8 -*-
"""
Gradient + wavelet / morphological hybrid (MATLAB: gradient_wavelet_morphological_filter.m).

v3 (literature-guided refactor):
- TIWT + SpcShrink-style semi-soft wavelet denoise (arXiv:2307.10509, 2023).
- Light gradient-guided morphological edge correction (top-hat residual, no final MA).
- Zero-mean scale alignment for z-score evaluation.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

from ._flat_morphology1d import (
    opening_closing_flat1d,
    preserve_signal_mean,
    preserve_signal_scale,
    smooth_ma,
)
from ._wavelet_denoise1d import tiwt_denoise

ArrayLike1D = Union[Sequence[float], np.ndarray]

_SE_TOPHAT = 5
_MORPH_EDGE_GAIN = 0.06


def _gradient_abs(x: np.ndarray) -> np.ndarray:
    n = x.size
    if n <= 1:
        return np.zeros_like(x)
    g = np.empty(n, dtype=np.float64)
    g[0] = x[1] - x[0]
    g[-1] = x[-1] - x[-2]
    g[1:-1] = (x[2:] - x[:-2]) * 0.5
    return np.abs(g)


def _morph_tophat_residual(x: np.ndarray, se: int) -> np.ndarray:
    smooth = opening_closing_flat1d(x, se)
    resid = x - smooth
    return smooth + 0.35 * resid


def _soft_gradient_weights(g: np.ndarray, *, thr: float) -> np.ndarray:
    t = max(float(thr), 1e-18)
    w = np.clip(g / t, 0.0, 1.0)
    return smooth_ma(w, 9)


def gradient_wavelet_morphological_filter(
    signal: ArrayLike1D,
    time_vector: ArrayLike1D | None = None,
) -> np.ndarray:
    del time_vector
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x.copy()

    wave = tiwt_denoise(x, wavelet="sym4", level=4, max_shift=6, use_spc=True)
    morph = _morph_tophat_residual(x, _SE_TOPHAT)

    g = _gradient_abs(x)
    thr = 0.38 * float(np.std(g)) + 1e-18
    w_edge = _soft_gradient_weights(g, thr=thr)

    den = wave + float(_MORPH_EDGE_GAIN) * w_edge * (morph - wave)
    den = preserve_signal_scale(x, den)
    return preserve_signal_mean(x, den)
