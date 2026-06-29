# -*- coding: utf-8 -*-
"""
1D flat structuring-element morphology (MATLAB multi_se / adaptive / gwmf style).

Erosion: min in window; dilation: max in window. Symmetric window of odd length w.
Uses scipy.ndimage for O(n) sliding min/max.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy import ndimage
except ImportError:  # pragma: no cover
    ndimage = None


def _ensure_odd_w(w: int) -> int:
    w = int(w)
    if w < 1:
        raise ValueError("window size must be >= 1")
    if w % 2 == 0:
        w += 1
    return w


def erosion_flat1d(x: np.ndarray, w: int) -> np.ndarray:
    w = _ensure_odd_w(w)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if ndimage is None:
        return _erosion_flat_numpy(x, w)
    return ndimage.minimum_filter1d(x, size=w, mode="nearest")


def dilation_flat1d(x: np.ndarray, w: int) -> np.ndarray:
    w = _ensure_odd_w(w)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if ndimage is None:
        return _dilation_flat_numpy(x, w)
    return ndimage.maximum_filter1d(x, size=w, mode="nearest")


def opening_flat1d(x: np.ndarray, w: int) -> np.ndarray:
    return dilation_flat1d(erosion_flat1d(x, w), w)


def closing_flat1d(x: np.ndarray, w: int) -> np.ndarray:
    return erosion_flat1d(dilation_flat1d(x, w), w)


def opening_closing_flat1d(x: np.ndarray, w: int) -> np.ndarray:
    """closing(opening(x)), same order as MATLAB helpers in repo."""
    return closing_flat1d(opening_flat1d(x, w), w)


def smooth_ma3(x: np.ndarray) -> np.ndarray:
    """MATLAB smooth(..., 3): 3-point moving average, edge padded."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x.copy()
    k = np.ones(3, dtype=np.float64) / 3.0
    xp = np.pad(x, (1, 1), mode="edge")
    return np.convolve(xp, k, mode="valid")


def smooth_ma(x: np.ndarray, k: int) -> np.ndarray:
    """Odd-length moving average with edge padding."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.size
    k = int(k) | 1
    if n == 0 or k < 3:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(k, dtype=np.float64) / float(k)
    return np.convolve(xp, ker, mode="valid")


def preserve_signal_mean(reference: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Align output mean to reference (stabilizes z-score / zero-mean SNR evaluation)."""
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    out = np.asarray(y, dtype=np.float64).reshape(-1)
    return out - float(np.mean(out)) + float(np.mean(ref))


def preserve_signal_scale(
    reference: np.ndarray,
    y: np.ndarray,
    *,
    min_scale: float = 0.55,
    max_scale: float = 1.85,
) -> np.ndarray:
    """Match zero-mean std of y to reference (helps zero-mean SNR after morph smoothing)."""
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    out = np.asarray(y, dtype=np.float64).reshape(-1)
    ref_z = ref - float(np.mean(ref))
    out_z = out - float(np.mean(out))
    sr = float(np.std(ref_z)) + 1e-12
    so = float(np.std(out_z)) + 1e-12
    scale = float(np.clip(sr / so, min_scale, max_scale))
    return out_z * scale + float(np.mean(ref))


def moving_variance(x: np.ndarray, window_size: int) -> np.ndarray:
    """Centered moving variance; edges padded like MATLAB loop."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.size
    w = int(window_size)
    if w < 1 or n == 0:
        return np.zeros_like(x)
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / w
    mean = np.convolve(xp, kernel, mode="valid")
    mean_sq = np.convolve(xp * xp, kernel, mode="valid")
    var = mean_sq - mean * mean
    return np.maximum(var, 1e-18)


def _erosion_flat_numpy(x: np.ndarray, w: int) -> np.ndarray:
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(x.size):
        out[i] = xp[i : i + w].min()
    return out


def _dilation_flat_numpy(x: np.ndarray, w: int) -> np.ndarray:
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(x.size):
        out[i] = xp[i : i + w].max()
    return out
