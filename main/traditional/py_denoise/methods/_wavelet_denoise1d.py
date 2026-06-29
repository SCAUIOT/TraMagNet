# -*- coding: utf-8 -*-
"""
1D wavelet denoising helpers (TIWT, semi-soft threshold, SpcShrink-style iteration).

References (ideas only, no external deps beyond pywt/scipy):
- SpcShrink: iterative control-chart wavelet shrinkage (arXiv:2307.10509, 2023)
- TIWT + semi-soft threshold: pulse/ECG denoising (PMC10200194, 2023)
- Seismic optimized DWT: universal + semi-soft hybrid (Chen et al. semi-soft)
"""

from __future__ import annotations

import numpy as np

_DEFAULT_WAVELET = "sym4"
_DEFAULT_LEVEL = 4
_TIWT_MAX_SHIFT = 6
_SPC_ALPHA = 0.015
_SPC_MAX_ITER = 12


def semi_soft_threshold(coeffs: np.ndarray, lam: float) -> np.ndarray:
    """Semi-soft shrinkage (continuous at ±λ, retains large peaks better than soft)."""
    c = np.asarray(coeffs, dtype=np.float64)
    lam = float(max(lam, 0.0))
    if lam <= 0:
        return c.copy()
    mag = np.abs(c)
    out = np.zeros_like(c)
    mask = mag > lam
    if np.any(mask):
        m = mag[mask]
        out[mask] = np.sign(c[mask]) * (m - (lam * lam) / m)
    return out


def _mad_sigma(det: np.ndarray) -> float:
    det = np.asarray(det, dtype=np.float64).ravel()
    if det.size == 0:
        return 1e-12
    return float(np.median(np.abs(det))) / 0.6745 + 1e-12


def _universal_threshold(det: np.ndarray) -> float:
    det = np.asarray(det, dtype=np.float64).ravel()
    n = max(det.size, 2)
    return _mad_sigma(det) * np.sqrt(2.0 * np.log(n))


def spc_shrink_detail(det: np.ndarray, *, alpha: float = _SPC_ALPHA, max_iter: int = _SPC_MAX_ITER) -> np.ndarray:
    """
    SpcShrink-inspired iterative discarding of small wavelet detail coefficients.
    """
    c = np.asarray(det, dtype=np.float64).copy()
    n = c.size
    if n == 0:
        return c
    cap = float(np.max(np.abs(c))) + 1e-18
    ucl0 = max(float(alpha) * cap, _universal_threshold(c) * 0.35)
    for _ in range(int(max_iter)):
        sigma = _mad_sigma(c)
        ucl = max(ucl0, sigma * np.sqrt(2.0 * np.log(max(n, 2))))
        new_c = np.where(np.abs(c) > ucl, c, 0.0)
        if np.allclose(new_c, c, rtol=0, atol=1e-15):
            break
        c = new_c
    lam = max(ucl0, _universal_threshold(c))
    return semi_soft_threshold(c, lam)


def _dwt_denoise_once(
    x: np.ndarray,
    *,
    wavelet: str = _DEFAULT_WAVELET,
    level: int = _DEFAULT_LEVEL,
    use_spc: bool = True,
) -> np.ndarray:
    import pywt

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    w = pywt.Wavelet(wavelet)
    max_lvl = pywt.dwt_max_level(x.size, w.dec_len)
    if max_lvl < 1:
        return x.copy()
    lvl = max(1, min(int(level), max_lvl))
    coeffs = pywt.wavedec(x, w, level=lvl, mode="symmetric")
    new_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        det = np.asarray(c, dtype=np.float64)
        if use_spc:
            new_coeffs.append(spc_shrink_detail(det))
        else:
            lam = _universal_threshold(det)
            new_coeffs.append(semi_soft_threshold(det, lam))
    y = pywt.waverec(new_coeffs, w, mode="symmetric")
    return y[: x.size].astype(np.float64)


def tiwt_denoise(
    x: np.ndarray,
    *,
    wavelet: str = _DEFAULT_WAVELET,
    level: int = _DEFAULT_LEVEL,
    max_shift: int = _TIWT_MAX_SHIFT,
    use_spc: bool = True,
) -> np.ndarray:
    """Translation-invariant DWT via cycle-spinning average."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.size
    if n < 8:
        return _dwt_denoise_once(x, wavelet=wavelet, level=level, use_spc=use_spc)
    n_shift = min(int(max_shift), max(1, n // 32))
    acc = np.zeros(n, dtype=np.float64)
    for s in range(n_shift):
        xs = np.roll(x, s)
        ys = _dwt_denoise_once(xs, wavelet=wavelet, level=level, use_spc=use_spc)
        acc += np.roll(ys, -s)
    return (acc / float(n_shift)).astype(np.float64)


def swt_denoise_detail_band(
    x: np.ndarray,
    *,
    wavelet: str = _DEFAULT_WAVELET,
    level: int = 2,
) -> np.ndarray:
    """SWT on short detail-like component; fallback to TIWT if length insufficient."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    try:
        import pywt
    except ImportError:
        return tiwt_denoise(x, wavelet=wavelet, level=level)

    w = pywt.Wavelet(wavelet)
    max_lvl = pywt.swt_max_level(x.size, w.dec_len)
    if max_lvl < 1:
        return tiwt_denoise(x, wavelet=wavelet, level=level)
    lvl = max(1, min(int(level), max_lvl))
    coeffs = pywt.swt(x, w, level=lvl, start_level=0, trim_approx=True)
    new_coeffs = []
    for (cA, cD) in coeffs:
        det = spc_shrink_detail(np.asarray(cD, dtype=np.float64))
        new_coeffs.append((cA, det))
    y = pywt.iswt(new_coeffs, w)
    return y[: x.size].astype(np.float64)
