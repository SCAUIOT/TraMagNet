# -*- coding: utf-8 -*-
"""

Multi-structuring-element morphological filter (MATLAB: multi_se_morphological_filter.m).



v3 (literature-guided refactor):

- Multiscale morphological decomposition (opening pyramid + band-pass details).

- SWT / SpcShrink on high-frequency detail bands (MDPI 2024 hybrid SWT idea).

- Huber-weighted robust fusion across SE scales + morphological median channel.

- MAD peak preservation + scale/mean alignment for z-score evaluation.

"""



from __future__ import annotations



from typing import Sequence, Union



import numpy as np



from ._flat_morphology1d import (

    moving_variance,

    opening_closing_flat1d,

    opening_flat1d,

    preserve_signal_mean,

    preserve_signal_scale,

    smooth_ma3,

)

from ._wavelet_denoise1d import swt_denoise_detail_band, tiwt_denoise



ArrayLike1D = Union[Sequence[float], np.ndarray]



_DEFAULT_WIDTHS = (3, 5, 7, 9, 11, 15)

_DEFAULT_VAR_WIN = 13

_RESIDUAL_BLEND = 0.18

_HUBER_DELTA = 1.5

_ROBUST_MEDIAN_WEIGHT = 0.42

_HIGH_DETAIL_ATTEN = 0.45

_MID_DETAIL_RETAIN = 0.78





def _huber_weights(residuals: np.ndarray, delta: float) -> np.ndarray:

    """Per-scale Huber weights from deviation from cross-scale median."""

    R = np.asarray(residuals, dtype=np.float64)

    med = np.median(R, axis=0, keepdims=True)

    r = R - med

    a = np.abs(r)

    d = float(max(delta, 1e-6))

    w = np.ones_like(a)

    big = a > d

    w[big] = d / (a[big] + 1e-18)

    return w





def _multiscale_morph_decompose(x: np.ndarray, widths: tuple[int, ...]) -> tuple[np.ndarray, list[np.ndarray]]:

    """Opening pyramid: structural = largest opening; details = differences between scales."""

    openings = [opening_flat1d(x, w) for w in widths]

    details: list[np.ndarray] = []

    prev = x

    for o in openings:

        details.append(prev - o)

        prev = o

    structural = openings[-1]

    return structural, details





def _denoise_high_detail(d: np.ndarray) -> np.ndarray:

    if d.size < 16:

        return d * _HIGH_DETAIL_ATTEN

    try:

        y = swt_denoise_detail_band(d, level=2)

    except Exception:

        y = tiwt_denoise(d, level=2, max_shift=4)

    return y * _HIGH_DETAIL_ATTEN





def multi_se_morphological_filter(

    signal: ArrayLike1D,

    time_vector: ArrayLike1D | None = None,

    *,

    window_sizes: tuple[int, ...] = _DEFAULT_WIDTHS,

    variance_window: int = _DEFAULT_VAR_WIN,

    peak_preserve: bool = True,

    peak_threshold_factor: float = 0.10,

    peak_blend: float = 0.90,

    peak_source: str = "original",

    peak_detect: str = "deviation",

    deviation_window: int = 21,

    deviation_factor: float = 2.0,

    baseline_width: int = 0,

    detail_retain: float = 0.0,

    residual_blend: float = _RESIDUAL_BLEND,

) -> np.ndarray:

    """

    Multiscale morphological + wavelet hybrid denoising.



    Unused legacy kwargs (baseline_width, detail_retain) kept for API compatibility.

    """

    del time_vector, baseline_width, detail_retain

    x = np.asarray(signal, dtype=np.float64).reshape(-1)

    if x.size == 0:

        return x.copy()



    widths = tuple(int(w) | 1 for w in window_sizes if int(w) >= 3)

    if len(widths) < 2:

        widths = _DEFAULT_WIDTHS



    structural, details = _multiscale_morph_decompose(x, widths)

    n_det = len(details)

    proc_details: list[np.ndarray] = []

    for i, d in enumerate(details):

        if i < min(2, n_det):

            proc_details.append(_denoise_high_detail(d))

        elif i < n_det - 1:

            proc_details.append(d * _MID_DETAIL_RETAIN)

        else:

            proc_details.append(d * 0.92)



    recon = structural + sum(proc_details)



    stacks = [opening_closing_flat1d(x, w) for w in widths[:5]]

    R = np.stack(stacks, axis=0)

    hw = _huber_weights(R, _HUBER_DELTA)

    med_fused = np.median(R, axis=0)

    mv = np.stack(

        [moving_variance(R[k], variance_window) for k in range(R.shape[0])],

        axis=0,

    )

    inv = 1.0 / mv

    weights = inv / inv.sum(axis=0, keepdims=True)

    var_fused = (weights * R).sum(axis=0)

    huber_fused = (hw * R).sum(axis=0) / (hw.sum(axis=0) + 1e-18)

    morph_fused = 0.5 * var_fused + 0.5 * huber_fused

    fused = (1.0 - _ROBUST_MEDIAN_WEIGHT) * morph_fused + _ROBUST_MEDIAN_WEIGHT * med_fused



    base = 0.55 * recon + 0.45 * fused

    base = smooth_ma3(base)



    if peak_preserve and x.size >= 3:

        if peak_detect == "edge":

            d1 = np.abs(np.diff(x))

            left = np.r_[d1[0], d1]

            right = np.r_[d1, d1[-1]]

            edge = np.maximum(left, right)

            thr = float(peak_threshold_factor) * float(np.std(edge) + 1e-18)

            mask = edge > thr

            mask[1:] |= mask[:-1]

            mask[:-1] |= mask[1:]

        elif peak_detect == "deviation":

            w = int(deviation_window)

            if w < 3:

                w = 3

            if w % 2 == 0:

                w += 1

            try:

                from scipy.ndimage import median_filter



                med = median_filter(x, size=w, mode="nearest")

                mad = median_filter(np.abs(x - med), size=w, mode="nearest")

            except Exception:

                pad = w // 2

                xp = np.pad(x, (pad, pad), mode="edge")

                med = np.empty_like(x)

                mad = np.empty_like(x)

                for i in range(x.size):

                    segment = xp[i : i + w]

                    m = float(np.median(segment))

                    med[i] = m

                    mad[i] = float(np.median(np.abs(segment - m)))

            mask = np.abs(x - med) > (float(deviation_factor) * (mad + 1e-12))

            mask[1:] |= mask[:-1]

            mask[:-1] |= mask[1:]

        else:

            raise ValueError("peak_detect must be 'edge' or 'deviation'")



        a = float(np.clip(peak_blend, 0.0, 1.0))

        if peak_source == "original":

            src = x

        elif peak_source == "smallest":

            src = R[0]

        else:

            raise ValueError("peak_source must be 'original' or 'smallest'")

        base = np.where(mask, a * src + (1.0 - a) * base, base)



    rb = float(np.clip(residual_blend, 0.0, 0.5))

    out = (1.0 - rb) * base + rb * x

    out = preserve_signal_scale(x, out)

    return preserve_signal_mean(x, out)


