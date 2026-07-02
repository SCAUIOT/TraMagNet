from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from data_common.eval_split import segment_in_eval_split
from data_common.txt_io import pad_or_resample_to_length, read_amplitude_np, subway_noisy_has_four_value_columns
from data_common.viz_method_splits import list_noisy_files_for_segment_keys


@dataclass(frozen=True)
class Metrics:
    mse: float
    rmse: float
    mae: float
    #: Time-domain SNR (dB) of denoised vs clean reference, i.e. ``_snr_time_db(clean, denoised)`` (each zero-mean); **does not** subtract noisy SNR
    snr_db: float
    pearson_r: float
    #: Noisy vs clean reference SNR (dB), i.e. ``_snr_time_db(clean, noisy)``
    snr_noisy_db: float

    @property
    def delta_snr_db(self) -> float:
        """Legacy compatibility: ΔSNR = denoised SNR − noisy SNR."""
        return float(self.snr_db - self.snr_noisy_db)


def _load_time_series_values_txt(path: Path, *, value_column: int = 2) -> np.ndarray:
    """
    Read a txt signal file and return a 1-D float array (amplitude values).

    Supported formats (data1-style):
    - Single column of floats: entire column is amplitude (e.g. some method result files)
    - 2 columns: use the last column as amplitude
    - ≥3 columns: default to column 3 (index / timestamp / value / ...)
    """
    # Same as ./data*/read_official.py and ./1/data/our_data_folder_dataset.py: use data_common.txt_io
    return read_amplitude_np(path, value_column=int(value_column)).astype(np.float64, copy=False)


def _resample_to_len(y: np.ndarray, target_len: int) -> np.ndarray:
    yy = np.asarray(y, dtype=np.float64).ravel()
    y_out, _mask = pad_or_resample_to_length(list(map(float, yy.tolist())), target_len, mode="resample_linear")
    return np.asarray(y_out, dtype=np.float64)


def _maybe_denormalize_like_training(
    *,
    reference_raw_1024: np.ndarray,
    den_1024: np.ndarray,
    mode: str = "always",
    assume_normalized_if_std_below: float = 20.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    During training (./1/data/our_data_folder_dataset.py) clean is z-scored:
      clean_z = (clean - mu) / sig
    If denoised appears to be in z-score space (small std), denormalize with the same mu/sig back to physical scale.
    """
    c = np.asarray(reference_raw_1024, dtype=np.float64).ravel()
    d = np.asarray(den_1024, dtype=np.float64).ravel()
    sig = float(np.std(c)) + eps
    mu = float(np.mean(c))

    mode = mode.lower().strip()
    if mode not in ("always", "never", "auto"):
        raise ValueError("mode must be one of: always/never/auto")

    if mode == "never":
        return d
    if mode == "always":
        return d * sig + mu

    # auto: denoised scale much smaller than clean -> treat as z-score-space output
    if float(np.std(d)) < float(assume_normalized_if_std_below) and float(np.std(c)) > float(assume_normalized_if_std_below):
        return d * sig + mu
    return d


def _snr_time_db(
    clean: np.ndarray,
    den: np.ndarray,
    *,
    eps: float = 1e-18,
    zero_mean: bool = True,
) -> float:
    """Time-domain SNR (dB): by default clean and den/noisy are **each zero-mean**, then 10·log10(Σclean² / Σ(den−clean)²)."""
    c = np.asarray(clean, dtype=np.float64).ravel()
    d = np.asarray(den, dtype=np.float64).ravel()
    n = min(c.size, d.size)
    c, d = c[:n], d[:n]
    if zero_mean:
        c = c - float(np.mean(c))
        d = d - float(np.mean(d))
    err = d - c
    p_sig = float(np.sum(c * c))
    p_n = float(np.sum(err * err))
    return float(10.0 * math.log10((p_sig + eps) / (p_n + eps)))


def _snr_time_db_noisy(clean: np.ndarray, noisy: np.ndarray, *, eps: float = 1e-18) -> float:
    """Noisy baseline SNR_time (alias for ``_snr_time_db``)."""
    return _snr_time_db(clean, noisy, eps=eps, zero_mean=True)


def _snr_time_db_denoised(clean: np.ndarray, den: np.ndarray, *, eps: float = 1e-18) -> float:
    """Denoised SNR_time (alias for ``_snr_time_db``, also zero-mean each)."""
    return _snr_time_db(clean, den, eps=eps, zero_mean=True)


def _snr_freq_db(
    x: np.ndarray,
    *,
    sample_rate_hz: float,
    f_cut_hz: float,
    eps: float = 1e-18,
) -> float:
    """Same form as ``loss_eval`` frequency-domain SNR: one-sided spectrum S=(0,f_cut] vs N=(f_cut, Nyquist]."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 4:
        return float("nan")
    x0 = x - float(np.mean(x))
    w = np.hanning(n)
    xw = x0 * w
    spec = np.fft.rfft(xw)
    p = (np.abs(spec) ** 2).astype(np.float64)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate_hz))
    lo = (freqs > 1e-6) & (freqs <= float(f_cut_hz))
    hi = freqs > float(f_cut_hz)
    ps = float(np.sum(p[lo]))
    pn = float(np.sum(p[hi]))
    if pn <= eps:
        return float("inf") if ps > eps else float("nan")
    return float(10.0 * math.log10((ps + eps) / (pn + eps)))


def _snr_joint_db(snr_t: float, snr_f: float, *, alpha: float, eps: float = 1e-18) -> float:
    """Same form as ``loss_eval`` joint SNR: linear power weighting then convert to dB."""
    if not (np.isfinite(snr_t) and np.isfinite(snr_f)):
        return float("nan")
    gt = 10.0 ** (float(snr_t) / 10.0)
    gf = 10.0 ** (float(snr_f) / 10.0)
    a = min(1.0, max(0.0, float(alpha)))
    gj = a * gt + (1.0 - a) * gf
    return float(10.0 * math.log10(gj + eps))


def _mean_finite(xs: list[float]) -> float:
    vals = [float(x) for x in xs if np.isfinite(x)]
    return float(np.mean(vals)) if vals else float("nan")


def snr_triplets_for_segment_like_evaluate_methods_on_data1(
    *,
    reference_1024: np.ndarray,
    noisy_1024: np.ndarray,
    den_1024: np.ndarray,
    denorm_mode: str,
    snr_sample_rate_hz: float,
    snr_f_cut_hz: float,
    snr_joint_alpha: float,
) -> tuple[np.ndarray, tuple[float, float, float], tuple[float, float, float]]:
    """
    On a single segment (usually resampled to ``segment_length``), compute six SNR values using the
    **same formulas** as the inner loop of ``evaluate_methods_on_data1``: the first group is time / freq / joint
    for **denoised output**; the second group is **noisy** vs the same ``reference_1024`` (consistent with summary
    table ``snr_noisy_*``).

    **Input conventions (aligned with reading ``reference_signal`` / ``noise_signal`` / ``result`` txt)**

    - ``reference_1024``: clean reference sequence (same scale as after ``_load_time_series_values_txt`` + ``_resample_to_len``).
    - ``noisy_1024``: noisy sequence in **physical/file scale**; **not** passed through ``_maybe_denormalize_like_training``;
      in ``evaluate_methods`` this is the ``noisy_1024`` array itself.
    - ``den_1024``: denoised network output (same semantics as ``d.squeeze()`` written to result column 3 by ``visualize_data``),
      often in z-score space; first denormalized via ``_maybe_denormalize_like_training`` to align with ``reference_1024`` before SNR.

    **Computation steps**

    1. Align lengths ``n = min(len(clean), len(noisy), len(den))`` and truncate to equal length.

    2. **Denormalize denoised** (consistent with training export evaluation)::

         den_raw = _maybe_denormalize_like_training(
             reference_raw_1024=reference_1024, den_1024=den_1024, mode=denorm_mode
         )

       When ``denorm_mode="always"``, let ``μ,σ`` be mean and std of ``reference_1024`` (plus eps),
       ``den_raw = den_1024 * σ + μ``, mapping model output back to amplitude comparable to ``reference_1024``.

    3. **SNR_time (dB, denoised)**: let ``c = reference_1024``, ``d = den_raw``,

       ``SNR_time = 10·log10( Σ c'² / Σ (d'−c')² )`` where ``c'``/``d'`` are each zero-mean (see ``_snr_time_db``).

    4. **SNR_freq (dB, denoised)**: zero-mean ``den_raw``, multiply by Hann, one-sided ``rfft``; signal band is
       power sum ``P_S`` for ``0 < f ≤ f_cut``, noise band is power sum ``P_N`` for ``f > f_cut`` (sample rate
       ``snr_sample_rate_hz``), ``SNR_freq = 10·log10(P_S / P_N)`` (see ``_snr_freq_db``).

    5. **SNR_joint (dB, denoised)**: ``γ_t = 10^(SNR_time/10)``, ``γ_f = 10^(SNR_freq/10)``,
       ``γ = α·γ_t + (1−α)·γ_f``, ``SNR_joint = 10·log10(γ)``, ``α = snr_joint_alpha``.

    6. **Noisy triplet**: replace ``den_raw`` with ``noisy_1024`` in steps 3–5 (**no denormalization** for noisy),
       time domain uses ``_snr_time_db`` (reference/noisy each zero-mean, same definition as denoised).

    Returns:
        ``(den_raw, (snr_time, snr_freq, snr_joint)_den, (snr_noisy_time, snr_noisy_freq, snr_noisy_joint))``
        where ``den_raw`` matches what ``compute_five_metrics(reference_1024, noisy_1024, den_raw)`` uses.
    """
    reference_1024 = np.asarray(reference_1024, dtype=np.float64).ravel()
    noisy_1024 = np.asarray(noisy_1024, dtype=np.float64).ravel()
    den_1024 = np.asarray(den_1024, dtype=np.float64).ravel()
    n = int(min(reference_1024.size, noisy_1024.size, den_1024.size))
    if n <= 0:
        nan3 = (float("nan"), float("nan"), float("nan"))
        return np.asarray([], dtype=np.float64), nan3, nan3
    reference_1024 = reference_1024[:n]
    noisy_1024 = noisy_1024[:n]
    den_1024 = den_1024[:n]

    den_raw = _maybe_denormalize_like_training(
        reference_raw_1024=reference_1024,
        den_1024=den_1024,
        mode=str(denorm_mode),
    )
    c_arr = np.asarray(reference_1024, dtype=np.float64).ravel()
    n_arr = np.asarray(noisy_1024, dtype=np.float64).ravel()
    d_arr = np.asarray(den_raw, dtype=np.float64).ravel()

    snr_time_db = _snr_time_db(c_arr, d_arr)
    snr_freq_db = _snr_freq_db(d_arr, sample_rate_hz=float(snr_sample_rate_hz), f_cut_hz=float(snr_f_cut_hz))
    snr_joint_db = _snr_joint_db(snr_time_db, snr_freq_db, alpha=float(snr_joint_alpha))

    snr_noisy_time_db, snr_noisy_freq_db, snr_noisy_joint_db = snr_noisy_triplet_for_segment_like_evaluate_methods_on_data1(
        reference_1024=c_arr,
        noisy_1024=n_arr,
        snr_sample_rate_hz=float(snr_sample_rate_hz),
        snr_f_cut_hz=float(snr_f_cut_hz),
        snr_joint_alpha=float(snr_joint_alpha),
    )

    return den_raw, (snr_time_db, snr_freq_db, snr_joint_db), (
        snr_noisy_time_db,
        snr_noisy_freq_db,
        snr_noisy_joint_db,
    )


def snr_noisy_triplet_for_segment_like_evaluate_methods_on_data1(
    *,
    reference_1024: np.ndarray,
    noisy_1024: np.ndarray,
    snr_sample_rate_hz: float,
    snr_f_cut_hz: float,
    snr_joint_alpha: float,
) -> tuple[float, float, float]:
    """
    **Noisy vs clean** triplet only; identical to ``snr_noisy_time_db`` / ``snr_noisy_freq_db`` /
    ``snr_noisy_joint_db`` in ``evaluate_methods_on_data1``: **no** ``_maybe_denormalize_like_training`` on noisy;
    ``reference_1024`` / ``noisy_1024`` must match arrays after reading txt and ``_resample_to_len``.
    Time-domain SNR uses ``_snr_time_db`` (each zero-mean).
    """
    c_arr = np.asarray(reference_1024, dtype=np.float64).ravel()
    n_arr = np.asarray(noisy_1024, dtype=np.float64).ravel()
    n = int(min(c_arr.size, n_arr.size))
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    c_arr = c_arr[:n]
    n_arr = n_arr[:n]
    snr_noisy_time_db = _snr_time_db(c_arr, n_arr)
    snr_noisy_freq_db = _snr_freq_db(
        n_arr, sample_rate_hz=float(snr_sample_rate_hz), f_cut_hz=float(snr_f_cut_hz)
    )
    snr_noisy_joint_db = _snr_joint_db(
        snr_noisy_time_db, snr_noisy_freq_db, alpha=float(snr_joint_alpha)
    )
    return snr_noisy_time_db, snr_noisy_freq_db, snr_noisy_joint_db


def compute_five_metrics(
    clean_t: np.ndarray,
    noisy_t: np.ndarray,
    denoised_t: np.ndarray,
    *,
    eps: float = 1e-12,
) -> Metrics:
    """
    Inputs: time-domain clean, noisy, and denoised signals (lengths must match).
    Outputs: MSE, RMSE, MAE, ``snr_db`` (dB), Pearson r, and ``snr_noisy_db`` (noisy SNR).

    ``snr_db`` definition (time-domain SNR, clean and denoised **each zero-mean**):
    - ``snr_db`` = ``_snr_time_db(clean, denoised)``
    - ``snr_noisy_db`` = ``_snr_time_db(clean, noisy)``
    - Summary table ``snr_db_mean`` is the mean of denoised SNR above (**not** minus noisy SNR)
    """
    clean = np.asarray(clean_t, dtype=np.float64).ravel()
    noisy = np.asarray(noisy_t, dtype=np.float64).ravel()
    den = np.asarray(denoised_t, dtype=np.float64).ravel()

    n = min(clean.size, noisy.size, den.size)
    if n == 0:
        raise ValueError("Input signal length is 0")
    if clean.size != n or noisy.size != n or den.size != n:
        clean = clean[:n]
        noisy = noisy[:n]
        den = den[:n]

    err = den - clean
    mse = float(np.mean(err * err))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    snr_noisy_db = _snr_time_db(clean, noisy, eps=eps)
    snr_db = _snr_time_db(clean, den, eps=eps)

    # Pearson r (hand-written to avoid extra dependencies)
    c0 = clean - float(np.mean(clean))
    d0 = den - float(np.mean(den))
    denom = float(np.sqrt(np.sum(c0 * c0) * np.sum(d0 * d0)))
    pearson_r = float(np.sum(c0 * d0) / (denom + eps))

    return Metrics(
        mse=mse,
        rmse=rmse,
        mae=mae,
        snr_db=float(snr_db),
        pearson_r=pearson_r,
        snr_noisy_db=float(snr_noisy_db),
    )


DNCNN_METHOD = "dncnn"
GAN_METHOD = "TraMagNet"
UNET_SINGLE_METHOD = "unet_single"


def _reference_filename_for_dataset(noisy_name: str, data_root: Path) -> str:
    from data_common.flat_pairing import reference_filename_for_noisy

    name = reference_filename_for_noisy(noisy_name, data_root=data_root)
    return name if name else noisy_name


def _iter_txt_files(dir_path: Path) -> Iterable[Path]:
    return (p for p in dir_path.glob("*.txt") if p.is_file())


def add_pt_inference_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_device_ckpt: bool = True,
) -> None:
    """dncnn / TraMagNet / UNet-only ablation K-fold .pt on-the-fly inference args."""
    from data_common.cv_ensemble import add_cv_ensemble_arguments

    g5 = parser.add_argument_group("TraMagNet K-fold inference (MagGAN)")
    g5.add_argument(
        "--tramagnet-runs-dir",
        type=str,
        default=None,
        dest="gan_runs_dir",
        metavar="DIR",
        help="Runs directory with fold_0..fold_{K-1}; auto-search if omitted",
    )
    g5.add_argument("--gan-z-mode", type=str, default="zero", choices=("zero", "random"))
    gc = parser.add_argument_group("DnCNN K-fold inference")
    gc.add_argument("--dncnn-runs-dir", type=str, default=None, dest="dncnn_runs_dir", metavar="DIR")
    gc.add_argument("--features", type=int, default=64)
    gc.add_argument("--middle-depth", type=int, default=10, dest="middle_depth")
    gc.add_argument("--num-residual", type=int, default=5, dest="num_residual")
    gc.add_argument("--use-attention", action="store_true", dest="use_attention")
    gc.add_argument("--no-attention", action="store_true", dest="no_attention")
    gc.add_argument("--attention-reduction", type=int, default=8, dest="attention_reduction")
    g9 = parser.add_argument_group("UNet-only ablation K-fold inference")
    g9.add_argument(
        "--unet-single-runs-dir",
        type=str,
        default=None,
        dest="unet_single_runs_dir",
        metavar="DIR",
        help="UNet-only ablation runs dir with fold_0..fold_{K-1}; auto-search under ablation/unet_single/output if omitted",
    )
    if include_device_ckpt:
        g5.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
        g5.add_argument("--ckpt", type=str, default="last", help="Prefer last or best checkpoint per fold")
    add_cv_ensemble_arguments(parser)


def init_pt_inferers(
    *,
    args: argparse.Namespace,
    methods: list[str],
    data_root: Path,
    dataset_tag: str,
    denorm_mode: str,
) -> tuple[Any | None, Any | None, Any | None]:
    """Initialize cnn / TraMagNet / UNet-only ablation inferers per ``methods``; returns ``None`` for methods not listed."""
    from data_common.infer_cv import DnCNNCvInferer, TraMagNetCvInferer, TRAMAGNET_METHOD, UnetSingleCvInferer

    dncnn_inferer = None
    gan_inferer = None
    unet_single_inferer = None
    if DNCNN_METHOD in methods:
        dncnn_inferer = DnCNNCvInferer(
            args=args,
            data_root=data_root,
            dataset_tag=dataset_tag,
            denorm=str(denorm_mode),
            value_scale=1.0,
        )
    if GAN_METHOD in methods or TRAMAGNET_METHOD in methods:
        gan_inferer = TraMagNetCvInferer(
            args=args,
            data_root=data_root,
            dataset_tag=dataset_tag,
            denorm=str(denorm_mode),
            value_scale=1.0,
        )
    if UNET_SINGLE_METHOD in methods:
        unet_single_inferer = UnetSingleCvInferer(
            args=args,
            data_root=data_root,
            dataset_tag=dataset_tag,
            denorm=str(denorm_mode),
            value_scale=1.0,
        )
    return dncnn_inferer, gan_inferer, unet_single_inferer


def _build_eval_row(
    *,
    method: str,
    noisy_name: str,
    reference_name: str,
    ch_tag: str,
    reference_1024: np.ndarray,
    noisy_1024: np.ndarray,
    den_1024: np.ndarray,
    include_snr_triplets: bool,
    denorm_mode: str,
    snr_sample_rate_hz: float,
    snr_f_cut_hz: float,
    snr_joint_alpha: float,
    pt_inferred: bool,
) -> dict:
    row_extra: dict = {}
    if include_snr_triplets:
        trip_denorm = "never" if pt_inferred else str(denorm_mode)
        _, (snr_time_db, snr_freq_db, snr_joint_db), (
            snr_noisy_time_db,
            snr_noisy_freq_db,
            snr_noisy_joint_db,
        ) = snr_triplets_for_segment_like_evaluate_methods_on_data1(
            reference_1024=reference_1024,
            noisy_1024=noisy_1024,
            den_1024=den_1024,
            denorm_mode=trip_denorm,
            snr_sample_rate_hz=float(snr_sample_rate_hz),
            snr_f_cut_hz=float(snr_f_cut_hz),
            snr_joint_alpha=float(snr_joint_alpha),
        )
        row_extra = {
            "snr_time_db": snr_time_db,
            "snr_freq_db": snr_freq_db,
            "snr_joint_db": snr_joint_db,
            "snr_noisy_time_db": snr_noisy_time_db,
            "snr_noisy_freq_db": snr_noisy_freq_db,
            "snr_noisy_joint_db": snr_noisy_joint_db,
        }
        den_raw = den_1024 if pt_inferred else _maybe_denormalize_like_training(
            reference_raw_1024=reference_1024,
            den_1024=den_1024,
            mode=denorm_mode,
        )
    else:
        den_raw = (
            den_1024
            if pt_inferred
            else _maybe_denormalize_like_training(
                reference_raw_1024=reference_1024,
                den_1024=den_1024,
                mode=denorm_mode,
            )
        )

    m = compute_five_metrics(reference_1024, noisy_1024, den_raw)
    m_noisy = compute_five_metrics(reference_1024, noisy_1024, noisy_1024)
    return {
        "method": method,
        "file": noisy_name,
        "reference_file": reference_name,
        "channel": ch_tag,
        "mse": m.mse,
        "rmse": m.rmse,
        "mae": m.mae,
        "snr_db": m.snr_db,
        "pearson_r": m.pearson_r,
        "snr_noisy_db": m.snr_noisy_db,
        "mse_noisy_baseline": m_noisy.mse,
        "rmse_noisy_baseline": m_noisy.rmse,
        "mae_noisy_baseline": m_noisy.mae,
        "pearson_r_noisy_baseline": m_noisy.pearson_r,
        **row_extra,
    }


def _evaluate_pt_method_on_test_set(
    *,
    method: str,
    inferer: Any,
    rows: list[dict],
    reference_dir: Path,
    noise_dir: Path,
    dataset_tag: str,
    segment_length: int,
    denorm_mode: str,
    method_keys: set[tuple[str, str]] | None,
    include_snr_triplets: bool,
    snr_sample_rate_hz: float,
    snr_f_cut_hz: float,
    snr_joint_alpha: float,
) -> None:
    """cnn / TraMagNet: K-fold .pt on-the-fly inference on respective test sets (does not read output/result)."""
    sample_names = list_noisy_files_for_segment_keys(method_keys, noise_dir)
    if not sample_names:
        print(f"[WARN] Skipping {method}: no usable noisy files in test set", flush=True)
        return

    seg = int(getattr(getattr(inferer, "pre_cfg", None), "segment_length", 0) or 0) or int(segment_length)

    for noisy_name in sample_names:
        reference_name = _reference_filename_for_dataset(noisy_name, reference_dir.parent)
        reference_path = reference_dir / reference_name
        noisy_path = noise_dir / noisy_name
        if not reference_path.is_file() or not noisy_path.is_file():
            print(f"[SKIP] {method} missing reference/noisy: {reference_name} / {noisy_name}", flush=True)
            continue

        noisy_has_dual = bool(subway_noisy_has_four_value_columns(noisy_path))

        try:
            if noisy_has_dual:
                reference0 = _resample_to_len(_load_time_series_values_txt(reference_path, value_column=2), seg)
                reference1 = _resample_to_len(_load_time_series_values_txt(reference_path, value_column=3), seg)
                noisy0 = _resample_to_len(_load_time_series_values_txt(noisy_path, value_column=2), seg)
                noisy1 = _resample_to_len(_load_time_series_values_txt(noisy_path, value_column=3), seg)
                den0, den1 = inferer.infer_dual(
                    reference_path=reference_path,
                    noisy_path=noisy_path,
                    noisy0=noisy0,
                    noisy1=noisy1,
                )
                if segment_in_eval_split(noisy_name, method_keys, channel="ch0"):
                    rows.append(
                        _build_eval_row(
                            method=method,
                            noisy_name=noisy_name,
                            reference_name=reference_name,
                            ch_tag="ch0",
                            reference_1024=reference0,
                            noisy_1024=noisy0,
                            den_1024=den0,
                            include_snr_triplets=include_snr_triplets,
                            denorm_mode=denorm_mode,
                            snr_sample_rate_hz=snr_sample_rate_hz,
                            snr_f_cut_hz=snr_f_cut_hz,
                            snr_joint_alpha=snr_joint_alpha,
                            pt_inferred=True,
                        )
                    )
                if segment_in_eval_split(noisy_name, method_keys, channel="ch1"):
                    rows.append(
                        _build_eval_row(
                            method=method,
                            noisy_name=noisy_name,
                            reference_name=reference_name,
                            ch_tag="ch1",
                            reference_1024=reference1,
                            noisy_1024=noisy1,
                            den_1024=den1,
                            include_snr_triplets=include_snr_triplets,
                            denorm_mode=denorm_mode,
                            snr_sample_rate_hz=snr_sample_rate_hz,
                            snr_f_cut_hz=snr_f_cut_hz,
                            snr_joint_alpha=snr_joint_alpha,
                            pt_inferred=True,
                        )
                    )
                continue

            if not segment_in_eval_split(noisy_name, method_keys, channel="ch0"):
                continue
            clean = _resample_to_len(_load_time_series_values_txt(reference_path, value_column=2), seg)
            noisy = _resample_to_len(_load_time_series_values_txt(noisy_path, value_column=2), seg)
            den = inferer.infer_single(
                reference_path=reference_path,
                noisy_path=noisy_path,
                fallback=noisy,
            )
            rows.append(
                _build_eval_row(
                    method=method,
                    noisy_name=noisy_name,
                    reference_name=reference_name,
                    ch_tag="ch0",
                    reference_1024=clean,
                    noisy_1024=noisy,
                    den_1024=den,
                    include_snr_triplets=include_snr_triplets,
                    denorm_mode=denorm_mode,
                    snr_sample_rate_hz=snr_sample_rate_hz,
                    snr_f_cut_hz=snr_f_cut_hz,
                    snr_joint_alpha=snr_joint_alpha,
                    pt_inferred=True,
                )
            )
        except Exception as e:
            print(f"[WARN] [{method}] Skipping {noisy_name}: {e}", flush=True)
            continue


def evaluate_methods_on_data1(
    *,
    reference_dir: Path,
    noise_dir: Path,
    output_root: Path,
    methods: list[str],
    skip_missing_method_dir: bool = True,
    denorm_mode: str = "always",
    dataset_tag: str = "data1",
    snr_sample_rate_hz: float = 360.0,
    snr_f_cut_hz: float = 20.0,
    snr_joint_alpha: float = 0.5,
    include_snr_triplets: bool = False,
    allowed_segment_keys: set[tuple[str, str]] | None = None,
    allowed_segment_keys_by_method: dict[str, set[tuple[str, str]] | None] | None = None,
    baseline_segment_keys: set[tuple[str, str]] | None = None,
    segment_length: int = 1024,
    dncnn_inferer: Any | None = None,
    gan_inferer: Any | None = None,
    unet_single_inferer: Any | None = None,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []

    for method in methods:
        method_keys = allowed_segment_keys
        if allowed_segment_keys_by_method is not None:
            method_keys = allowed_segment_keys_by_method.get(method, allowed_segment_keys)

        if method == DNCNN_METHOD:
            if dncnn_inferer is None:
                print("[WARN] Skipping cnn: K-fold inferer not initialized", flush=True)
                continue
            _evaluate_pt_method_on_test_set(
                method=DNCNN_METHOD,
                inferer=dncnn_inferer,
                rows=rows,
                reference_dir=reference_dir,
                noise_dir=noise_dir,
                dataset_tag=dataset_tag,
                segment_length=int(segment_length),
                denorm_mode=denorm_mode,
                method_keys=method_keys,
                include_snr_triplets=include_snr_triplets,
                snr_sample_rate_hz=snr_sample_rate_hz,
                snr_f_cut_hz=snr_f_cut_hz,
                snr_joint_alpha=snr_joint_alpha,
            )
            continue

        if method == GAN_METHOD:
            if gan_inferer is None:
                print("[WARN] Skipping TraMagNet: K-fold inferer not initialized", flush=True)
                continue
            _evaluate_pt_method_on_test_set(
                method=GAN_METHOD,
                inferer=gan_inferer,
                rows=rows,
                reference_dir=reference_dir,
                noise_dir=noise_dir,
                dataset_tag=dataset_tag,
                segment_length=int(segment_length),
                denorm_mode=denorm_mode,
                method_keys=method_keys,
                include_snr_triplets=include_snr_triplets,
                snr_sample_rate_hz=snr_sample_rate_hz,
                snr_f_cut_hz=snr_f_cut_hz,
                snr_joint_alpha=snr_joint_alpha,
            )
            continue

        if method == UNET_SINGLE_METHOD:
            if unet_single_inferer is None:
                print("[WARN] Skipping UNet-only ablation: K-fold inferer not initialized", flush=True)
                continue
            _evaluate_pt_method_on_test_set(
                method=UNET_SINGLE_METHOD,
                inferer=unet_single_inferer,
                rows=rows,
                reference_dir=reference_dir,
                noise_dir=noise_dir,
                dataset_tag=dataset_tag,
                segment_length=int(segment_length),
                denorm_mode=denorm_mode,
                method_keys=method_keys,
                include_snr_triplets=include_snr_triplets,
                snr_sample_rate_hz=snr_sample_rate_hz,
                snr_f_cut_hz=snr_f_cut_hz,
                snr_joint_alpha=snr_joint_alpha,
            )
            continue

        result_dir = output_root / method / dataset_tag / "result"
        if not result_dir.exists():
            if skip_missing_method_dir:
                print(f"[WARN] Skipping method {method}: result directory not found {result_dir}")
                continue
            raise FileNotFoundError(f"Result directory not found: {result_dir}")

        for den_path in _iter_txt_files(result_dir):
            noisy_name = den_path.name
            reference_name = _reference_filename_for_dataset(noisy_name, reference_dir.parent)

            reference_path = reference_dir / reference_name
            noisy_path = noise_dir / noisy_name

            if not reference_path.exists():
                raise FileNotFoundError(
                    f"[{method}] Missing clean file: {reference_path} (expected same name as {noisy_name})"
                )
            if not noisy_path.exists():
                raise FileNotFoundError(f"[{method}] Missing noisy file: {noisy_path} (must match result filename)")

            noisy_has_dual = bool(subway_noisy_has_four_value_columns(noisy_path))
            den_has_dual = bool(subway_noisy_has_four_value_columns(den_path))

            # data3: if noisy is dual-channel (columns 3/4), split into ch0/ch1 samples;
            # clean reference must align with the same amplitude column (same as training/our_data_dataset).
            channel_value_columns = [2, 3] if noisy_has_dual else [2]

            for ch_i, vcol in enumerate(channel_value_columns):
                ch_tag = f"ch{ch_i}" if noisy_has_dual else "ch0"
                if method_keys is not None and (noisy_name, ch_tag) not in method_keys:
                    continue
                try:
                    clean = _load_time_series_values_txt(reference_path, value_column=vcol)
                    noisy = _load_time_series_values_txt(noisy_path, value_column=vcol)
                    if den_has_dual:
                        den = _load_time_series_values_txt(den_path, value_column=vcol)
                    else:
                        den = _load_time_series_values_txt(den_path, value_column=2)
                        if noisy_has_dual and ch_i == 1:
                            # noisy has a second channel but den lacks it: skip to avoid counting the same den twice
                            print(
                                f"[WARN] Skipping ch1 for {method} / {dataset_tag} / {noisy_name}: "
                                f"noisy has four columns (dual-channel) but denoised result does not "
                                f"(confirm output exports dual-channel)"
                            )
                            continue

                    reference_1024 = _resample_to_len(clean, segment_length)
                    noisy_1024 = _resample_to_len(noisy, segment_length)
                    den_1024 = _resample_to_len(den, segment_length)
                    row = _build_eval_row(
                        method=method,
                        noisy_name=noisy_name,
                        reference_name=reference_name,
                        ch_tag=ch_tag,
                        reference_1024=reference_1024,
                        noisy_1024=noisy_1024,
                        den_1024=den_1024,
                        include_snr_triplets=include_snr_triplets,
                        denorm_mode=denorm_mode,
                        snr_sample_rate_hz=snr_sample_rate_hz,
                        snr_f_cut_hz=snr_f_cut_hz,
                        snr_joint_alpha=snr_joint_alpha,
                        pt_inferred=False,
                    )
                except Exception as e:
                    print(
                        f"[WARN] [{method}] Skipping {noisy_name} (ch{ch_i}): {e}",
                        flush=True,
                    )
                    break
                rows.append(row)

    summary_rows = summarize_eval_detail_rows(
        rows,
        baseline_segment_keys=baseline_segment_keys
        if baseline_segment_keys is not None
        else allowed_segment_keys,
        include_snr_triplets=include_snr_triplets,
    )
    return rows, summary_rows


def summarize_eval_detail_rows(
    rows: list[dict],
    *,
    baseline_segment_keys: set[tuple[str, str]] | None = None,
    include_snr_triplets: bool = False,
) -> list[dict]:
    """Build summary table from detail rows produced by ``evaluate_methods_on_data1`` / on-the-fly inference (includes Noisy baseline row)."""
    by_method: dict[str, list[dict]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    summary_rows: list[dict] = []
    baseline_by_key: dict[tuple[str, str], dict] = {}
    for r in rows:
        fk = (str(r["file"]), str(r["channel"]))
        if baseline_segment_keys is not None and fk not in baseline_segment_keys:
            continue
        if fk not in baseline_by_key:
            baseline_by_key[fk] = r

    if baseline_by_key:
        br = list(baseline_by_key.values())

        def _mean_key(rows_in: list[dict], key: str) -> float:
            return float(np.mean([float(x[key]) for x in rows_in]))

        noisy_summary: dict = {
            "method": "Noisy",
            "count": len(br),
            "mse_mean": _mean_key(br, "mse_noisy_baseline"),
            "rmse_mean": _mean_key(br, "rmse_noisy_baseline"),
            "mae_mean": _mean_key(br, "mae_noisy_baseline"),
            "snr_db_mean": _mean_key(br, "snr_noisy_db"),
            "pearson_r_mean": _mean_key(br, "pearson_r_noisy_baseline"),
        }
        if include_snr_triplets:
            noisy_summary.update(
                {
                    "snr_time_db_mean": _mean_finite([float(x["snr_noisy_time_db"]) for x in br]),
                    "snr_freq_db_mean": _mean_finite([float(x["snr_noisy_freq_db"]) for x in br]),
                    "snr_joint_db_mean": _mean_finite([float(x["snr_noisy_joint_db"]) for x in br]),
                }
            )
        summary_rows.append(noisy_summary)

    for method, rs in sorted(by_method.items(), key=lambda x: x[0]):
        def mean(key: str) -> float:
            return float(np.mean([x[key] for x in rs]))

        row_d: dict = {
            "method": method,
            "count": len(rs),
            "mse_mean": mean("mse"),
            "rmse_mean": mean("rmse"),
            "mae_mean": mean("mae"),
            "snr_db_mean": mean("snr_db"),
            "pearson_r_mean": mean("pearson_r"),
        }
        if include_snr_triplets:
            row_d.update(
                {
                    "snr_time_db_mean": _mean_finite([float(x["snr_time_db"]) for x in rs]),
                    "snr_freq_db_mean": _mean_finite([float(x["snr_freq_db"]) for x in rs]),
                    "snr_joint_db_mean": _mean_finite([float(x["snr_joint_db"]) for x in rs]),
                }
            )
        summary_rows.append(row_d)

    return summary_rows


def _format_float(x: float) -> str:
    if not np.isfinite(x):
        return str(x)
    ax = abs(x)
    if (ax != 0.0 and ax < 1e-3) or ax >= 1e6:
        return f"{x:.6e}"
    return f"{x:.6f}"


def print_summary_table(summary_rows: list[dict]) -> None:
    if not summary_rows:
        print("[INFO] No data to summarize (output-root/method/result path may be wrong, or result directory is empty)")
        return

    has_snr = any("snr_time_db_mean" in r for r in summary_rows)
    headers = [
        "method",
        "count",
        "mse_mean",
        "rmse_mean",
        "mae_mean",
        "snr_db_mean",
        "pearson_r_mean",
    ]
    if has_snr:
        headers.extend(["snr_time_db_mean", "snr_freq_db_mean", "snr_joint_db_mean"])
    table = []
    for r in summary_rows:
        row_cells = [
            str(r["method"]),
            str(r["count"]),
            _format_float(float(r["mse_mean"])),
            _format_float(float(r["rmse_mean"])),
            _format_float(float(r["mae_mean"])),
            _format_float(float(r["snr_db_mean"])),
            _format_float(float(r["pearson_r_mean"])),
        ]
        if has_snr:
            row_cells.extend(
                [
                    _format_float(float(r["snr_time_db_mean"])),
                    _format_float(float(r["snr_freq_db_mean"])),
                    _format_float(float(r["snr_joint_db_mean"])),
                ]
            )
        table.append(row_cells)

    widths = [len(h) for h in headers]
    for row in table:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(items: list[str]) -> str:
        return " | ".join(items[i].ljust(widths[i]) for i in range(len(items)))

    if has_snr:
        print(
            "[legend] First row Noisy: snr_db_mean = noisy SNR (dB, reference/noisy each zero-mean); "
            "snr_time/freq/joint_db_mean = mean of noisy triplet SNR (same definition as loss_eval). "
            "Other rows: snr_db_mean = denoised SNR (dB, clean/denoised each zero-mean, not minus noisy); "
            "snr_*_db_mean = mean of denoised triplet SNR.",
            flush=True,
        )
    else:
        print(
            "[legend] First row Noisy: snr_db_mean = noisy vs reference reference SNR (dB); "
            "mse/rmse/mae/pearson_r are noisy baseline. "
            "Other rows: snr_db_mean = denoised SNR (dB, not minus noisy); MSE, RMSE, MAE, Pearson r are per-method means.",
            flush=True,
        )
    print(line(headers))
    print("-+-".join("-" * w for w in widths))
    for row in table:
        print(line(row))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute MSE, RMSE, MAE, denoised SNR (dB), Pearson r aligned with reference/noisy and print summary table. "
        "Traditional methods read output/result; dncnn/TraMagNet/UNet-only ablation use K-fold .pt on-the-fly inference (same as viz). "
        "Default --split test: DnCNN baseline single-dataset hold-out; other methods (incl. TraMagNet) use MagGAN data134 hold-out.",
    )
    parser.add_argument("--data-root", type=str, default="./data1", help="Directory containing reference_signal/noise_signal")
    parser.add_argument("--output-root", type=str, default="./output", help="Output root containing per-method directories")
    parser.add_argument(
        "--band",
        type=str,
        default="all",
        choices=("low", "middle", "high", "all"),
        help="Frequency bands to enumerate when building split keys (same as training --band, default all).",
    )
    parser.add_argument(
        "--subway-dual-channels",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="subway_dual_channels",
        help="Whether to split ch0/ch1 when data3 subway has four columns (same as training).",
    )
    parser.add_argument(
        "--dataset-tag",
        type=str,
        default=None,
        help="Dataset subdirectory name under output (e.g. data1/data2/data3). Default: last segment of --data-root path.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="TraMagNet,dncnn,gradient_wavelet_morphological_filter,multi_se_morphological_filter",
        help="Comma-separated method names; dncnn/TraMagNet/UNet-only ablation use .pt inference, not output/result",
    )
    add_pt_inference_arguments(parser)
    parser.add_argument("--segment-length", type=int, default=1024, help="Resample length (traditional methods; pt uses runs config)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: raise if any method directory is missing (default: skip missing and print WARN)",
    )
    parser.add_argument(
        "--denorm",
        type=str,
        default="always",
        choices=["always", "never", "auto"],
        help="Whether to denormalize denoised per training z-score: always/never/auto",
    )
    from data_common.viz_method_splits import (
        build_dncnn_test_segment_keys,
        build_gan_test_segment_keys,
        print_method_test_banners,
    )
    from data_common.viz_split import add_viz_split_arguments

    add_viz_split_arguments(parser, default_split="test")
    args = parser.parse_args()

    _repo = Path(__file__).resolve().parent
    from data_common.dataset_paths import dataset_tag_for_path
    from data_common.resolve_dataset_root import resolve_dataset_root

    data_root = Path(resolve_dataset_root(args.data_root, repo=_repo))
    reference_dir = data_root / "reference_signal"
    noise_dir = data_root / "noise_signal"
    output_root = Path(args.output_root)
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    dataset_tag = str(args.dataset_tag).strip() if args.dataset_tag else dataset_tag_for_path(data_root)

    split_s = str(args.split).lower().strip()
    dncnn_keys = build_dncnn_test_segment_keys(
        data_root,
        split=split_s,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
        shuffle_split=bool(args.shuffle_split),
        band=str(args.band),
        subway_dual_channels=bool(args.subway_dual_channels),
    )
    gan_keys = build_gan_test_segment_keys(
        data_root,
        _repo,
        split=split_s,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
        shuffle_split=bool(args.shuffle_split),
        band=str(args.band),
        subway_dual_channels=bool(args.subway_dual_channels),
    )
    print_method_test_banners(
        split=split_s,
        dncnn_keys=dncnn_keys,
        gan_keys=gan_keys,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
        shuffle_split=bool(args.shuffle_split),
    )
    print(
        "[INFO] Per-method test sets: inline 8:2 holdout from seed/train_ratio (same split rules for DnCNN and TraMagNet on this data-root).",
        flush=True,
    )

    keys_by_method: dict[str, set[tuple[str, str]] | None] = {}
    for m in methods:
        keys_by_method[m] = dncnn_keys if m == DNCNN_METHOD else gan_keys

    dncnn_inferer, gan_inferer, unet_single_inferer = init_pt_inferers(
        args=args,
        methods=methods,
        data_root=data_root,
        dataset_tag=dataset_tag,
        denorm_mode=str(args.denorm),
    )
    if DNCNN_METHOD in methods or GAN_METHOD in methods or UNET_SINGLE_METHOD in methods:
        print(
            "[INFO] dncnn/TraMagNet/UNet-only ablation: K-fold .pt on-the-fly inference (does not read output/dncnn|TraMagNet|UNet-only ablation/result).",
            flush=True,
        )

    _, summary_rows = evaluate_methods_on_data1(
        reference_dir=reference_dir,
        noise_dir=noise_dir,
        output_root=output_root,
        methods=methods,
        skip_missing_method_dir=(not args.strict),
        denorm_mode=str(args.denorm),
        dataset_tag=dataset_tag,
        include_snr_triplets=False,
        allowed_segment_keys_by_method=keys_by_method,
        baseline_segment_keys=gan_keys,
        segment_length=int(args.segment_length),
        dncnn_inferer=dncnn_inferer,
        gan_inferer=gan_inferer,
        unet_single_inferer=unet_single_inferer,
    )
    print_summary_table(summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

