#!/usr/bin/env python3
"""
Batch SNR evaluation for checkpoint directories (``--mode runs-root``) and exported txt results (``--mode report``).

Uses in-repo ``TraMagNet`` only to load ``OurDataDataset``, ``UNet`` (checkpoint format must match).

**SNR aligned with ``eval_metrics.evaluate_methods_on_data1`` (checkpoint path)**

- **Forward pass**: same as ``visualize_data.py`` → ``viz_tramagnet_runner`` — ``model(noisy,z)`` on batch ``noisy``,
  yielding tensor with **same semantics** as ``d.squeeze()`` when exporting result txt (not written to disk).
- **Three SNR values**: ``clean`` / ``noisy`` both read from **txt** (same as ``evaluate_methods_on_data1``); z-domain denoised
  ``den_1024`` is forward ``pred``, denormalized via ``eval_metrics.snr_triplets_for_segment_like_evaluate_methods_on_data1``
  to compute ``SNR_time`` / ``SNR_freq`` / ``SNR_joint`` (**not** dataloader ``n_z*σ+μ`` as noisy).

**Default (``--mode report``)**: summarize **txt results** under ``output/<method>/…`` and print three SNR per method.

**``--mode runs-root``**: evaluate checkpoint table under a single ``--runs-root`` (e.g. ``TraMagNet/output/data134/runs``).

Forward per sample on dataset (``--split`` default ``test``=fixed 20%% hold-out; ``all``=no split filter), aggregate three SNR (mean).

Inference z: **always all-zero for denoising / evaluation** (independent of ``randz``/``zeroz`` in run dir names).
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Iterator

import numpy as np


def _hide_cuda_if_cpu_device_argv() -> None:
    """Run before ``import torch``: if CLI requests CPU, hide GPU to avoid loading large DLLs like cuBLAS."""
    if os.environ.get("LOSS_EVAL_DEVICE", "").strip().lower() in ("cpu", "1", "yes", "true"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--device" and i + 1 < len(args) and str(args[i + 1]).strip().lower() == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            return
        if a.startswith("--device=") and a.split("=", 1)[1].strip().lower() == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            return


_hide_cuda_if_cpu_device_argv()

try:
    import torch
except OSError as e:
    msg = str(e).lower()
    if sys.platform == "win32" and (
        "cublas" in msg
        or "cuda" in msg
        or "1455" in str(e)
        or "page file" in str(e).lower()
        or "pagefile" in msg
    ):
        print(
            "[FATAL] PyTorch failed to load CUDA dynamic libraries (WinError 1455 usually means insufficient page file/commit memory).\n"
            "  System: Control Panel → System → Advanced → Performance → Advanced → Virtual memory — increase page file and reboot.\n"
            "  Runtime (skip CUDA, CPU only):\n"
            "    ① Add ``--device cpu`` to the command (this script hides GPU before import torch).\n"
            "    ② If still failing, run in terminal first then rerun the same command:\n"
            "       PowerShell:  $env:CUDA_VISIBLE_DEVICES=''\n"
            "       CMD:         set CUDA_VISIBLE_DEVICES=\n"
            "       Then:        python loss_eval.py ... --device cpu\n",
            file=sys.stderr,
            flush=True,
        )
    raise

_REPO = Path(__file__).resolve().parent

_TRAMAGNET = _REPO / "TraMagNet"

_SLUG_ORDER = [
    "msestft_2_8",
    "msestft_4_6",
    "msestft_5_5",
    "msestft_6_4",
    "msestft_8_2",
]
_RUN_DIR_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<slug>msestft_\d+_\d+|l1only|mseonly|stftonly|l1mse|"
    r"l1msestft_0_4_6|msestft_0_4_6|l1msestft_4_4_2|msestft_128_0_8)_"
    r"(?P<z>randz|zeroz)_e(?P<ep>\d+)$"
)

_SLUG_LABEL: dict[str, str] = {
    "msestft_2_8": "MSE+STFT norm(2-8)",
    "msestft_4_6": "MSE+STFT norm(4-6)",
    "msestft_5_5": "MSE+STFT norm(5-5)",
    "msestft_6_4": "MSE+STFT norm(6-4)",
    "msestft_8_2": "MSE+STFT norm(8-2)",
}


def _slug_label(slug: str) -> str:
    if slug in _SLUG_LABEL:
        return _SLUG_LABEL[slug]
    m = re.fullmatch(r"msestft_(\d+)_(\d+)", slug)
    if m:
        return f"MSE+STFT norm({m.group(1)}-{m.group(2)})"
    return slug


def _display_name_from_run_dir(name: str) -> str:
    m = _RUN_DIR_RE.match(name)
    if not m:
        return name[:40]
    slug = m.group("slug")
    zpart = m.group("z")
    base = _slug_label(slug)
    zcn = "z random" if zpart == "randz" else "z zero"
    return f"{base} | {zcn}"


def _sort_key_run_dir(name: str) -> tuple:
    m = _RUN_DIR_RE.match(name)
    if not m:
        return (99, name)
    slug = m.group("slug")
    zpart = m.group("z")
    ep = int(m.group("ep"))
    try:
        si = _SLUG_ORDER.index(slug)
    except ValueError:
        si = 98
    zi = 0 if zpart == "randz" else 1
    return (si, zi, ep, name)


def _format_float(x: float) -> str:
    if not np.isfinite(x):
        return str(x)
    ax = abs(x)
    if (ax != 0.0 and ax < 1e-3) or ax >= 1e6:
        return f"{x:.6e}"
    return f"{x:.6f}"


NAME_COL_W = 42


def _name_cell(s: str, *, w: int = NAME_COL_W) -> str:
    t = str(s)
    if len(t) > w:
        t = (t[: w - 3] + "...") if w > 3 else t[:w]
    return t.ljust(w)


def _phys_clean_noisy_z_from_batch(batch: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``reference_phys``, physical noisy ``noisy_phys``, z-domain noisy ``noisy_z`` (same as ``OurDataDataset``)."""
    n_z = np.asarray(batch["noisy"].squeeze(0).detach().cpu().numpy(), dtype=np.float64).ravel()
    cp = batch.get("reference_phys")
    if cp is None:
        c_phys = np.asarray(batch["reference"].squeeze(0).detach().cpu().numpy(), dtype=np.float64).ravel()
        return c_phys, n_z, n_z
    c_phys = np.asarray(cp.squeeze(0).detach().cpu().numpy(), dtype=np.float64).ravel()
    mu = float(np.mean(c_phys))
    sig = float(np.std(c_phys)) + 1e-6
    n_phys = n_z * sig + mu
    return c_phys, n_phys, n_z


def txt_reference_noisy_1024_from_ds_sample(ds: object, ii: int) -> tuple[np.ndarray, np.ndarray]:
    """Same as ``evaluate_methods_on_data1``: read segment from ``reference_signal`` / ``noise_signal`` **txt** and resample."""
    from eval_metrics import _load_time_series_values_txt, _resample_to_len

    if not hasattr(ds, "_pairs") or not hasattr(ds, "_indices"):
        raise AttributeError("txt_reference_noisy_1024_from_ds_sample requires OurDataDataset (with _pairs/_indices)")
    *_meta, c_fn, n_fn, vcol = ds._pairs[ds._indices[int(ii)]]  # type: ignore[attr-defined]
    seg = int(getattr(getattr(ds, "cfg", None), "segment_length", 1024))
    reference_1024 = _resample_to_len(_load_time_series_values_txt(Path(c_fn), value_column=int(vcol)), seg)
    noisy_1024 = _resample_to_len(_load_time_series_values_txt(Path(n_fn), value_column=int(vcol)), seg)
    return reference_1024, noisy_1024


def den_z_from_pred(pred: torch.Tensor) -> np.ndarray:
    """Model output in z-domain (same as ``d.squeeze()`` written to result column 3 by viz)."""
    return np.asarray(pred.squeeze(0).detach().cpu().numpy(), dtype=np.float64).ravel()


def snr_triplets_from_ds_sample(
    ds: object,
    ii: int,
    pred: torch.Tensor,
    *,
    denorm_mode: str,
    snr_sample_rate_hz: float,
    f_cut_hz: float,
    joint_alpha: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Three SNR values per segment (denoised + noisy); noisy/clean both from **txt**.

    Returns:
        ``((snr_time, snr_freq, snr_joint)_den, (snr_noisy_time, snr_noisy_freq, snr_noisy_joint))``
    """
    from eval_metrics import snr_triplets_for_segment_like_evaluate_methods_on_data1

    reference_1024, noisy_1024 = txt_reference_noisy_1024_from_ds_sample(ds, ii)
    _, den_trip, noisy_trip = snr_triplets_for_segment_like_evaluate_methods_on_data1(
        reference_1024=reference_1024,
        noisy_1024=noisy_1024,
        den_1024=den_z_from_pred(pred),
        denorm_mode=str(denorm_mode),
        snr_sample_rate_hz=float(snr_sample_rate_hz),
        snr_f_cut_hz=float(f_cut_hz),
        snr_joint_alpha=float(joint_alpha),
    )
    return den_trip, noisy_trip


def _phys_clean_noisy_and_den_z_from_batch_like_viz(
    batch: dict,
    pred: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Legacy helper: returns ``(reference_phys, noisy_phys, den_z)``.

    **For SNR evaluation use** ``snr_triplets_from_ds_sample`` (noisy from txt). ``noisy_phys`` here is still
    ``n_z*σ+μ``, for non-SNR visualization/debug only.
    """
    c_phys, n_phys, _n_z = _phys_clean_noisy_z_from_batch(batch)
    d_z = den_z_from_pred(pred)
    return c_phys, n_phys, d_z


def _snr_freq_db(
    x: np.ndarray,
    *,
    sample_rate_hz: float,
    f_cut_hz: float,
    eps: float = 1e-18,
) -> float:
    """One-sided spectrum: S = (0, f_cut], N = (f_cut, Nyquist]; zero-mean + Hann."""
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
    # Signal band: just above 0 to f_cut; noise band: above f_cut
    lo = (freqs > 1e-6) & (freqs <= float(f_cut_hz))
    hi = freqs > float(f_cut_hz)
    ps = float(np.sum(p[lo]))
    pn = float(np.sum(p[hi]))
    if pn <= eps:
        return float("inf") if ps > eps else float("nan")
    return float(10.0 * math.log10((ps + eps) / (pn + eps)))


def _snr_joint_db(snr_t: float, snr_f: float, *, alpha: float, eps: float = 1e-18) -> float:
    if not (np.isfinite(snr_t) and np.isfinite(snr_f)):
        return float("nan")
    gt = 10.0 ** (float(snr_t) / 10.0)
    gf = 10.0 ** (float(snr_f) / 10.0)
    a = float(alpha)
    a = min(1.0, max(0.0, a))
    gj = a * gt + (1.0 - a) * gf
    return float(10.0 * math.log10(gj + eps))


def _iter_run_dirs(runs_root: Path, *, cv_folds: int = 5) -> list[Path]:
    """List grid job directories under ``runs_root`` (including K-fold ``runs/fold_*`` layout)."""
    from data_common.cv_ensemble import job_dir_has_ckpt

    if not runs_root.is_dir():
        return []
    out: list[Path] = []
    for p in runs_root.iterdir():
        if not p.is_dir():
            continue
        name = p.name.lower()
        if name.startswith("fold_"):
            continue
        if job_dir_has_ckpt(p, cv_folds=int(cv_folds)):
            out.append(p)
    return sorted(out, key=lambda x: _sort_key_run_dir(x.name))


def _setup_unet_eval_imports() -> None:
    """Add repo root and ``TraMagNet`` to ``sys.path`` so ``data.*`` / ``models.*`` match training."""
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    tramagnet = _REPO / "TraMagNet"
    if str(tramagnet) not in sys.path:
        sys.path.insert(0, str(tramagnet))

def _datasets_for_split(
    *,
    data_root: str,
    reference_subdir: str,
    noisy_subdir: str,
    band: str,
    segment_length: int,
    train_ratio: float,
    seed: int,
    shuffle_split: bool,
    split: str,
    cv_folds: int = 0,
    cv_fold: int = 0,
    allowed_segment_keys: set[tuple[str, str]] | None = None,
):
    """
    Build ``OurDataDataset`` list.

    If ``allowed_segment_keys`` provided (merged manifest test/holdout), load **full library pairs**
    (``split=all``, ``strict_all_bands=False``, same as ``build_eval_segment_keys``),
    then filter by manifest keys in ``_iter_indices``; do not intersect with single-dataset ``holdout_eval``.
    """
    _setup_unet_eval_imports()
    from data.our_data_dataset import OurDataConfig, OurDataDataset

    split_s = str(split).lower().strip()
    strict_all_bands = True
    cv_folds_eff = int(cv_folds)
    cv_fold_eff = int(cv_fold)
    if allowed_segment_keys is not None and split_s in ("test", "holdout"):
        split_s = "all"
        strict_all_bands = False
        # Manifest filter needs full library pairs; do not use K-fold subset (else count << txt section)
        cv_folds_eff = 0
        cv_fold_eff = 0

    common = dict(
        root=data_root,
        reference_subdir=reference_subdir,
        noisy_subdir=noisy_subdir,
        band=band,  # type: ignore[arg-type]
        segment_length=int(segment_length),
        train_ratio=float(train_ratio),
        seed=int(seed),
        shuffle_split=bool(shuffle_split),
        split_round=True,
        cv_folds=cv_folds_eff,
        cv_fold=cv_fold_eff,
        resample_mode="resample_linear",
        strict_all_bands=strict_all_bands,
        match_noisy_scale_to_reference=False,
        zscore_using_reference=False,
    )
    if split_s == "all":
        return [
            OurDataDataset(OurDataConfig(**common, train=True)),
            OurDataDataset(OurDataConfig(**common, train=False, holdout_eval=True)),
        ]
    if split_s in ("test", "holdout"):
        return [OurDataDataset(OurDataConfig(**common, train=False, holdout_eval=True))]
    if split_s == "train":
        return [OurDataDataset(OurDataConfig(**common, train=True))]
    if split_s == "cv_train":
        if int(cv_folds) <= 0:
            raise ValueError("split=cv_train requires cv_folds > 0")
        return [OurDataDataset(OurDataConfig(**common, train=True))]
    if split_s == "cv_val":
        if int(cv_folds) <= 0:
            raise ValueError("split=cv_val requires cv_folds > 0")
        return [OurDataDataset(OurDataConfig(**common, train=False))]
    raise ValueError("split must be all|test|holdout|train|cv_train|cv_val")


def _segment_key_from_dataset_index(ds, ii: int) -> tuple[str, str]:
    """Same as ``build_eval_segment_keys`` ``(noisy filename, ch0|ch1)``."""
    from pathlib import Path

    from data_common.eval_split import _channel_tag_for_pair

    flat_idx = ds._indices[int(ii)]
    row = ds._pairs[flat_idx]
    n_fn = row[-2]
    vcol = row[-1]
    ch = _channel_tag_for_pair(value_column=int(vcol), noisy_path=Path(n_fn))
    return (Path(n_fn).name, ch)


def _iter_indices(
    dss: list,
    max_samples: int,
    *,
    allowed_segment_keys: set[tuple[str, str]] | None = None,
) -> Iterator[tuple[int, int]]:
    """(ds_idx, idx); when ``allowed_segment_keys`` non-empty, same manifest filter as txt section."""
    n_done = 0
    for di, ds in enumerate(dss):
        for i in range(len(ds)):
            if allowed_segment_keys is not None:
                if _segment_key_from_dataset_index(ds, i) not in allowed_segment_keys:
                    continue
            yield di, i
            n_done += 1
            if max_samples > 0 and n_done >= max_samples:
                return


@torch.no_grad()
def _evaluate_one_run(
    *,
    run_dir: Path,
    ckpt_prefer: str,
    dss: list,
    device: torch.device,
    sample_rate_hz: float,
    f_cut_hz: float,
    joint_alpha: float,
    eval_seed: int,
    denorm_mode: str,
    max_samples: int,
    allowed_segment_keys: set[tuple[str, str]] | None = None,
) -> tuple[str, int, float, float, float]:
    _ = int(eval_seed)  # Same arg as ztest5 / eval_metrics; forward z always 0, no longer used for sampling
    _setup_unet_eval_imports()
    from models.unet import UNET_LATENT_CHANNELS, UNET_LATENT_LENGTH, UNet, complete_unet_state_dict

    from data_common.cv_ensemble import pick_ckpt_in_dir

    ckpt_path = pick_ckpt_in_dir(run_dir, ckpt_prefer)
    try:
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(ckpt_path, map_location=device)

    if isinstance(payload, dict) and "model" in payload:
        sd = payload["model"]
    elif isinstance(payload, dict) and "generator" in payload:
        sd = payload["generator"]
    elif isinstance(payload, dict):
        raise ValueError(f"unsupported checkpoint keys in {ckpt_path}: {list(payload.keys())}")
    else:
        sd = payload

    model = UNet().to(device)
    model.load_state_dict(complete_unet_state_dict(model, sd), strict=True)
    model.eval()

    name = run_dir.name
    st_list: list[float] = []
    sf_list: list[float] = []
    sj_list: list[float] = []
    n_used = 0

    def _forward_zero_z(batch_noisy: torch.Tensor) -> torch.Tensor:
        """Independent of run dir ``randz``/``zeroz``: latent always 0 at eval (same as ``--z-mode zero``)."""
        bsz = batch_noisy.size(0)
        z = torch.zeros(
            bsz,
            UNET_LATENT_CHANNELS,
            UNET_LATENT_LENGTH,
            device=device,
            dtype=batch_noisy.dtype,
        )
        raw = model(batch_noisy, z)
        return raw

    for di, ii in _iter_indices(dss, max_samples, allowed_segment_keys=allowed_segment_keys):
        ds = dss[di]
        batch = ds[ii]
        noisy = batch["noisy"].unsqueeze(0).to(device)
        pred = _forward_zero_z(noisy)

        (st, sf, sj), _ = snr_triplets_from_ds_sample(
            ds,
            ii,
            pred,
            denorm_mode=str(denorm_mode),
            snr_sample_rate_hz=float(sample_rate_hz),
            f_cut_hz=float(f_cut_hz),
            joint_alpha=float(joint_alpha),
        )
        if np.isfinite(st):
            st_list.append(float(st))
        if np.isfinite(sf):
            sf_list.append(float(sf))
        if np.isfinite(sj):
            sj_list.append(float(sj))
        n_used += 1

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    disp = _display_name_from_run_dir(name)
    return disp, n_used, _mean(st_list), _mean(sf_list), _mean(sj_list)


@torch.no_grad()
def _evaluate_cv_ensemble_unet(
    *,
    ckpt_paths: list[Path],
    display_name: str,
    dss: list,
    device: torch.device,
    sample_rate_hz: float,
    f_cut_hz: float,
    joint_alpha: float,
    denorm_mode: str,
    max_samples: int,
    allowed_segment_keys: set[tuple[str, str]] | None = None,
) -> tuple[str, int, float, float, float]:
    """K-fold weight ensemble: pred = mean(pred_1..pred_K), z always 0 (same as single-model eval)."""
    _setup_unet_eval_imports()
    from data_common.ensemble_infer import load_unet_ensemble, unet_ensemble_forward, unet_make_z

    try:
        probe = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)
    except TypeError:
        probe = torch.load(ckpt_paths[0], map_location="cpu")
    gan = isinstance(probe, dict) and "generator" in probe

    members = load_unet_ensemble(ckpt_paths, device, gan_generator=gan)

    st_list: list[float] = []
    sf_list: list[float] = []
    sj_list: list[float] = []
    n_used = 0

    for di, ii in _iter_indices(dss, max_samples, allowed_segment_keys=allowed_segment_keys):
        ds = dss[di]
        batch = ds[ii]
        noisy = batch["noisy"].unsqueeze(0).to(device)
        bsz = noisy.size(0)
        z = unet_make_z(bsz, device, noisy.dtype, "zero")
        pred = unet_ensemble_forward(members, noisy, z)

        (st, sf, sj), _ = snr_triplets_from_ds_sample(
            ds,
            ii,
            pred,
            denorm_mode=str(denorm_mode),
            snr_sample_rate_hz=float(sample_rate_hz),
            f_cut_hz=float(f_cut_hz),
            joint_alpha=float(joint_alpha),
        )
        if np.isfinite(st):
            st_list.append(float(st))
        if np.isfinite(sf):
            sf_list.append(float(sf))
        if np.isfinite(sj):
            sj_list.append(float(sj))
        n_used += 1

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    return str(display_name), n_used, _mean(st_list), _mean(sf_list), _mean(sj_list)


def _evaluate_noisy_baseline(
    *,
    dss: list,
    sample_rate_hz: float,
    f_cut_hz: float,
    joint_alpha: float,
    denorm_mode: str,
    max_samples: int,
    allowed_segment_keys: set[tuple[str, str]] | None = None,
) -> tuple[str, int, float, float, float]:
    """Noisy row: same as ``evaluate_methods_on_data1`` — read reference/noisy from **txt**."""
    from eval_metrics import snr_noisy_triplet_for_segment_like_evaluate_methods_on_data1

    _ = denorm_mode  # Same as txt convention; noisy triplet SNR does not use ``_maybe_denormalize_like_training``

    st_list: list[float] = []
    sf_list: list[float] = []
    sj_list: list[float] = []
    n_used = 0
    for di, ii in _iter_indices(dss, max_samples, allowed_segment_keys=allowed_segment_keys):
        ds = dss[di]
        reference_1024, noisy_1024 = txt_reference_noisy_1024_from_ds_sample(ds, ii)
        st, sf, sj = snr_noisy_triplet_for_segment_like_evaluate_methods_on_data1(
            reference_1024=reference_1024,
            noisy_1024=noisy_1024,
            snr_sample_rate_hz=float(sample_rate_hz),
            snr_f_cut_hz=float(f_cut_hz),
            snr_joint_alpha=float(joint_alpha),
        )
        if np.isfinite(st):
            st_list.append(float(st))
        if np.isfinite(sf):
            sf_list.append(float(sf))
        if np.isfinite(sj):
            sj_list.append(float(sj))
        n_used += 1

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    return "Noisy baseline", n_used, _mean(st_list), _mean(sf_list), _mean(sj_list)


def format_snr_table_rows(
    rows: list[tuple[str, int, float, float, float]],
    *,
    name_col_w: int | None = None,
) -> str:
    """ASCII table (same as printed by ``_print_table``) for writing to a file."""
    w0 = int(name_col_w) if name_col_w is not None else NAME_COL_W
    headers = ["model", "count", "snr_time_db", "snr_freq_db", "snr_joint_db"]
    table: list[list[str]] = []
    for name, cnt, st, sf, sj in rows:
        table.append(
            [
                str(name),
                str(int(cnt)),
                _format_float(float(st)),
                _format_float(float(sf)),
                _format_float(float(sj)),
            ]
        )
    widths = [len(h) for h in headers]
    widths[0] = w0
    for row in table:
        for i, cell in enumerate(row):
            if i == 0:
                continue
            widths[i] = max(widths[i], len(cell))

    def line(items: list[str]) -> str:
        m = _name_cell(items[0], w=w0)
        c = items[1].ljust(widths[1])
        rest = " | ".join(items[i].ljust(widths[i]) for i in range(2, len(items)))
        return f"{m}|{c} | {rest}"

    sep0 = "-" * w0 + "+" + "-" * widths[1]
    sep_rest = "-+-".join("-" * widths[i] for i in range(2, len(widths)))
    sep = sep0 + "-+-" + sep_rest

    out_lines = [
        "[legend] SNR_time: after denorm, clean & signal each zero-mean, "
        "10*log10(sum(clean^2)/sum((den-clean)^2)). "
        "SNR_freq: Hann-windowed rFFT of den; power ratio (0,f_cut] vs (f_cut, Nyquist]. "
        "SNR_joint: 10*log10(alpha*10^(SNR_t/10)+(1-alpha)*10^(SNR_f/10)).",
        line(headers),
        sep,
    ]
    for row in table:
        out_lines.append(line(row))
    return "\n".join(out_lines) + "\n"


def _print_table(rows: list[tuple[str, int, float, float, float]]) -> None:
    print(format_snr_table_rows(rows), end="", flush=True)


def _print_txt_methods_snr_one_line_each(
    summary_rows: list[dict],
    methods: list[str],
    *,
    fmt,
) -> None:
    """Same fields as ``eval_metrics`` summary: one row per method + three SNR (dB)."""
    bym = {str(r["method"]): r for r in summary_rows}
    print(
        "[txt methods] Each row: method<TAB>snr_time_db<TAB>snr_freq_db<TAB>snr_joint_db "
        "(same as eval_metrics summary snr_*_db_mean; includes Noisy baseline if present).",
        flush=True,
    )
    print("", flush=True)

    order = ["Noisy"] + [m for m in methods if m != "Noisy"]
    present = [m for m in order if bym.get(m) is not None]
    if not present:
        print("[INFO] No txt method summary rows (summary empty or method names mismatch).", flush=True)
        print("", flush=True)
        return

    col_w = max(len("method"), max(len(str(m)) for m in present))

    print(
        f"{'method'.ljust(col_w)}\tsnr_time_db\tsnr_freq_db\tsnr_joint_db",
        flush=True,
    )
    print("", flush=True)

    for m in order:
        r = bym.get(m)
        if r is None:
            continue
        print(
            f"{str(m).ljust(col_w)}\t{fmt(float(r['snr_time_db_mean']))}\t"
            f"{fmt(float(r['snr_freq_db_mean']))}\t{fmt(float(r['snr_joint_db_mean']))}",
            flush=True,
        )
    print("", flush=True)


def _resolve_repo_path_str(p: str | Path) -> Path:
    x = Path(p).expanduser()
    return x.resolve() if x.is_absolute() else (_REPO / x).resolve()


def _main_runs_root_only(args: argparse.Namespace) -> int:
    """Evaluate checkpoint under each subdir of ``--runs-root`` only (legacy single table)."""
    from data_common.resolve_dataset_root import resolve_dataset_root

    data_root = resolve_dataset_root(args.data_root, repo=_REPO)
    runs_root = Path(args.runs_root).expanduser()
    runs_root = runs_root.resolve() if runs_root.is_absolute() else (_REPO / runs_root).resolve()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dss = _datasets_for_split(
        data_root=data_root,
        reference_subdir=str(args.reference_subdir),
        noisy_subdir=str(args.noisy_subdir),
        band=str(args.band),
        segment_length=int(args.segment_length),
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
        shuffle_split=bool(args.shuffle_split),
        split=str(args.split),
        cv_folds=int(args.cv_folds),
        cv_fold=int(args.cv_fold),
    )
    max_samples = max(0, int(args.max_samples))

    rows: list[tuple[str, int, float, float, float]] = []
    rows.append(
        _evaluate_noisy_baseline(
            dss=dss,
            sample_rate_hz=float(args.sample_rate_hz),
            f_cut_hz=float(args.f_cut_hz),
            joint_alpha=float(args.joint_alpha),
            denorm_mode=str(args.denorm),
            max_samples=max_samples,
        )
    )

    from data_common.cv_ensemble import (
        cv_ensemble_folds_from_args,
        list_available_fold_ckpt_paths,
        list_fold_ckpt_paths,
    )

    nf = cv_ensemble_folds_from_args(args, default=int(args.cv_folds))
    ens_paths: list[Path] | None = None
    if nf >= 2:
        ens_paths = list_fold_ckpt_paths(runs_root, cv_folds=nf, prefer=str(args.ckpt))
        if ens_paths is None:
            ens_paths = list_available_fold_ckpt_paths(runs_root, cv_folds=nf, prefer=str(args.ckpt))
            if ens_paths is not None and len(ens_paths) < nf:
                print(
                    f"[INFO] K-fold incomplete, using {len(ens_paths)}/{nf}-fold ensemble: {runs_root}",
                    flush=True,
                )

    if ens_paths is not None and len(ens_paths) >= 1:
        try:
            rows.append(
                _evaluate_cv_ensemble_unet(
                    ckpt_paths=ens_paths,
                    display_name=f"cv{nf}_ensemble",
                    dss=dss,
                    device=device,
                    sample_rate_hz=float(args.sample_rate_hz),
                    f_cut_hz=float(args.f_cut_hz),
                    joint_alpha=float(args.joint_alpha),
                    denorm_mode=str(args.denorm),
                    max_samples=max_samples,
                )
            )
            print(
                f"[ensemble] {runs_root}: averaging {len(ens_paths)}-fold checkpoints",
                flush=True,
            )
        except Exception as e:
            print(f"[WARN] K-fold ensemble eval failed, falling back to per-subdir eval: {e}", flush=True)
            ens_paths = None

    if ens_paths is None:
        try:
            rows.append(
                _evaluate_one_run(
                    run_dir=runs_root,
                    ckpt_prefer=str(args.ckpt),
                    dss=dss,
                    device=device,
                    sample_rate_hz=float(args.sample_rate_hz),
                    f_cut_hz=float(args.f_cut_hz),
                    joint_alpha=float(args.joint_alpha),
                    eval_seed=int(args.eval_seed),
                    denorm_mode=str(args.denorm),
                    max_samples=max_samples,
                )
            )
        except Exception:
            pass

        run_dirs = _iter_run_dirs(runs_root)
        if not run_dirs:
            print(
                f"[WARN] No subdirs with best.pt/last.pt under runs-root: {runs_root}",
                flush=True,
            )

        for rd in run_dirs:
            try:
                rows.append(
                    _evaluate_one_run(
                        run_dir=rd,
                        ckpt_prefer=str(args.ckpt),
                        dss=dss,
                        device=device,
                        sample_rate_hz=float(args.sample_rate_hz),
                        f_cut_hz=float(args.f_cut_hz),
                        joint_alpha=float(args.joint_alpha),
                        eval_seed=int(args.eval_seed),
                        denorm_mode=str(args.denorm),
                        max_samples=max_samples,
                    )
                )
            except Exception as e:
                print(f"[WARN] Skipping {rd.name}: {e}", flush=True)

    _print_table(rows)
    return 0


def _main_report(args: argparse.Namespace) -> int:
    """txt method three SNR from exported ``output/`` results."""
    import eval_metrics as em
    from data_common.dataset_paths import dataset_tag_for_path
    from data_common.resolve_dataset_root import resolve_dataset_root

    data_root = Path(resolve_dataset_root(args.data_root, repo=_REPO))
    reference_dir = data_root / "reference_signal"
    noise_dir = data_root / "noise_signal"
    output_root = Path(args.output_root).expanduser()
    output_root = output_root.resolve() if output_root.is_absolute() else (_REPO / output_root).resolve()
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    dataset_tag = str(args.dataset_tag).strip() if args.dataset_tag else dataset_tag_for_path(data_root)

    from data_common.viz_method_splits import (
        build_dncnn_test_segment_keys,
        build_gan_test_segment_keys,
        print_method_test_banners,
    )

    split_s = str(args.split).lower().strip()
    dncnn_keys = build_dncnn_test_segment_keys(
        data_root,
        split=split_s,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
        shuffle_split=bool(args.shuffle_split),
        band=str(args.band),
        subway_dual_channels=bool(getattr(args, "subway_dual_channels", True)),
    )
    gan_keys = build_gan_test_segment_keys(
        data_root,
        _REPO,
        split=split_s,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
        shuffle_split=bool(args.shuffle_split),
        band=str(args.band),
        subway_dual_channels=bool(getattr(args, "subway_dual_channels", True)),
    )
    print_method_test_banners(
        split=split_s,
        dncnn_keys=dncnn_keys,
        gan_keys=gan_keys,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
        shuffle_split=bool(args.shuffle_split),
    )

    keys_by_method = {m: (dncnn_keys if m == em.DNCNN_METHOD else gan_keys) for m in methods}

    dncnn_inferer, gan_inferer, unet_single_inferer = em.init_pt_inferers(
        args=args,
        methods=methods,
        data_root=data_root,
        dataset_tag=dataset_tag,
        denorm_mode=str(args.denorm),
    )

    _, summary_rows = em.evaluate_methods_on_data1(
        reference_dir=reference_dir,
        noise_dir=noise_dir,
        output_root=output_root,
        methods=methods,
        skip_missing_method_dir=(not args.strict),
        denorm_mode=str(args.denorm),
        dataset_tag=dataset_tag,
        snr_sample_rate_hz=float(args.sample_rate_hz),
        snr_f_cut_hz=float(args.f_cut_hz),
        snr_joint_alpha=float(args.joint_alpha),
        include_snr_triplets=True,
        allowed_segment_keys_by_method=keys_by_method,
        baseline_segment_keys=gan_keys,
        segment_length=int(args.segment_length),
        dncnn_inferer=dncnn_inferer,
        gan_inferer=gan_inferer,
        unet_single_inferer=unet_single_inferer,
    )

    print("", flush=True)
    print("========== txt methods (eval_metrics & output/ results) ==========", flush=True)
    print("", flush=True)
    _print_txt_methods_snr_one_line_each(summary_rows, methods, fmt=em._format_float)
    print("", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))

    p = argparse.ArgumentParser(
        description="``--mode report``: txt method three SNR from ``output/`` exports; "
        "``--mode runs-root``: evaluate checkpoints under ``--runs-root`` only."
    )
    p.add_argument(
        "--mode",
        type=str,
        default="report",
        choices=("report", "runs-root"),
        help="report: txt SNR summary; runs-root: checkpoint table under ``--runs-root`` only.",
    )
    p.add_argument("--data-root", type=str, default="data1", help="Data root (data1/data3/data4 or ../datasets/…)")
    p.add_argument("--reference-subdir", type=str, default="reference_signal", dest="reference_subdir")
    p.add_argument("--noisy-subdir", type=str, default="noise_signal")
    p.add_argument("--band", type=str, default="all", choices=("low", "middle", "high", "all"))
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--shuffle-split", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--segment-length", type=int, default=1024)
    p.add_argument("--denorm", type=str, default="always", choices=["always", "never", "auto"])
    p.add_argument("--sample-rate-hz", type=float, default=360.0)
    p.add_argument("--f-cut-hz", type=float, default=20.0)
    p.add_argument("--joint-alpha", type=float, default=0.5)
    p.add_argument(
        "--ckpt",
        type=str,
        default="last",
        choices=("best", "last"),
        help="Prefer checkpoint in subdir: default last.pt; ``--ckpt best`` uses best.pt.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="cuda / cpu; on Windows use cpu for CUDA DLL/1455 page-file errors (must appear in argv to hide GPU before import).",
    )
    p.add_argument(
        "--eval-seed",
        type=int,
        default=12345,
        help="Placeholder: UNet eval forward z always 0.",
    )
    p.add_argument("--max-samples", type=int, default=0, help="0=all")

    p.add_argument(
        "--output-root",
        type=str,
        default="./output",
        help="[report] Output root with per-method ``result`` (same as eval_metrics --output-root).",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="TraMagNet,dncnn,gradient_wavelet_morphological_filter,multi_se_morphological_filter",
        help="[report] Comma-separated method names (same as eval_metrics --methods).",
    )
    p.add_argument("--dataset-tag", type=str, default=None, help="[report] Default: data-root directory name.")
    p.add_argument("--strict", action="store_true", help="[report] Raise if method directory missing.")

    p.add_argument(
        "--runs-root",
        type=str,
        default=str(_TRAMAGNET / "output" / "data134" / "runs"),
        help="[runs-root mode] Checkpoint runs directory (e.g. TraMagNet/output/data134/runs).",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("all", "test", "holdout", "train", "cv_train", "cv_val"),
        help="Data split: default test=fixed 20%% hold-out (shared by txt and checkpoint sections); all=no split filter on result.",
    )
    p.add_argument(
        "--subway-dual-channels",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="subway_dual_channels",
        help="Whether data3 subway uses dual channels when building txt eval split keys.",
    )
    p.add_argument("--cv-folds", type=int, default=5, dest="cv_folds")
    p.add_argument("--cv-fold", type=int, default=0, dest="cv_fold")
    import eval_metrics as em

    em.add_pt_inference_arguments(p, include_device_ckpt=False)

    args = p.parse_args(argv)

    if str(args.mode) == "runs-root":
        return _main_runs_root_only(args)
    return _main_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
