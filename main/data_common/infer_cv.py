"""K-fold checkpoint inference for CNN, TraMagNet, and UNet-only ablation."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_MAIN = Path(__file__).resolve().parent.parent
_REPO = _MAIN
_PUBLIC = _MAIN.parent
_TRAMAGNET = _REPO / "TraMagNet"
_CNN = _REPO / "cnn"
_UNET8 = _REPO / "unet8"
_UNET_SINGLE = _PUBLIC / "ablation" / "unet_single"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _prepend_nn_to_syspath(nn_dir: Path) -> None:
    """Each NN subdir has ``models`` / ``data`` packages; when switching inference roots, refresh ``sys.path`` and drop cached same-name modules."""
    p = str(nn_dir.resolve())
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
    for name in list(sys.modules):
        if name == "models" or name.startswith("models.") or name == "data" or name.startswith("data."):
            del sys.modules[name]

from data_common.cv_ensemble import (
    InferenceCkptPlan,
    add_cv_ensemble_arguments,
    cv_ensemble_folds_from_args,
    fold_dir_for_config,
    list_fold_ckpt_paths,
)
from data_common.eval_split import (
    noisy_in_eval_split,
    resolve_eval_split_manifest_path,
    segment_in_eval_split,
)
from data_common.viz_method_splits import (
    build_cnn_test_segment_keys,
    build_gan_test_segment_keys,
    format_overlap_split_banner,
    intersect_segment_keys,
    list_noisy_files_for_segment_keys,
)
from data_common.txt_io import (
    pad_or_resample_to_length,
    read_one_file_with_meta,
    read_two_channel_file,
    subway_noisy_has_four_value_columns,
)
from data_common.viz_ckpt_resolve import add_viz_job_arguments, resolve_viz_inference_plan
from data_common.viz_split import BooleanOptionalAction, maybe_sync_split_from_runs_config
from eval_metrics import (
    _reference_filename_from_noisy_or_result,
    _load_time_series_values_txt,
    _maybe_denormalize_like_training,
    _resample_to_len,
)


def _data_tag_from_root(data_root: str) -> str:
    p = Path(data_root)
    try:
        return p.expanduser().resolve().name
    except OSError:
        return p.name or "data"


# viz_compare_eight_panel + dataset_tag=datatmp only: matches flat naming in data_common.pair_specs
_DATATMP_NOISY_TO_REFERENCE = re.compile(r"^(.+)_(low|middle|high)\.txt$", re.IGNORECASE)


def _reference_name_for_eight_panel(noisy_name: str, dataset_tag: str) -> str:
    """Derive reference_signal filename from noisy/result name; datatmp uses its own rule, others use eval_metrics."""
    if dataset_tag.strip().lower() == "datatmp":
        m = _DATATMP_NOISY_TO_REFERENCE.match(noisy_name)
        if m:
            return f"{m.group(1)}.txt"
    return _reference_filename_from_noisy_or_result(noisy_name)


def _setup_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.unicode_minus": False,
        }
    )


RowSingle = tuple[str, np.ndarray]
RowDual = tuple[str, np.ndarray, np.ndarray]
TRAMAGNET_METHOD = "TraMagNet"
OUR_METHOD = TRAMAGNET_METHOD
CNN_METHOD = "cnn"
UNET8_METHOD = "UNet-only ablation"
NN_INFER_METHODS = frozenset({CNN_METHOD, UNET8_METHOD, OUR_METHOD})
FIG2_OUTPUT_STEM_SUFFIX = "-2"
DEFAULT_METHODS_LEGACY = (
    "multi_se_morphological_filter,"
    "gradient_wavelet_morphological_filter,DnCNN baseline,TraMagNet"
)
DEFAULT_METHODS_V2 = (
    "multi_se_morphological_filter,"
    "gradient_wavelet_morphological_filter,DnCNN baseline,UNet-only ablation,TraMagNet"
)
EXCLUDED_METHODS = frozenset(
    {"2unet", "TraMagNet", "adaptive_multi_scale_filter"}
)
METHOD_LABEL_MAP = {
    "gradient_wavelet_morphological_filter": "Gradient wavelet filter",
    "multi_se_morphological_filter": "Multi-morphological filter",
    "DnCNN baseline": "CNN",
    "TraMagNet": "TraMagNet(Our)",
    "UNet-only ablation": "Unet",
}

# Six light-to-deep blue shades (six panels)
_BLUE_PANEL_COLORS = [
    "#6E9AED",
    "#5A8AE3",
    "#5082DE",
    "#3C6ECC",
    "#2F5EB8",
    "#1F458C",
]

_PANEL_LINEWIDTH = 1.75
# data3 dual-channel: offset column-4 trace upward to separate from column 3; see --dual-col2-y-offset
DATA3_DUAL_DEFAULT_COL2_Y_OFFSET = 200.0
DATA3_DUAL_PHYSICAL_COL2_Y_OFFSET = 200.0

# Centered subplot titles (panel labels (a)… drawn separately top-left)
PANEL_HEADINGS_LEGACY = [
    "Raw signal",
    "Noisy input",
    "Multi-morphological filter",
    "Gradient wavelet filter",
    "CNN",
    "TraMagNet(Our)",
]
PANEL_HEADINGS_V2 = [
    "Raw signal",
    "Noisy input",
    "Multi-morphological filter",
    "Gradient wavelet filter",
    "CNN",
    "Unet",
    "TraMagNet(Our)",
]
# Legacy alias
SIX_PANEL_HEADINGS = PANEL_HEADINGS_V2


def _normalize_figure_layout(raw: str) -> str:
    s = str(raw or "legacy").strip().lower()
    if s in ("legacy", "v1", "raw", "rawTraMagNet"):
        return "legacy"
    if s in ("v2", "unet8", "noisyUNet-only ablation"):
        return "v2"
    raise ValueError(f"Unknown figure-layout {raw!r}; choose legacy | v2")


def _default_methods_for_layout(layout: str) -> str:
    return DEFAULT_METHODS_V2 if layout == "v2" else DEFAULT_METHODS_LEGACY


def _panel_headings_for_layout(layout: str) -> list[str]:
    return list(PANEL_HEADINGS_V2 if layout == "v2" else PANEL_HEADINGS_LEGACY)


def _output_stem_suffix_for_layout(layout: str) -> str:
    return FIG2_OUTPUT_STEM_SUFFIX if layout == "v2" else ""


def _expected_denoise_method_count(layout: str) -> int:
    return 5 if layout == "v2" else 4


def _expected_panel_count(layout: str) -> int:
    return 2 + _expected_denoise_method_count(layout)


def _resolve_plot_slice(dataset_tag: str, n: int) -> slice:
    """data1 → [0,500); data3 → [0,700); otherwise full segment."""
    tag = str(dataset_tag).strip().lower()
    n = int(n)
    if tag == "data1":
        return slice(0, min(500, n))
    if tag == "data3":
        return slice(0, min(700, n))
    return slice(0, n)


def _match_noisy_scale_to_reference_np(
    c: np.ndarray,
    n: np.ndarray,
    *,
    eps: float = 1e-6,
    max_sig_ratio: float = 100.0,
) -> np.ndarray:
    c64 = np.asarray(c, dtype=np.float64).ravel()
    n64 = np.asarray(n, dtype=np.float64).ravel()
    m = min(c64.size, n64.size)
    c64, n64 = c64[:m], n64[:m]
    mu_c = float(np.mean(c64))
    sig_c = float(np.std(c64)) + eps
    mu_n = float(np.mean(n64))
    sig_n = float(np.std(n64)) + eps
    ratio = (sig_c / sig_n) if sig_n > 0 else 1.0
    ratio = float(np.clip(ratio, 1.0 / max_sig_ratio, max_sig_ratio))
    return ((n64 - mu_n) * ratio + mu_c).astype(np.float64)


def _zscore_using_reference_stats(y: np.ndarray, reference: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    c = np.asarray(reference, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    m = min(c.size, y.size)
    mu = float(np.mean(c[:m]))
    sig = float(np.std(c[:m])) + eps
    return ((y[:m] - mu) / sig).astype(np.float64)


def _normalize_panel_for_plot(
    reference: np.ndarray,
    y: np.ndarray,
    *,
    match_noisy_scale: bool = False,
    eps: float = 1e-6,
) -> np.ndarray:
    c = np.asarray(reference, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    m = min(c.size, y.size)
    c, y = c[:m], y[:m]
    if match_noisy_scale:
        y = _match_noisy_scale_to_reference_np(c, y, eps=eps)
    return _zscore_using_reference_stats(y, c, eps=eps)


def _normalize_and_slice_rows_for_plot(
    rows: list[RowSingle] | list[RowDual],
    *,
    reference: np.ndarray,
    reference1: np.ndarray | None,
    dual: bool,
    plot_sl: slice,
    plot_noisy_match_scale: bool = False,
) -> list[RowSingle] | list[RowDual]:
    out: list[RowSingle] | list[RowDual] = []
    for i, row in enumerate(rows):
        match_noisy = i == 1 and bool(plot_noisy_match_scale)
        if dual:
            label, y0, y1 = row  # type: ignore[misc]
            c0 = reference
            c1 = reference1 if reference1 is not None else reference
            ny0 = _normalize_panel_for_plot(c0, y0, match_noisy_scale=match_noisy)[plot_sl]
            ny1 = _normalize_panel_for_plot(c1, y1, match_noisy_scale=match_noisy)[plot_sl]
            out.append((label, ny0, ny1))  # type: ignore[arg-type]
        else:
            label, y = row  # type: ignore[misc]
            ny = _normalize_panel_for_plot(reference, y, match_noisy_scale=match_noisy)[plot_sl]
            out.append((label, ny))  # type: ignore[arg-type]
    return out


def _ylabel_for_plot_y_scale(plot_y_scale: str) -> str:
    if str(plot_y_scale).strip().lower() == "physical":
        return "Magnetic field (mGauss)"
    return "Normalized amplitude"


def _prepare_rows_for_plot(
    rows: list[RowSingle] | list[RowDual],
    *,
    reference: np.ndarray,
    reference1: np.ndarray | None,
    dual: bool,
    plot_sl: slice,
    plot_y_scale: str,
    plot_noisy_match_scale: bool,
    value_scale: float,
) -> list[RowSingle] | list[RowDual]:
    if str(plot_y_scale).strip().lower() == "physical":
        scale = float(value_scale)
        out: list[RowSingle] | list[RowDual] = []
        for row in rows:
            if dual:
                label, y0, y1 = row  # type: ignore[misc]
                out.append(
                    (
                        label,
                        np.asarray(y0, dtype=np.float64).ravel()[plot_sl] * scale,
                        np.asarray(y1, dtype=np.float64).ravel()[plot_sl] * scale,
                    )
                )
            else:
                label, y = row  # type: ignore[misc]
                out.append((label, np.asarray(y, dtype=np.float64).ravel()[plot_sl] * scale))
        return out
    return _normalize_and_slice_rows_for_plot(
        rows,
        reference=reference,
        reference1=reference1,
        dual=dual,
        plot_sl=plot_sl,
        plot_noisy_match_scale=plot_noisy_match_scale,
    )


def _panel_curve_arrays(
    row: RowSingle | RowDual,
    *,
    dual_channel: bool,
    dual_col2_y_offset: float,
) -> list[np.ndarray]:
    if dual_channel:
        y0 = np.asarray(row[1], dtype=np.float64).ravel()
        y1 = np.asarray(row[2], dtype=np.float64).ravel()
        mlen = min(y0.size, y1.size)
        return [y0[:mlen], y1[:mlen] + float(dual_col2_y_offset)]
    return [np.asarray(row[1], dtype=np.float64).ravel()]


def _shared_ylim_from_rows(
    rows: list[RowSingle | RowDual],
    *,
    dual_channel: bool,
    dual_col2_y_offset: float,
    ref_panel_indices: tuple[int, ...],
) -> tuple[float, float] | None:
    ys: list[np.ndarray] = []
    for i in ref_panel_indices:
        if i < 0 or i >= len(rows):
            continue
        ys.extend(
            _panel_curve_arrays(
                rows[i],
                dual_channel=dual_channel,
                dual_col2_y_offset=dual_col2_y_offset,
            )
        )
    if not ys:
        return None
    y_lo = min(float(np.min(a)) for a in ys)
    y_hi = max(float(np.max(a)) for a in ys)
    span = max(y_hi - y_lo, 1e-9)
    return y_lo - 0.02 * span, y_hi + 0.22 * span


def _prepend_input_panel_rows(
    layout: str,
    *,
    dual: bool,
    reference0: np.ndarray,
    reference1: np.ndarray,
    noisy0: np.ndarray,
    noisy1: np.ndarray,
    reference_single: np.ndarray,
    noisy_single: np.ndarray,
) -> list[RowSingle] | list[RowDual]:
    del layout
    if dual:
        return [
            ("Raw signal", reference0, reference1),
            ("Noisy input", noisy0, noisy1),
        ]
    return [
        ("Raw signal", reference_single),
        ("Noisy input", noisy_single),
    ]


def _display_method_label(method: str) -> str:
    if method == OUR_METHOD:
        return METHOD_LABEL_MAP.get("TraMagNet", "TraMagNet(Our)")
    return METHOD_LABEL_MAP.get(method, method)


def _single_panel_color(i: int, n: int) -> str:
    if n <= 0:
        return _BLUE_PANEL_COLORS[-1]
    j = min(max(i, 0), len(_BLUE_PANEL_COLORS) - 1)
    return _BLUE_PANEL_COLORS[j]


# Dual-channel column 4: low-saturation warm shades light-to-deep, complementary to column-3 cool blue
_DUAL_COL2_WARM = [
    "#E8B5A8",
    "#D09B8A",
    "#C48E7C",
    "#B8816E",
    "#A06752",
    "#945A46",
]


def _dual_column2_color(i: int, _n: int) -> str:
    """Dual-channel second trace (column 4): warm terracotta/coral gradient, distinct from blue channel 3."""
    hi = len(_DUAL_COL2_WARM) - 1
    j = min(max(i, 0), hi)
    return _DUAL_COL2_WARM[j]


@dataclass(frozen=True)
class _GanPreprocessCfg:
    segment_length: int
    resample_mode: str
    match_noisy_scale: bool
    zscore_using_reference: bool
    eps: float = 1e-6


def _sync_gan_preprocess_from_config(config_dir: Path, args: argparse.Namespace) -> _GanPreprocessCfg:
    seg = int(args.segment_length)
    resample_mode = "resample_linear"
    match_noisy_scale = False
    zscore_using_reference = False
    for cfg_path in (config_dir / "config.txt", config_dir.parent / "config.txt"):
        if not cfg_path.is_file():
            continue
        try:
            d = ast.literal_eval(cfg_path.read_text(encoding="utf-8").strip())
        except (OSError, SyntaxError, TypeError, ValueError, MemoryError):
            continue
        if not isinstance(d, dict):
            continue
        if "segment_length" in d:
            seg = int(d["segment_length"])
        if "resample_mode" in d:
            resample_mode = str(d["resample_mode"])
        if "match_noisy_scale" in d:
            match_noisy_scale = bool(d["match_noisy_scale"])
        if "zscore_using_reference" in d:
            zscore_using_reference = bool(d["zscore_using_reference"])
        print(
            f"[TraMagNet-cv] synced preprocessing from {cfg_path}: segment_length={seg} "
            f"resample_mode={resample_mode} match_noisy_scale={match_noisy_scale} "
            f"zscore_using_reference={zscore_using_reference}",
            flush=True,
        )
        break
    return _GanPreprocessCfg(
        segment_length=seg,
        resample_mode=resample_mode,
        match_noisy_scale=match_noisy_scale,
        zscore_using_reference=zscore_using_reference,
    )


def _resolve_gan5_inference_plan(
    viz_ns: SimpleNamespace,
    *,
    repo: Path,
    data_root: str,
    dataset_tag: str,
) -> InferenceCkptPlan:
    """Resolve TraMagNet K-fold weights; fix ztest5 job root being treated as a single fold."""
    plan = resolve_viz_inference_plan(
        viz_ns,
        repo=repo,
        data_root=data_root,
        data_tag=dataset_tag,
        nn_dir=_TRAMAGNET,
    )
    nf = cv_ensemble_folds_from_args(viz_ns)
    if bool(getattr(viz_ns, "no_cv_ensemble", False)) or nf < 2 or plan.mode == "ensemble":
        return plan
    prefer = str(getattr(viz_ns, "ckpt", "last"))
    bases: list[Path] = []
    if getattr(viz_ns, "runs_dir", None):
        bases.append(Path(str(viz_ns.runs_dir)))
    bases.extend(
        [
            plan.config_dir.parent / "runs",
            plan.config_dir,
            plan.config_dir.parent,
            plan.ckpt_paths[0].parent.parent,
        ]
    )
    seen: set[str] = set()
    for base in bases:
        try:
            key = str(base.resolve())
        except OSError:
            key = str(base)
        if key in seen or not base.is_dir():
            continue
        seen.add(key)
        ens = list_fold_ckpt_paths(base, cv_folds=nf, prefer=prefer)
        if ens is not None and len(ens) >= 2:
            label = f"cv{len(ens)}_ensemble" if len(ens) < nf else f"cv{nf}_ensemble"
            return InferenceCkptPlan(
                mode="ensemble",
                ckpt_paths=tuple(ens),
                config_dir=fold_dir_for_config(base, cv_fold=0),
                label=label,
            )
    return plan


class TraMagNetCvInferer:
    """TraMagNet K-fold checkpoint ensemble inference (preprocess/denorm aligned with viz_TraMagNet_runner)."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        data_root: Path,
        dataset_tag: str,
        denorm: str,
        value_scale: float,
    ) -> None:
        _prepend_nn_to_syspath(_TRAMAGNET)
        import torch
        from data_common.ensemble_infer import load_unet_ensemble, unet_ensemble_forward
        from models.unet import UNET_LATENT_CHANNELS, UNET_LATENT_LENGTH

        self._torch = torch
        self._unet_ensemble_forward = unet_ensemble_forward
        self.denorm = str(denorm)
        self.value_scale = float(value_scale)
        self._infer_z_mode = str(getattr(args, "gan_z_mode", "zero"))

        viz_ns = SimpleNamespace(
            runs_dir=getattr(args, "gan_runs_dir", None),
            job_name=getattr(args, "job_name", None),
            ckpt=str(getattr(args, "ckpt", "last")),
            cv_ensemble_folds=int(getattr(args, "cv_ensemble_folds", 5)),
            no_cv_ensemble=bool(getattr(args, "no_cv_ensemble", False)),
        )
        plan = _resolve_gan5_inference_plan(
            viz_ns,
            repo=_REPO,
            data_root=str(data_root),
            dataset_tag=dataset_tag,
        )
        maybe_sync_split_from_runs_config(args, runs_dir=plan.config_dir, log_prefix="[TraMagNet-cv]")
        self.pre_cfg = _sync_gan_preprocess_from_config(plan.config_dir, args)
        ckpt_paths = list(plan.ckpt_paths)
        if plan.mode != "ensemble":
            print(
                f"[WARN] TraMagNet full K-fold weights not found; using single checkpoint: {ckpt_paths[0]}",
                flush=True,
            )
        else:
            print(
                f"[TraMagNet-cv] K-fold ensemble: {len(ckpt_paths)} checkpoints ({plan.label})",
                flush=True,
            )

        device_name = str(getattr(args, "device", "cuda"))
        self.device = torch.device(
            "cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.members = load_unet_ensemble(ckpt_paths, self.device, gan_generator=True)
        self._UNET_LATENT_CHANNELS = UNET_LATENT_CHANNELS
        self._UNET_LATENT_LENGTH = UNET_LATENT_LENGTH

    def _make_z(self, batch_size: int, dtype: torch.dtype) -> "torch.Tensor":
        if self._infer_z_mode.strip().lower() == "zero":
            return self._torch.zeros(
                batch_size,
                self._UNET_LATENT_CHANNELS,
                self._UNET_LATENT_LENGTH,
                device=self.device,
                dtype=dtype,
            )
        return self._torch.randn(
            batch_size,
            self._UNET_LATENT_CHANNELS,
            self._UNET_LATENT_LENGTH,
            device=self.device,
            dtype=dtype,
        )

    @staticmethod
    def _denorm_plot(reference_phys: np.ndarray, den_z: np.ndarray, *, denorm: str) -> np.ndarray:
        return _maybe_denormalize_like_training(
            reference_raw_1024=reference_phys,
            den_1024=den_z,
            mode=denorm,
        )

    def _denoise_tensor(self, noisy: "torch.Tensor") -> "torch.Tensor":
        z = self._make_z(noisy.size(0), noisy.dtype)
        return self._unet_ensemble_forward(self.members, noisy, z)

    def _preprocess_single_pair(
        self, reference_path: Path, noisy_path: Path, *, value_column: int = 2
    ) -> tuple["torch.Tensor", np.ndarray]:
        _prepend_nn_to_syspath(_TRAMAGNET)
        from data.our_data_dataset import _affine_match_mean_std, _mean_std, _zscore

        c_s, _ = read_one_file_with_meta(reference_path, value_column=value_column)
        n_s, _ = read_one_file_with_meta(noisy_path, value_column=value_column)
        seg = int(self.pre_cfg.segment_length)
        rm = str(self.pre_cfg.resample_mode)
        c_r, _ = pad_or_resample_to_length(c_s.value, seg, mode=rm)
        n_r, _ = pad_or_resample_to_length(n_s.value, seg, mode=rm)
        t = self._torch
        reference = t.tensor(c_r, dtype=t.float32).unsqueeze(0)
        noisy = t.tensor(n_r, dtype=t.float32).unsqueeze(0)
        if self.pre_cfg.match_noisy_scale:
            noisy = _affine_match_mean_std(noisy, reference, eps=float(self.pre_cfg.eps))
        reference_phys = reference.clone()
        if self.pre_cfg.zscore_using_reference:
            mu, sig = _mean_std(reference, eps=float(self.pre_cfg.eps))
            reference = _zscore(reference, mu, sig)
            noisy = _zscore(noisy, mu, sig)
        reference = t.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0)
        noisy = t.nan_to_num(noisy, nan=0.0, posinf=0.0, neginf=0.0)
        return noisy.to(self.device), np.asarray(c_r, dtype=np.float64)

    def infer_single(
        self,
        *,
        reference_path: Path,
        noisy_path: Path,
        fallback: np.ndarray,
        value_column: int = 2,
    ) -> np.ndarray:
        try:
            noisy, reference_phys = self._preprocess_single_pair(
                reference_path, noisy_path, value_column=value_column
            )
            den = self._denoise_tensor(noisy.unsqueeze(0)).squeeze(0).squeeze(0).detach().cpu().numpy()
            den_plot = self._denorm_plot(reference_phys, den, denorm=self.denorm)
            if self.value_scale != 1.0:
                den_plot = den_plot * self.value_scale
            return den_plot
        except Exception as e:
            print(
                f"[WARN] TraMagNet-cv inference failed ({noisy_path.name}); MagGAN panel uses noisy placeholder: {e}",
                flush=True,
            )
            return np.asarray(fallback, dtype=np.float64).copy()

    def infer_dual(
        self,
        *,
        reference_path: Path,
        noisy_path: Path,
        noisy0: np.ndarray,
        noisy1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        _prepend_nn_to_syspath(_TRAMAGNET)
        def _mean_std(x: "torch.Tensor", *, eps: float = 1e-6) -> tuple["torch.Tensor", "torch.Tensor"]:
            mu = x.mean()
            sig = x.std(unbiased=False).clamp_min(eps)
            return mu, sig

        def _affine_match_mean_std(
            noisy: "torch.Tensor", reference: "torch.Tensor", *, eps: float = 1e-6
        ) -> "torch.Tensor":
            mu_c, sig_c = _mean_std(reference, eps=eps)
            mu_n, sig_n = _mean_std(noisy, eps=eps)
            return (noisy - mu_n) * (sig_c / sig_n) + mu_c

        def _zscore(x: "torch.Tensor", mu: "torch.Tensor", sig: "torch.Tensor") -> "torch.Tensor":
            return (x - mu) / sig

        c_a_s, _ = read_one_file_with_meta(reference_path, value_column=2)
        c_b_s, _ = read_one_file_with_meta(reference_path, value_column=3)
        tc, _ = read_two_channel_file(noisy_path)
        seg = int(self.pre_cfg.segment_length)
        rm = str(self.pre_cfg.resample_mode)
        c_a_r, _ = pad_or_resample_to_length(c_a_s.value, seg, mode=rm)
        c_b_r, _ = pad_or_resample_to_length(c_b_s.value, seg, mode=rm)
        a_r, _ = pad_or_resample_to_length(tc.value_a, seg, mode=rm)
        b_r, _ = pad_or_resample_to_length(tc.value_b, seg, mode=rm)

        t = self._torch
        reference_a = t.tensor(c_a_r, dtype=t.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        reference_b = t.tensor(c_b_r, dtype=t.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        noisy_a = t.tensor(a_r, dtype=t.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        noisy_b = t.tensor(b_r, dtype=t.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        if self.pre_cfg.match_noisy_scale:
            noisy_a = _affine_match_mean_std(noisy_a, reference_a)
            noisy_b = _affine_match_mean_std(noisy_b, reference_b)
        if self.pre_cfg.zscore_using_reference:
            mu_a, sig_a = _mean_std(reference_a)
            mu_b, sig_b = _mean_std(reference_b)
            noisy_a_z = _zscore(noisy_a, mu_a, sig_a)
            noisy_b_z = _zscore(noisy_b, mu_b, sig_b)
        else:
            noisy_a_z = noisy_a
            noisy_b_z = noisy_b
        den_a = self._denoise_tensor(noisy_a_z).squeeze().detach().cpu().numpy()
        den_b = self._denoise_tensor(noisy_b_z).squeeze().detach().cpu().numpy()
        d0 = self._denorm_plot(np.asarray(c_a_r, dtype=np.float64), den_a, denorm=self.denorm)
        d1 = self._denorm_plot(np.asarray(c_b_r, dtype=np.float64), den_b, denorm=self.denorm)
        if self.value_scale != 1.0:
            d0 = d0 * self.value_scale
            d1 = d1 * self.value_scale
        return d0, d1


def _sync_preprocess_from_run_config(
    config_dir: Path,
    args: argparse.Namespace,
    *,
    log_prefix: str,
) -> _GanPreprocessCfg:
    seg = int(args.segment_length)
    resample_mode = "resample_linear"
    match_noisy_scale = False
    zscore_using_reference = False
    for cfg_path in (
        config_dir / "run_config.txt",
        config_dir / "config.txt",
        config_dir.parent / "run_config.txt",
        config_dir.parent / "config.txt",
    ):
        if not cfg_path.is_file():
            continue
        try:
            text = cfg_path.read_text(encoding="utf-8")
        except OSError:
            continue
        d: dict | None = None
        if cfg_path.name == "config.txt":
            try:
                parsed = ast.literal_eval(text.strip())
                if isinstance(parsed, dict):
                    d = parsed
            except (SyntaxError, TypeError, ValueError, MemoryError):
                d = None
        if d is None:
            for line in text.splitlines():
                if "OurDataConfig" not in line or "{" not in line:
                    continue
                try:
                    parsed = ast.literal_eval(line[line.index("{") :])
                    if isinstance(parsed, dict):
                        d = parsed
                        break
                except (SyntaxError, TypeError, ValueError, MemoryError):
                    continue
        if not isinstance(d, dict):
            continue
        if "segment_length" in d:
            seg = int(d["segment_length"])
        if "resample_mode" in d:
            resample_mode = str(d["resample_mode"])
        if "match_noisy_scale_to_reference" in d:
            match_noisy_scale = bool(d["match_noisy_scale_to_reference"])
        elif "match_noisy_scale" in d:
            match_noisy_scale = bool(d["match_noisy_scale"])
        if "zscore_using_reference" in d:
            zscore_using_reference = bool(d["zscore_using_reference"])
        print(
            f"{log_prefix} synced preprocessing from {cfg_path}: segment_length={seg} "
            f"resample_mode={resample_mode} match_noisy_scale={match_noisy_scale} "
            f"zscore_using_reference={zscore_using_reference}",
            flush=True,
        )
        break
    return _GanPreprocessCfg(
        segment_length=seg,
        resample_mode=resample_mode,
        match_noisy_scale=match_noisy_scale,
        zscore_using_reference=zscore_using_reference,
    )


def _resolve_cnn3_inference_plan(
    viz_ns: SimpleNamespace,
    *,
    repo: Path,
    data_root: str,
    dataset_tag: str,
) -> InferenceCkptPlan:
    plan = resolve_viz_inference_plan(
        viz_ns,
        repo=repo,
        data_root=data_root,
        data_tag=dataset_tag,
        nn_dir=_CNN,
    )
    nf = cv_ensemble_folds_from_args(viz_ns)
    if bool(getattr(viz_ns, "no_cv_ensemble", False)) or nf < 2 or plan.mode == "ensemble":
        return plan
    prefer = str(getattr(viz_ns, "ckpt", "last"))
    bases: list[Path] = []
    if getattr(viz_ns, "cnn_runs_dir", None):
        bases.append(Path(str(viz_ns.cnn_runs_dir)))
    bases.extend(
        [
            plan.config_dir.parent / "runs",
            plan.config_dir,
            plan.config_dir.parent,
            plan.ckpt_paths[0].parent.parent,
        ]
    )
    seen: set[str] = set()
    for base in bases:
        try:
            key = str(base.resolve())
        except OSError:
            key = str(base)
        if key in seen or not base.is_dir():
            continue
        seen.add(key)
        ens = list_fold_ckpt_paths(base, cv_folds=nf, prefer=prefer)
        if ens is not None and len(ens) >= 2:
            label = f"cv{len(ens)}_ensemble" if len(ens) < nf else f"cv{nf}_ensemble"
            return InferenceCkptPlan(
                mode="ensemble",
                ckpt_paths=tuple(ens),
                config_dir=fold_dir_for_config(base, cv_fold=0),
                label=label,
            )
    return plan


class DnCNNCvInferer:
    """DnCNN baseline K-fold ensemble inference (preprocessing matches training OurDataDataset)."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        data_root: Path,
        dataset_tag: str,
        denorm: str,
        value_scale: float,
    ) -> None:
        _prepend_nn_to_syspath(_CNN)
        import torch
        from data_common.ensemble_infer import load_dncnn_ensemble, tensor_ensemble_forward
        from models.dncnn_1d import DnCNN1D, dncnn_config_from_argparse

        self._torch = torch
        self._tensor_ensemble_forward = tensor_ensemble_forward
        self.denorm = str(denorm)
        self.value_scale = float(value_scale)

        viz_ns = SimpleNamespace(
            runs_dir=getattr(args, "cnn_runs_dir", None),
            job_name=getattr(args, "cnn_job_name", None),
            ckpt=str(getattr(args, "ckpt", "last")),
            cv_ensemble_folds=int(getattr(args, "cv_ensemble_folds", 5)),
            no_cv_ensemble=bool(getattr(args, "no_cv_ensemble", False)),
        )
        plan = _resolve_cnn3_inference_plan(
            viz_ns,
            repo=_REPO,
            data_root=str(data_root),
            dataset_tag=dataset_tag,
        )
        maybe_sync_split_from_runs_config(args, runs_dir=plan.config_dir, log_prefix="[DnCNN-cv]")
        self.pre_cfg = _sync_preprocess_from_run_config(
            plan.config_dir, args, log_prefix="[DnCNN-cv]"
        )
        ckpt_paths = list(plan.ckpt_paths)
        device_name = str(getattr(args, "device", "cuda"))
        self.device = torch.device(
            "cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu"
        )
        if plan.mode != "ensemble":
            print(
                f"[WARN] DnCNN baseline full K-fold weights not found; using single checkpoint: {ckpt_paths[0]}",
                flush=True,
            )
            model = DnCNN1D(dncnn_config_from_argparse(args)).to(self.device)
            payload = torch.load(ckpt_paths[0], map_location=self.device)
            sd = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
            model.load_state_dict(sd, strict=True)
            model.eval()
            self._models = None
            self._model_single = model
        else:
            print(
                f"[DnCNN-cv] K-fold ensemble: {len(ckpt_paths)} checkpoints ({plan.label})",
                flush=True,
            )
            self._models = load_dncnn_ensemble(ckpt_paths, self.device, args)
            self._model_single = None

    def _denoise_tensor(self, noisy: "torch.Tensor") -> "torch.Tensor":
        if self._models is not None:
            return self._tensor_ensemble_forward(self._models, noisy)
        assert self._model_single is not None
        return self._model_single(noisy)

    def infer_single(
        self,
        *,
        reference_path: Path,
        noisy_path: Path,
        fallback: np.ndarray,
        value_column: int = 2,
    ) -> np.ndarray:
        try:
            _prepend_nn_to_syspath(_CNN)
            from data.our_data_dataset import _affine_match_mean_std, _mean_std, _zscore

            c_s, _ = read_one_file_with_meta(reference_path, value_column=value_column)
            n_s, _ = read_one_file_with_meta(noisy_path, value_column=value_column)
            seg = int(self.pre_cfg.segment_length)
            rm = str(self.pre_cfg.resample_mode)
            c_r, _ = pad_or_resample_to_length(c_s.value, seg, mode=rm)
            n_r, _ = pad_or_resample_to_length(n_s.value, seg, mode=rm)
            t = self._torch
            reference = t.tensor(c_r, dtype=t.float32).unsqueeze(0)
            noisy = t.tensor(n_r, dtype=t.float32).unsqueeze(0)
            if self.pre_cfg.match_noisy_scale:
                noisy = _affine_match_mean_std(noisy, reference, eps=float(self.pre_cfg.eps))
            reference_phys = reference.clone()
            if self.pre_cfg.zscore_using_reference:
                mu, sig = _mean_std(reference, eps=float(self.pre_cfg.eps))
                reference = _zscore(reference, mu, sig)
                noisy = _zscore(noisy, mu, sig)
            noisy = t.nan_to_num(noisy, nan=0.0, posinf=0.0, neginf=0.0)
            noisy = noisy.to(self.device)
            den = (
                self._denoise_tensor(noisy.unsqueeze(0))
                .squeeze(0)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
            den_plot = TraMagNetCvInferer._denorm_plot(
                np.asarray(c_r, dtype=np.float64), den, denorm=self.denorm
            )
            if self.value_scale != 1.0:
                den_plot = den_plot * self.value_scale
            return den_plot
        except Exception as e:
            print(
                f"[WARN] DnCNN baseline-cv inference failed ({noisy_path.name}); CNN panel uses noisy placeholder: {e}",
                flush=True,
            )
            return np.asarray(fallback, dtype=np.float64).copy()

    def infer_dual(
        self,
        *,
        reference_path: Path,
        noisy_path: Path,
        noisy0: np.ndarray,
        noisy1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        d0 = self.infer_single(
            reference_path=reference_path,
            noisy_path=noisy_path,
            fallback=noisy0,
            value_column=2,
        )
        d1 = self.infer_single(
            reference_path=reference_path,
            noisy_path=noisy_path,
            fallback=noisy1,
            value_column=3,
        )
        return d0, d1


def _load_ckpt_state(path: Path, device) -> dict:
    import torch

    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"checkpoint missing model key: {path}")
    return payload["model"]


def _resolve_nn_cv_plan(
    viz_ns: SimpleNamespace,
    *,
    repo: Path,
    data_root: str,
    dataset_tag: str,
    nn_dir: Path,
) -> InferenceCkptPlan:
    plan = resolve_viz_inference_plan(
        viz_ns,
        repo=repo,
        data_root=data_root,
        data_tag=dataset_tag,
        nn_dir=nn_dir,
    )
    nf = cv_ensemble_folds_from_args(viz_ns)
    if bool(getattr(viz_ns, "no_cv_ensemble", False)) or nf < 2 or plan.mode == "ensemble":
        return plan
    prefer = str(getattr(viz_ns, "ckpt", "last"))
    bases: list[Path] = []
    if getattr(viz_ns, "runs_dir", None):
        bases.append(Path(str(viz_ns.runs_dir)))
    from data_common.viz_export_workers import checkpoint_run_candidates

    bases.extend(
        checkpoint_run_candidates(
            repo=repo,
            data_tag=dataset_tag,
            nn_dir=nn_dir,
            cv_folds=nf,
            ckpt_prefer=prefer,
        )
    )
    bases.extend(
        [
            plan.config_dir.parent / "runs",
            plan.config_dir,
            plan.config_dir.parent,
            plan.ckpt_paths[0].parent.parent,
        ]
    )
    seen: set[str] = set()
    for base in bases:
        try:
            key = str(base.resolve())
        except OSError:
            key = str(base)
        if key in seen or not base.is_dir():
            continue
        seen.add(key)
        ens = list_fold_ckpt_paths(base, cv_folds=nf, prefer=prefer)
        if ens is not None and len(ens) >= 2:
            label = f"cv{len(ens)}_ensemble" if len(ens) < nf else f"cv{nf}_ensemble"
            return InferenceCkptPlan(
                mode="ensemble",
                ckpt_paths=tuple(ens),
                config_dir=fold_dir_for_config(base, cv_fold=0),
                label=label,
            )
    return plan


class Unet8CvInferer:
    """UNet-only ablation K-fold ``UNetAblation`` ensemble inference (evaluate z=0)."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        data_root: Path,
        dataset_tag: str,
        denorm: str,
        value_scale: float,
    ) -> None:
        _prepend_nn_to_syspath(_8UNET)
        import torch
        from models.unet import UNET_LATENT_CHANNELS, UNET_LATENT_LENGTH, UNetAblation

        self._torch = torch
        self.denorm = str(denorm)
        self.value_scale = float(value_scale)
        self._infer_z_mode = str(getattr(args, "unet8_z_mode", "zero"))
        self._UNET_LATENT_CHANNELS = UNET_LATENT_CHANNELS
        self._UNET_LATENT_LENGTH = UNET_LATENT_LENGTH

        viz_ns = SimpleNamespace(
            runs_dir=getattr(args, "unet8_runs_dir", None),
            job_name=getattr(args, "job_name", None),
            ckpt=str(getattr(args, "ckpt", "last")),
            cv_ensemble_folds=int(getattr(args, "cv_ensemble_folds", 5)),
            no_cv_ensemble=bool(getattr(args, "no_cv_ensemble", False)),
        )
        plan = _resolve_nn_cv_plan(
            viz_ns,
            repo=_REPO,
            data_root=str(data_root),
            dataset_tag=dataset_tag,
            nn_dir=_8UNET,
        )
        self.pre_cfg = _sync_gan_preprocess_from_config(plan.config_dir, args)
        ckpt_paths = list(plan.ckpt_paths)
        device_name = str(getattr(args, "device", "cuda"))
        self.device = torch.device(
            "cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu"
        )
        models: list[torch.nn.Module] = []
        for ckpt in ckpt_paths:
            model = UNetAblation().to(self.device)
            model.load_state_dict(_load_ckpt_state(ckpt, self.device), strict=True)
            model.eval()
            models.append(model)
        self._models = models
        if plan.mode != "ensemble":
            print(
                f"[WARN] UNet-only ablation full K-fold weights not found; using single checkpoint: {ckpt_paths[0]}",
                flush=True,
            )
        else:
            print(
                f"[UNet-only ablation-cv] K-fold ensemble: {len(ckpt_paths)} checkpoints ({plan.label})",
                flush=True,
            )

    def _make_z(self, batch_size: int, dtype) -> "torch.Tensor":
        if self._infer_z_mode.strip().lower() == "zero":
            return self._torch.zeros(
                batch_size,
                self._UNET_LATENT_CHANNELS,
                self._UNET_LATENT_LENGTH,
                device=self.device,
                dtype=dtype,
            )
        from models.unet import sample_latent

        return sample_latent(batch_size, device=self.device, dtype=dtype)

    def _denoise_tensor(self, noisy: "torch.Tensor") -> "torch.Tensor":
        z = self._make_z(noisy.size(0), noisy.dtype)
        acc = None
        n = max(1, len(self._models))
        for model in self._models:
            out = model(noisy, z)
            acc = out if acc is None else acc + out
        assert acc is not None
        return acc / float(n)

    def _preprocess_single_pair(
        self, reference_path: Path, noisy_path: Path, *, value_column: int = 2
    ) -> tuple["torch.Tensor", np.ndarray]:
        _prepend_nn_to_syspath(_8UNET)
        from data.our_data_dataset import _affine_match_mean_std, _mean_std, _zscore

        c_s, _ = read_one_file_with_meta(reference_path, value_column=value_column)
        n_s, _ = read_one_file_with_meta(noisy_path, value_column=value_column)
        seg = int(self.pre_cfg.segment_length)
        rm = str(self.pre_cfg.resample_mode)
        c_r, _ = pad_or_resample_to_length(c_s.value, seg, mode=rm)
        n_r, _ = pad_or_resample_to_length(n_s.value, seg, mode=rm)
        t = self._torch
        reference = t.tensor(c_r, dtype=t.float32).unsqueeze(0)
        noisy = t.tensor(n_r, dtype=t.float32).unsqueeze(0)
        if self.pre_cfg.match_noisy_scale:
            noisy = _affine_match_mean_std(noisy, reference, eps=float(self.pre_cfg.eps))
        if self.pre_cfg.zscore_using_reference:
            mu, sig = _mean_std(reference, eps=float(self.pre_cfg.eps))
            reference = _zscore(reference, mu, sig)
            noisy = _zscore(noisy, mu, sig)
        noisy = t.nan_to_num(noisy, nan=0.0, posinf=0.0, neginf=0.0)
        return noisy.to(self.device), np.asarray(c_r, dtype=np.float64)

    def infer_single(
        self,
        *,
        reference_path: Path,
        noisy_path: Path,
        fallback: np.ndarray,
        value_column: int = 2,
    ) -> np.ndarray:
        try:
            noisy, reference_phys = self._preprocess_single_pair(
                reference_path, noisy_path, value_column=value_column
            )
            den = self._denoise_tensor(noisy.unsqueeze(0)).squeeze(0).squeeze(0).detach().cpu().numpy()
            den_plot = TraMagNetCvInferer._denorm_plot(reference_phys, den, denorm=self.denorm)
            if self.value_scale != 1.0:
                den_plot = den_plot * self.value_scale
            return den_plot
        except Exception as e:
            print(
                f"[WARN] UNet-only ablation-cv inference failed ({noisy_path.name}); Unet panel uses noisy placeholder: {e}",
                flush=True,
            )
            return np.asarray(fallback, dtype=np.float64).copy()

    def infer_dual(
        self,
        *,
        reference_path: Path,
        noisy_path: Path,
        noisy0: np.ndarray,
        noisy1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        d0 = self.infer_single(
            reference_path=reference_path,
            noisy_path=noisy_path,
            fallback=noisy0,
            value_column=2,
        )
        d1 = self.infer_single(
            reference_path=reference_path,
            noisy_path=noisy_path,
            fallback=noisy1,
            value_column=3,
        )
        return d0, d1


class UnetSingleCvInferer:
    """UNet-only ablation K-fold ``UNetSingle`` ensemble inference (single-channel noisy, no z)."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        data_root: Path,
        dataset_tag: str,
        denorm: str,
        value_scale: float,
    ) -> None:
        _prepend_nn_to_syspath(_UNET_SINGLE)
        import torch
        from models.unet import UNetSingle

        self._torch = torch
        self.denorm = str(denorm)
        self.value_scale = float(value_scale)

        viz_ns = SimpleNamespace(
            runs_dir=getattr(args, "unet_single_runs_dir", None),
            job_name=getattr(args, "job_name", None),
            ckpt=str(getattr(args, "ckpt", "last")),
            cv_ensemble_folds=int(getattr(args, "cv_ensemble_folds", 5)),
            no_cv_ensemble=bool(getattr(args, "no_cv_ensemble", False)),
        )
        plan = _resolve_nn_cv_plan(
            viz_ns,
            repo=_REPO,
            data_root=str(data_root),
            dataset_tag=dataset_tag,
            nn_dir=_UNET_SINGLE,
        )
        self.pre_cfg = _sync_gan_preprocess_from_config(plan.config_dir, args)
        ckpt_paths = list(plan.ckpt_paths)
        device_name = str(getattr(args, "device", "cuda"))
        self.device = torch.device(
            "cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu"
        )
        models: list[torch.nn.Module] = []
        for ckpt in ckpt_paths:
            model = UNetSingle().to(self.device)
            model.load_state_dict(_load_ckpt_state(ckpt, self.device), strict=True)
            model.eval()
            models.append(model)
        self._models = models
        if plan.mode != "ensemble":
            print(
                f"[WARN] UNet-only ablation full K-fold weights not found; using single checkpoint: {ckpt_paths[0]}",
                flush=True,
            )
        else:
            print(
                f"[UNet-only ablation-cv] K-fold ensemble: {len(ckpt_paths)} checkpoints ({plan.label})",
                flush=True,
            )

    def _denoise_tensor(self, noisy: "torch.Tensor") -> "torch.Tensor":
        acc = None
        n = max(1, len(self._models))
        for model in self._models:
            out = model(noisy)
            acc = out if acc is None else acc + out
        assert acc is not None
        return acc / float(n)

    def _preprocess_single_pair(
        self, reference_path: Path, noisy_path: Path, *, value_column: int = 2
    ) -> tuple["torch.Tensor", np.ndarray]:
        _prepend_nn_to_syspath(_UNET_SINGLE)
        from data.our_data_dataset import _affine_match_mean_std, _mean_std, _zscore

        c_s, _ = read_one_file_with_meta(reference_path, value_column=value_column)
        n_s, _ = read_one_file_with_meta(noisy_path, value_column=value_column)
        seg = int(self.pre_cfg.segment_length)
        rm = str(self.pre_cfg.resample_mode)
        c_r, _ = pad_or_resample_to_length(c_s.value, seg, mode=rm)
        n_r, _ = pad_or_resample_to_length(n_s.value, seg, mode=rm)
        t = self._torch
        reference = t.tensor(c_r, dtype=t.float32).unsqueeze(0)
        noisy = t.tensor(n_r, dtype=t.float32).unsqueeze(0)
        if self.pre_cfg.match_noisy_scale:
            noisy = _affine_match_mean_std(noisy, reference, eps=float(self.pre_cfg.eps))
        if self.pre_cfg.zscore_using_reference:
            mu, sig = _mean_std(reference, eps=float(self.pre_cfg.eps))
            reference = _zscore(reference, mu, sig)
            noisy = _zscore(noisy, mu, sig)
        noisy = t.nan_to_num(noisy, nan=0.0, posinf=0.0, neginf=0.0)
        return noisy.to(self.device), np.asarray(c_r, dtype=np.float64)

    def infer_single(
        self,
        *,
        reference_path: Path,
        noisy_path: Path,
        fallback: np.ndarray,
        value_column: int = 2,
    ) -> np.ndarray:
        try:
            noisy, reference_phys = self._preprocess_single_pair(
                reference_path, noisy_path, value_column=value_column
            )
            den = self._denoise_tensor(noisy.unsqueeze(0)).squeeze(0).squeeze(0).detach().cpu().numpy()
            den_plot = TraMagNetCvInferer._denorm_plot(reference_phys, den, denorm=self.denorm)
            if self.value_scale != 1.0:
                den_plot = den_plot * self.value_scale
            return den_plot
        except Exception as e:
            print(
                f"[WARN] UNet-only ablation-cv inference failed ({noisy_path.name}); using noisy placeholder: {e}",
                flush=True,
            )
            return np.asarray(fallback, dtype=np.float64).copy()

    def infer_dual(
        self,
        *,
        reference_path: Path,
        noisy_path: Path,
        noisy0: np.ndarray,
        noisy1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        d0 = self.infer_single(
            reference_path=reference_path,
            noisy_path=noisy_path,
            fallback=noisy0,
            value_column=2,
        )
        d1 = self.infer_single(
            reference_path=reference_path,
            noisy_path=noisy_path,
            fallback=noisy1,
            value_column=3,
        )
        return d0, d1


UNET_SINGLE_METHOD = 'unet_single'
__all__ = ['DnCNNCvInferer', 'TraMagNetCvInferer', 'UnetSingleCvInferer', 'CNN_METHOD', 'TRAMAGNET_METHOD', 'OUR_METHOD', 'UNET_SINGLE_METHOD']

# Backward-compatible aliases
Cnn3CvInferer = DnCNNCvInferer
Gan5CvInferer = TraMagNetCvInferer
Unet9CvInferer = UnetSingleCvInferer
