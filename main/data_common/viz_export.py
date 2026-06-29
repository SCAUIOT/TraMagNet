# -*- coding: utf-8 -*-
"""Save triple-curve figures; each data*/viz_export.py may forward ``main_cli`` here."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Optional

import numpy as np


def _configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "DengXian",
                "Arial Unicode MS",
                "Noto Sans CJK SC",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
        }
    )


def save_triplet_figure(
    out_path: Path,
    reference: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    *,
    title: str = "",
    figsize: tuple[float, float] = (10.0, 4.0),
    dpi: int = 120,
    xlabel: str = "Sample index",
    ylabel: str = "Amplitude",
    noisy_label: str = "Noisy",
    denoised_label: str = "Denoised",
    reference_label: str = "Reference",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_matplotlib()
    m = min(int(reference.size), int(noisy.size), int(denoised.size))
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)[:m]
    noisy = np.asarray(noisy, dtype=np.float64).reshape(-1)[:m]
    denoised = np.asarray(denoised, dtype=np.float64).reshape(-1)[:m]
    t = np.arange(m)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(t, noisy, linewidth=0.9, color="tab:orange", label=noisy_label)
    ax.plot(t, denoised, linewidth=0.9, color="tab:green", label=denoised_label)
    ax.plot(t, reference, linewidth=0.9, color="tab:blue", label=reference_label)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    gc.collect()


def save_gan_compare_figure(
    out_path: Path,
    reference: np.ndarray,
    noisy: np.ndarray,
    den_orig: np.ndarray,
    den_quant: np.ndarray,
    den_struct: np.ndarray,
    den_finetune: np.ndarray,
    *,
    title: str = "",
    figsize: tuple[float, float] = (12.0, 8.0),
    dpi: int = 160,
    xlabel: str = "Sample index",
    ylabel: str = "Amplitude (preprocessed)",
) -> None:
    """Six-panel comparison (3 rows × 2 cols): reference|noisy / original|quantized / compressed|finetuned."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_matplotlib()
    arrays = [
        np.asarray(reference, dtype=np.float64).reshape(-1),
        np.asarray(noisy, dtype=np.float64).reshape(-1),
        np.asarray(den_orig, dtype=np.float64).reshape(-1),
        np.asarray(den_quant, dtype=np.float64).reshape(-1),
        np.asarray(den_struct, dtype=np.float64).reshape(-1),
        np.asarray(den_finetune, dtype=np.float64).reshape(-1),
    ]
    m = min(int(a.size) for a in arrays)
    arrays = [a[:m] for a in arrays]
    t = np.arange(m)
    labels = ("Reference", "Noisy", "Original denoised", "Quantized denoised", "Compressed denoised", "Finetuned denoised")
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown")
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True)
    axes_flat = axes.ravel()
    for ax, y, lab, col in zip(axes_flat, arrays, labels, colors):
        ax.plot(t, y, linewidth=0.9, color=col, label=lab)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        ax.set_title(lab, fontsize=10)
    for ax in axes[2, :]:
        ax.set_xlabel(xlabel)
    if title:
        fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    gc.collect()


def save_dual_column_triplet_figure(
    out_path: Path,
    reference: np.ndarray,
    noisy_a: np.ndarray,
    denoised_a: np.ndarray,
    noisy_b: np.ndarray,
    denoised_b: np.ndarray,
    *,
    title: str = "",
    reference_b: np.ndarray | None = None,
) -> None:
    """data3-style dual amplitude columns: two stacked subplots; same reference by default; ``reference_b`` sets reference ref for column 4."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_matplotlib()
    reference2 = reference if reference_b is None else reference_b
    m = min(
        int(reference.size),
        int(np.asarray(reference2).size),
        int(noisy_a.size),
        int(denoised_a.size),
        int(noisy_b.size),
        int(denoised_b.size),
    )
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)[:m]
    reference2 = np.asarray(reference2, dtype=np.float64).reshape(-1)[:m]
    noisy_a = np.asarray(noisy_a, dtype=np.float64).reshape(-1)[:m]
    denoised_a = np.asarray(denoised_a, dtype=np.float64).reshape(-1)[:m]
    noisy_b = np.asarray(noisy_b, dtype=np.float64).reshape(-1)[:m]
    denoised_b = np.asarray(denoised_b, dtype=np.float64).reshape(-1)[:m]
    t = np.arange(m)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, n_raw, d_raw, cref, subtitle in (
        (axes[0], noisy_a, denoised_a, reference, "Noisy / Denoised / Reference — column 3"),
        (axes[1], noisy_b, denoised_b, reference2, "Noisy / Denoised / Reference — column 4"),
    ):
        ax.plot(t, n_raw, linewidth=0.9, color="tab:orange", label="Noisy")
        ax.plot(t, d_raw, linewidth=0.9, color="tab:green", label="Denoised")
        ax.plot(t, cref, linewidth=0.9, color="tab:blue", label="Reference")
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("Amplitude")
        ax.legend(loc="best")
        ax.set_title(subtitle)
    axes[1].set_xlabel("Sample index")
    if title:
        fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    gc.collect()


def main_cli(argv: Optional[list[str]] = None, *, default_data_root: Optional[Path] = None) -> None:
    p = argparse.ArgumentParser(description="Read first pair under data root and plot reference/noisy/(denoised) triplet")
    p.add_argument(
        "--data-root",
        type=str,
        default="" if default_data_root is None else str(default_data_root),
        help="Data root (with reference_signal / noise_signal); default is current dataN dir when omitted",
    )
    p.add_argument("--reference-subdir", type=str, default="reference_signal")
    p.add_argument("--noisy-subdir", type=str, default="noise_signal")
    p.add_argument("--band", type=str, default="low")
    p.add_argument("--pair-index", type=int, default=0, dest="pair_index")
    p.add_argument("--denoised-txt", type=str, default="", help="Denoised result txt; if omitted, copy noisy as placeholder")
    p.add_argument("--out", type=str, default="triplet_preview.jpg")
    p.add_argument("--subway-dual-channels", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--strict-all-bands",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When band=all, keep only samples with low+middle+high all present (DnCNN default)",
    )
    args = p.parse_args(argv)

    root = Path(args.data_root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Data root does not exist: {root}")

    from .pair_specs import list_pair_specs
    from .txt_io import read_amplitude_np

    specs = list_pair_specs(
        root,
        reference_subdir=args.reference_subdir,
        noisy_subdir=args.noisy_subdir,
        band=args.band,
        subway_dual_channels=bool(args.subway_dual_channels),
        strict_all_bands=bool(args.strict_all_bands),
    )
    if not specs:
        raise SystemExit(f"No paired samples found: {root}")
    if args.pair_index < 0 or args.pair_index >= len(specs):
        raise SystemExit(f"pair-index out of range: {args.pair_index} ({len(specs)} pairs total)")
    sp = specs[args.pair_index]
    c = read_amplitude_np(sp.reference_path, value_column=2)
    n = read_amplitude_np(sp.noisy_path, value_column=sp.value_column)
    if args.denoised_txt.strip():
        d = np.loadtxt(args.denoised_txt, dtype=np.float64)
    else:
        d = n.copy()
    title = f"{sp.reference_path.name} / {sp.noisy_path.name} vcol={sp.value_column}"
    save_triplet_figure(Path(args.out), c, n, d, title=title)
    print(f"Wrote: {Path(args.out).resolve()}", flush=True)


if __name__ == "__main__":
    main_cli()
