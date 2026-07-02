#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Morphological filter baselines (gradient wavelet / multi-SE).

**Preprocessing protocol (important):** Unlike deep learning models in this repo
(default: noisy-only normalization, no clean leakage), traditional filters here
use the project's **paired reference/noisy offline evaluation** setting. Clean signal
statistics may be used when aligning or plotting paired segments. Do not describe
these results as blind deployment denoising; document the protocol in paper methods.

**Reading and pairing**: Uses ``data_common.pair_specs.list_pair_specs`` and
``data_common.txt_io`` (same as all training code).

**Visualization**: Uses ``data_common.viz_export.save_triplet_figure`` /
``save_dual_column_triplet_figure`` (same module as each ``data*/viz_export.py``).

Output dirs remain ``output/<method>/<data1|data2|data3>/image`` and ``…/result``.

data3 ``+subway`` four-column files: dual-column subplot JPG + merged single ``{stem}.txt`` (cols 3/4 are two denoised channels).

Usage (from repo root)::

    python run_py_denoise_methods.py
    python run_py_denoise_methods.py --max-pairs 5 --data-roots data3
    python run_py_denoise_methods.py --methods gradient_wavelet_morphological_filter,multi_se_morphological_filter --workers 8
"""

from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parent
MAIN = ROOT.parent
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_common.pair_specs import PairSpec, list_pair_specs
from data_common.resolve_dataset_root import resolve_dataset_root
from data_common.txt_io import (
    read_amplitude_np,
    read_two_channel_file,
    subway_noisy_has_four_value_columns,
)
from data_common.viz_export import (
    save_dual_column_triplet_figure,
    save_triplet_figure,
)

@dataclass(frozen=True)
class DenoiseTask:
    method_name: str
    tag: str
    out_img: str
    out_res: str
    reference_path: str
    noisy_path: str
    value_column: int
    dual: bool


def _safe_filename(s: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", s)


def _match_noisy_scale_to_reference(
    c: np.ndarray, n: np.ndarray, *, eps: float = 1e-6, max_sig_ratio: float = 100.0
) -> np.ndarray:
    c64 = np.asarray(c, dtype=np.float64)
    n64 = np.asarray(n, dtype=np.float64)
    mu_c = np.mean(c64)
    sig_c = float(np.std(c64)) + eps
    mu_n = np.mean(n64)
    sig_n = float(np.std(n64)) + eps
    ratio = (sig_c / sig_n) if sig_n > 0 else 1.0
    ratio = float(np.clip(ratio, 1.0 / max_sig_ratio, max_sig_ratio))
    return (n64 - mu_n) * ratio + mu_c


def _normalize_pair(c: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same as 1/data/our_data_folder_dataset: match_scale + z-score with clean μ/σ."""
    n = _match_noisy_scale_to_reference(c, n)
    mu = float(np.mean(c))
    sig = float(np.std(c)) + 1e-6
    c = (c - mu) / sig
    n = (n - mu) / sig
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    n = np.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0)
    return c.astype(np.float64), n.astype(np.float64)


def _trim_filter_output(d: np.ndarray, m: int) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    if d.size > m:
        return d[:m]
    if d.size < m:
        return np.pad(d, (0, m - d.size), mode="edge")
    return d


def _filter_fn_by_name(method_name: str) -> Callable[[np.ndarray], np.ndarray]:
    if method_name == "gradient_wavelet_morphological_filter":
        from py_denoise.methods.gradient_wavelet_morphological_filter import (
            gradient_wavelet_morphological_filter,
        )

        return gradient_wavelet_morphological_filter
    if method_name == "multi_se_morphological_filter":
        from py_denoise.methods.multi_se_morphological_filter import multi_se_morphological_filter

        return multi_se_morphological_filter
    raise ValueError(f"Unknown method: {method_name}")


def _method_fns() -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    return {
        "gradient_wavelet_morphological_filter": _filter_fn_by_name(
            "gradient_wavelet_morphological_filter"
        ),
        "multi_se_morphological_filter": _filter_fn_by_name("multi_se_morphological_filter"),
    }


def _build_task(
    *,
    method_name: str,
    tag: str,
    out_img: Path,
    out_res: Path,
    sp: PairSpec,
) -> DenoiseTask | None:
    noisy_path = sp.noisy_path
    use_dual = subway_noisy_has_four_value_columns(noisy_path)
    if use_dual:
        tc, tst = read_two_channel_file(noisy_path)
        if tst.get("kept_lines", 0) < 2:
            use_dual = False
    return DenoiseTask(
        method_name=str(method_name),
        tag=str(tag),
        out_img=str(out_img),
        out_res=str(out_res),
        reference_path=str(sp.reference_path),
        noisy_path=str(noisy_path),
        value_column=int(sp.value_column),
        dual=bool(use_dual),
    )


def _process_task(task: DenoiseTask) -> tuple[bool, str]:
    """One sample: read → filter → write result / image. Called by multiprocessing workers."""
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    fn = _filter_fn_by_name(task.method_name)
    reference_path = Path(task.reference_path)
    noisy_path = Path(task.noisy_path)
    out_img = Path(task.out_img)
    out_res = Path(task.out_res)
    out_img.mkdir(parents=True, exist_ok=True)
    out_res.mkdir(parents=True, exist_ok=True)

    if task.dual:
        tc, tst = read_two_channel_file(noisy_path)
        if tst.get("kept_lines", 0) < 2:
            return False, f"Insufficient dual-column valid rows: {noisy_path.name}"
        c_a = read_amplitude_np(reference_path, value_column=2)
        c_b = read_amplitude_np(reference_path, value_column=3)
        n_a = np.asarray(tc.value_a, dtype=np.float64)
        n_b = np.asarray(tc.value_b, dtype=np.float64)
        m = min(int(c_a.size), int(c_b.size), int(n_a.size), int(n_b.size))
        if m < 2:
            return False, f"Skipped (too short): {noisy_path.name}"
        c_a, c_b = c_a[:m], c_b[:m]
        n_a, n_b = n_a[:m], n_b[:m]
        c_na, n_na = _normalize_pair(c_a, n_a)
        c_nb, n_nb = _normalize_pair(c_b, n_b)
        d_a = _trim_filter_output(fn(n_na), m)
        d_b = _trim_filter_output(fn(n_nb), m)
        stem = _safe_filename(noisy_path.stem)
        title = f"{task.method_name} | {task.tag} | {noisy_path.name} | dual-column"
        save_dual_column_triplet_figure(
            out_img / f"{stem}.jpg",
            c_na,
            n_na,
            d_a,
            n_nb,
            d_b,
            title=title,
            reference_b=c_nb,
        )
        merged_path = out_res / f"{stem}.txt"
        with merged_path.open("w", encoding="utf-8", newline="\n") as f:
            for ii, (va, vb) in enumerate(zip(d_a.tolist(), d_b.tolist())):
                f.write(f"{ii}\t{ii}\t{va:.8f}\t{vb:.8f}\n")
        return True, noisy_path.name

    c = read_amplitude_np(reference_path, value_column=2)
    n = read_amplitude_np(noisy_path, value_column=int(task.value_column))
    m = min(int(c.size), int(n.size))
    if m < 2:
        return False, f"Skipped (too short): {noisy_path.name}"
    c0, n0 = c[:m], n[:m]
    c_n, n_n = _normalize_pair(c0, n0)
    d_n = _trim_filter_output(fn(n_n), m)
    stem = _safe_filename(noisy_path.stem)
    title = f"{task.method_name} | {task.tag} | {noisy_path.name}"
    save_triplet_figure(out_img / f"{stem}.jpg", c_n, n_n, d_n, title=title)
    np.savetxt(out_res / f"{stem}.txt", d_n, fmt="%.8f")
    return True, noisy_path.name


def _run_tasks_serial(tasks: list[DenoiseTask], *, log_prefix: str) -> tuple[int, int]:
    ok = 0
    total = len(tasks)
    for i, task in enumerate(tasks, start=1):
        success, _msg = _process_task(task)
        if success:
            ok += 1
        if i % 50 == 0 or i == total:
            print(f"{log_prefix}: {i}/{total} …", flush=True)
    return ok, total - ok


def _run_tasks_parallel(tasks: list[DenoiseTask], *, workers: int, log_prefix: str) -> tuple[int, int]:
    ok = 0
    fail = 0
    total = len(tasks)
    done = 0
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        futures = {pool.submit(_process_task, t): t for t in tasks}
        for fut in as_completed(futures):
            done += 1
            try:
                success, msg = fut.result()
            except Exception as e:
                fail += 1
                task = futures[fut]
                print(f"[ERR] {task.noisy_path}: {e}", flush=True)
                continue
            if success:
                ok += 1
            else:
                fail += 1
                if msg:
                    print(f"[warn] {msg}", flush=True)
            if done % 50 == 0 or done == total:
                print(f"{log_prefix}: {done}/{total} …", flush=True)
    return ok, fail


def main() -> None:
    p = argparse.ArgumentParser(
        description="py_denoise methods + data_common pairing/IO + viz_export plots"
    )
    p.add_argument(
        "--data-roots",
        nargs="+",
        default=["data1", "data3", "data4"],
        help="Dataset dir names (data1/data3/data4) or paths under ../datasets",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default="output",
        help="Output root directory",
    )
    p.add_argument(
        "--reference-subdir",
        "--reference-subdir",
        type=str,
        default="reference_signal",
        dest="reference_subdir",
        help="Same as list_pair_specs",
    )
    p.add_argument(
        "--noisy-subdir",
        type=str,
        default="noise_signal",
        help="Same as list_pair_specs",
    )
    p.add_argument(
        "--band",
        type=str,
        default="all",
        choices=("low", "middle", "high", "all"),
        help="Pairing band; default all matches prior multi-band noise_signal traversal",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="gradient_wavelet_morphological_filter,multi_se_morphological_filter",
        help="Comma-separated method names; default runs both morphological filters",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel worker count; 1 = serial, default 8",
    )
    p.add_argument("--max-pairs", type=int, default=0, help="Max PairSpecs per dataset; 0 = no limit")
    args = p.parse_args()

    all_methods = _method_fns()
    wanted = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    unknown = [m for m in wanted if m not in all_methods]
    if unknown:
        raise SystemExit(f"[ERR] Unknown method(s): {unknown}; options: {sorted(all_methods)}")
    methods = {k: all_methods[k] for k in wanted}
    out_base = Path(args.output_root)
    workers = max(1, int(args.workers))

    for method_name in methods:
        n_done = 0
        written_tags: list[str] = []

        for tag in args.data_roots:
            data_root = Path(resolve_dataset_root(tag, repo=MAIN)).resolve()
            if not data_root.is_dir():
                print(f"[skip] No data directory: {tag} -> {data_root}", flush=True)
                continue
            out_img = out_base / method_name / tag / "image"
            out_res = out_base / method_name / tag / "result"
            out_img.mkdir(parents=True, exist_ok=True)
            out_res.mkdir(parents=True, exist_ok=True)

            specs = list_pair_specs(
                data_root,
                reference_subdir=args.reference_subdir,
                noisy_subdir=args.noisy_subdir,
                band=args.band,
                subway_dual_channels=False,
                strict_all_bands=False,
            )
            if not specs:
                print(f"[{method_name}] {tag}: list_pair_specs returned empty", flush=True)
                continue
            limit = args.max_pairs if args.max_pairs > 0 else len(specs)
            tasks: list[DenoiseTask] = []
            for sp in specs[:limit]:
                task = _build_task(
                    method_name=method_name,
                    tag=tag,
                    out_img=out_img,
                    out_res=out_res,
                    sp=sp,
                )
                if task is not None:
                    tasks.append(task)

            if not tasks:
                continue

            log_prefix = f"[{method_name}] {tag} workers={workers}"
            if workers <= 1:
                ok, _fail = _run_tasks_serial(tasks, log_prefix=log_prefix)
            else:
                ok, _fail = _run_tasks_parallel(tasks, workers=workers, log_prefix=log_prefix)

            n_done += ok
            if ok:
                written_tags.append(tag)
                print(
                    f"[{method_name}] {tag}: {ok} samples -> {out_img.resolve()} | {out_res.resolve()}",
                    flush=True,
                )

        tags_s = ", ".join(written_tags) if written_tags else "(none)"
        print(f"[{method_name}] Total {n_done} samples; wrote dataset subdirs: {tags_s}", flush=True)


if __name__ == "__main__":
    main()
