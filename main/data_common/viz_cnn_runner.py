from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import torch

_REPO = Path(__file__).resolve().parents[1]
_CNN = _REPO / "cnn"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_CNN) not in sys.path:
    sys.path.insert(0, str(_CNN))

from data_common.cv_ensemble import add_cv_ensemble_arguments
from data_common.ensemble_infer import load_dncnn_ensemble, tensor_ensemble_forward
from data_common.viz_ckpt_resolve import resolve_viz_inference_plan
from data_common.viz_pooled import add_viz_pooled_arguments
from data_common.viz_split import (
    add_viz_split_arguments,
    chosen_sample_ids_from_specs,
    describe_split_for_log,
    maybe_sync_split_from_runs_config,
    our_data_dataset_split_kwargs,
)
from data_common.viz_export_workers import (
    checkpoint_run_candidates,
    default_export_worker_count,
    empty_viz_output_dirs,
    split_into_n_chunks,
)

_VIZ3_MP: dict = {}


def _data_tag_from_root(data_root: str) -> str:
    from data_common.dataset_paths import dataset_tag_for_path, resolve_dataset_root

    p = Path(resolve_dataset_root(data_root, repo=_REPO))
    return dataset_tag_for_path(p)


def _resolve_data_root(path_str: str) -> str:
    from data_common.resolve_dataset_root import resolve_dataset_root

    return resolve_dataset_root(path_str, repo=_REPO)


def _pick_ckpt(runs_dir: Path, ckpt_arg: str) -> Path:
    mode = ckpt_arg.strip().lower()
    if mode not in ("last", "best"):
        p = Path(ckpt_arg)
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p
    candidates = []
    names = ("best.pt", "last.pt") if mode == "best" else ("last.pt", "best.pt")
    for name in names:
        p = runs_dir / name
        if p.is_file():
            candidates.append(p)
    if not candidates and runs_dir.is_dir():
        pts = [p for p in runs_dir.glob("*.pt") if p.is_file()]
        pts.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        candidates = pts
    if not candidates:
        raise FileNotFoundError(f"No usable .pt in directory: {runs_dir}")
    return candidates[0]


def _mean_std(x: torch.Tensor, *, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    mu = x.mean()
    sig = x.std(unbiased=False).clamp_min(eps)
    return mu, sig


def _affine_match_mean_std(noisy: torch.Tensor, reference: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    mu_c, sig_c = _mean_std(reference, eps=eps)
    mu_n, sig_n = _mean_std(noisy, eps=eps)
    return (noisy - mu_n) * (sig_c / sig_n) + mu_c


def _zscore(x: torch.Tensor, mu: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
    return (x - mu) / sig


def _cnn_mp_init(pack: dict) -> None:
    global _VIZ3_MP
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    sys.path.insert(0, pack["repo"])
    sys.path.insert(0, pack["threecnn"])
    import torch as _t

    _t.set_num_threads(1)

    from data.our_data_dataset import OurDataConfig, OurDataDataset
    from models.dncnn_1d import DnCNN1D, dncnn_config_from_argparse

    device = _t.device("cpu")
    cfg_common = dict(
        root=pack["data_root"],
        reference_subdir=pack["reference_subdir"],
        noisy_subdir=pack["noisy_subdir"],
        band=pack["band"],
        segment_length=int(pack["segment_length"]),
        train_ratio=float(pack["train_ratio"]),
        seed=int(pack["seed"]),
        shuffle_split=bool(pack["shuffle_split"]),
        split_round=True,
        resample_mode=pack["resample_mode"],
        strict_all_bands=not bool(pack.get("manifest_dual", False)),
        match_noisy_scale_to_reference=bool(pack["match_noisy_scale"]),
        zscore_using_reference=bool(pack["zscore_using_reference"]),
    )
    if pack.get("manifest_dual", False):
        export_dss = [
            OurDataDataset(
                OurDataConfig(**cfg_common, train=True, holdout_eval=False, cv_folds=0, cv_fold=0)
            ),
            OurDataDataset(
                OurDataConfig(**cfg_common, train=False, holdout_eval=True, cv_folds=0, cv_fold=0)
            ),
        ]
    else:
        export_dss = [
            OurDataDataset(
                OurDataConfig(
                    **cfg_common,
                    train=bool(pack["train"]),
                    holdout_eval=bool(pack["holdout_eval"]),
                    cv_folds=int(pack["cv_folds"]),
                    cv_fold=int(pack["cv_fold"]),
                )
            )
        ]
    ns = SimpleNamespace(**pack["model_args"])
    mp_state = {
        "device": device,
        "dss": export_dss,
        "manifest_dual": bool(pack.get("manifest_dual", False)),
        "img_dir": Path(pack["img_dir"]),
        "res_dir": Path(pack["res_dir"]),
        "band": str(pack["band"]),
    }
    if pack.get("ensemble"):
        from data_common.ensemble_infer import load_dncnn_ensemble

        mp_state["models"] = load_dncnn_ensemble(pack["ckpt_paths"], device, ns)
    else:
        model = DnCNN1D(dncnn_config_from_argparse(ns)).to(device)
        payload = _t.load(pack["ckpt_path"], map_location=device)
        sd = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        model.load_state_dict(sd, strict=True)
        model.eval()
        mp_state["model"] = model
    _VIZ3_MP = mp_state


@torch.no_grad()
def _cnn_mp_run_chunk(chunk_indices: list[int], infer_bs: int) -> int:
    from data_common.viz_export import save_triplet_figure

    g = _VIZ3_MP
    device = g["device"]
    models = g.get("models")
    model = g.get("model")
    dss = g["dss"]
    manifest_dual = bool(g.get("manifest_dual", False))
    img_dir = g["img_dir"]
    res_dir = g["res_dir"]
    band = g["band"]
    infer_bs = max(1, int(infer_bs))
    exported = 0
    keys: list[str] = []
    c_list: list[torch.Tensor] = []
    n_list: list[torch.Tensor] = []

    def flush() -> int:
        nonlocal keys, c_list, n_list
        if not keys:
            return 0
        c_stack = torch.cat(c_list, dim=0).to(device)
        n_stack = torch.cat(n_list, dim=0).to(device)
        if models is not None:
            d_stack = tensor_ensemble_forward(models, n_stack)
        else:
            assert model is not None
            d_stack = model(n_stack)
        for bi, k in enumerate(keys):
            c = c_stack[bi : bi + 1]
            n = n_stack[bi : bi + 1]
            d = d_stack[bi : bi + 1]
            save_triplet_figure(
                img_dir / f"{k}.png",
                c.squeeze().detach().cpu().numpy(),
                n.squeeze().detach().cpu().numpy(),
                d.squeeze().detach().cpu().numpy(),
                title=f"{k} (band={band}) — reference vs noisy vs denoised",
                figsize=(12.0, 4.0),
                dpi=160,
                xlabel="Sample",
                ylabel="Amplitude (preprocessed)",
                noisy_label="noisy",
                denoised_label="denoised",
                reference_label="reference",
            )
            y = d.squeeze().detach().cpu().float().numpy()
            with (res_dir / f"{k}.txt").open("w", encoding="utf-8", newline="\n") as f:
                for ii, v in enumerate(y.tolist()):
                    f.write(f"{ii}\t{ii}\t{v}\n")
        n_out = len(keys)
        keys, c_list, n_list = [], [], []
        return n_out

    for item in chunk_indices:
        ds_i, j = (int(item[0]), int(item[1])) if isinstance(item, (list, tuple)) else (0, int(item))
        it = dss[ds_i][j]
        keys.append(str(it.get("key", f"idx_{j}")))
        c_list.append(it["reference"].unsqueeze(0))
        n_list.append(it["noisy"].unsqueeze(0))
        if len(keys) >= infer_bs:
            exported += flush()
    exported += flush()
    return exported


@torch.no_grad()
def _run_one_split(args: argparse.Namespace, *, split: str, clear_outputs: bool = True) -> None:
    from data.our_data_dataset import OurDataConfig, OurDataDataset
    from models.dncnn_1d import DnCNN1D, dncnn_config_from_argparse
    from data_common.pair_specs import list_pair_specs
    from data_common.viz_pooled import (
        add_viz_pooled_arguments,
        dataset_export_indices,
        merged_dataset_export_indices,
        resolve_viz_pool_plan,
        spec_in_allowed_keys,
        split_kwargs_for_viz_target,
    )
    from data_common.txt_io import (
        pad_or_resample_to_length,
        read_one_file_with_meta,
        read_two_channel_file,
        subway_noisy_has_four_value_columns,
    )
    from data_common.viz_export import save_dual_column_triplet_figure, save_triplet_figure

    pool_plan = resolve_viz_pool_plan(args, _REPO, split=split)
    tgt = pool_plan.targets[0]
    data_root = tgt.data_root
    tag = tgt.dataset_tag
    allowed_keys = tgt.allowed_keys
    plan = resolve_viz_inference_plan(
        args, repo=_REPO, data_root=data_root, data_tag=tag, nn_dir=_CNN
    )
    ckpt_paths = list(plan.ckpt_paths)
    is_ensemble = plan.mode == "ensemble"
    ckpt_path = ckpt_paths[0]

    maybe_sync_split_from_runs_config(args, runs_dir=plan.config_dir, log_prefix="[cnn-viz]")
    split_kw = split_kwargs_for_viz_target(
        split,
        allowed_keys=allowed_keys,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
        shuffle_split=bool(args.shuffle_split),
        cv_folds=int(getattr(args, "cv_folds", 0)),
        cv_fold=int(getattr(args, "cv_fold", 0)),
    )
    if allowed_keys is None:
        print(
            f"[cnn-viz] split={split} -> {describe_split_for_log(split, cv_folds=int(args.cv_folds), cv_fold=int(args.cv_fold))} "
            f"(no ztest5 manifest; using single-dataset 8:2 split)",
            flush=True,
        )

    _out_base = Path("output") / _CNN.name / tag
    img_dir = Path(args.output_dir) if args.output_dir else (_out_base / "image")
    res_dir = Path(args.result_dir) if args.result_dir else (_out_base / "result")
    if clear_outputs:
        empty_viz_output_dirs(img_dir, res_dir)
    else:
        img_dir.mkdir(parents=True, exist_ok=True)
        res_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    root_p = Path(data_root)
    noisy_dir = root_p / args.noisy_subdir
    has_subway_dual = False
    if noisy_dir.is_dir():
        for p in noisy_dir.glob("*+subway.txt"):
            if subway_noisy_has_four_value_columns(p):
                has_subway_dual = True
                break

    cfg_common = dict(
        root=data_root,
        reference_subdir=args.reference_subdir,
        noisy_subdir=args.noisy_subdir,
        band=args.band,  # type: ignore[arg-type]
        segment_length=int(args.segment_length),
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
        shuffle_split=bool(args.shuffle_split),
        split_round=True,
        resample_mode=args.resample_mode,  # type: ignore[arg-type]
        strict_all_bands=(allowed_keys is None),
        match_noisy_scale_to_reference=bool(args.match_noisy_scale),
        zscore_using_reference=bool(args.zscore_using_reference),
    )
    if allowed_keys is not None:
        # Same as loss_eval.build_datasets_for_eval: train pool + holdout pool, then filter by manifest keys
        export_dss = [
            OurDataDataset(
                OurDataConfig(**cfg_common, train=True, holdout_eval=False, cv_folds=0, cv_fold=0)
            ),
            OurDataDataset(
                OurDataConfig(**cfg_common, train=False, holdout_eval=True, cv_folds=0, cv_fold=0)
            ),
        ]
    else:
        export_dss = [OurDataDataset(OurDataConfig(**cfg_common, **split_kw))]
    manifest_dual = allowed_keys is not None

    infer_bs = max(1, int(getattr(args, "infer_batch_size", 1)))
    nw = max(1, int(getattr(args, "export_workers", 1)))

    import numpy as np

    models_ens = None
    model_single = None
    if is_ensemble:
        models_ens = load_dncnn_ensemble(ckpt_paths, device, args)
        print(f"[{split}] cv ensemble ({len(ckpt_paths)} folds)", flush=True)
    else:
        model_single = DnCNN1D(dncnn_config_from_argparse(args)).to(device)
        payload = torch.load(ckpt_path, map_location=device)
        sd = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        model_single.load_state_dict(sd, strict=True)
        model_single.eval()
        print(f"[{split}] ckpt={ckpt_path}", flush=True)

    def _denoise_batch(n: torch.Tensor) -> torch.Tensor:
        if models_ens is not None:
            return tensor_ensemble_forward(models_ens, n)
        assert model_single is not None
        return model_single(n)

    if has_subway_dual:
        specs = list_pair_specs(
            root_p,
            reference_subdir=args.reference_subdir,
            noisy_subdir=args.noisy_subdir,
            band=args.band,
            subway_dual_channels=False,
            strict_all_bands=False,
        )
        chosen_sids = None
        if allowed_keys is None:
            chosen_sids = chosen_sample_ids_from_specs(
                specs,
                split=split,
                train_ratio=float(args.train_ratio),
                seed=int(args.seed),
                shuffle_split=bool(args.shuffle_split),
                cv_folds=int(args.cv_folds),
                cv_fold=int(args.cv_fold),
            )
        total = len(specs)
        stride = max(1, int(args.stride))
        limit = int(args.max_items)
        exported = 0
        for j in range(0, total, stride):
            if limit > 0 and exported >= limit:
                break
            sp = specs[j]
            if allowed_keys is not None:
                if not spec_in_allowed_keys(sp, allowed_keys):
                    continue
            else:
                m = re.match(r"^sample(\d+)_", Path(sp.reference_path).name, re.IGNORECASE)
                if m and chosen_sids and (m.group(1) not in chosen_sids):
                    continue
            noisy_path = Path(sp.noisy_path)
            if not (noisy_path.name.endswith("+subway.txt") and subway_noisy_has_four_value_columns(noisy_path)):
                continue

            c_a_s, _ = read_one_file_with_meta(sp.reference_path, value_column=2)
            c_b_s, _ = read_one_file_with_meta(sp.reference_path, value_column=3)
            tc, st = read_two_channel_file(noisy_path)
            if st.get("kept_lines", 0) < 2:
                continue
            c_a = c_a_s.value
            c_b = c_b_s.value
            a = tc.value_a
            b = tc.value_b
            L = min(len(c_a), len(c_b), len(a), len(b))
            if L < 2:
                continue
            c_a = c_a[:L]
            c_b = c_b[:L]
            a = a[:L]
            b = b[:L]

            c_a_r, _ = pad_or_resample_to_length(c_a, int(args.segment_length), mode="resample_linear")
            c_b_r, _ = pad_or_resample_to_length(c_b, int(args.segment_length), mode="resample_linear")
            a_r, _ = pad_or_resample_to_length(a, int(args.segment_length), mode="resample_linear")
            b_r, _ = pad_or_resample_to_length(b, int(args.segment_length), mode="resample_linear")

            reference_a = torch.tensor(c_a_r, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            reference_b = torch.tensor(c_b_r, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            noisy_a = torch.tensor(a_r, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            noisy_b = torch.tensor(b_r, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            if bool(args.match_noisy_scale):
                noisy_a = _affine_match_mean_std(noisy_a, reference_a)
                noisy_b = _affine_match_mean_std(noisy_b, reference_b)
            if bool(args.zscore_using_reference):
                mu_a, sig_a = _mean_std(reference_a)
                mu_b, sig_b = _mean_std(reference_b)
                reference_a_z = _zscore(reference_a, mu_a, sig_a)
                reference_b_z = _zscore(reference_b, mu_b, sig_b)
                noisy_a_z = _zscore(noisy_a, mu_a, sig_a)
                noisy_b_z = _zscore(noisy_b, mu_b, sig_b)
            else:
                reference_a_z = reference_a
                reference_b_z = reference_b
                noisy_a_z = noisy_a
                noisy_b_z = noisy_b

            den_a = _denoise_batch(noisy_a_z)
            den_b = _denoise_batch(noisy_b_z)

            stem = noisy_path.stem
            title = f"{stem} (subway dual) — reference vs noisy vs denoised"
            save_dual_column_triplet_figure(
                img_dir / f"{stem}.jpg",
                reference_a_z.squeeze().detach().cpu().numpy(),
                noisy_a_z.squeeze().detach().cpu().numpy(),
                den_a.squeeze().detach().cpu().numpy(),
                noisy_b_z.squeeze().detach().cpu().numpy(),
                den_b.squeeze().detach().cpu().numpy(),
                title=title,
                reference_b=reference_b_z.squeeze().detach().cpu().numpy(),
            )
            merged_path = res_dir / f"{stem}.txt"
            da = den_a.squeeze().detach().cpu().numpy().reshape(-1)
            db = den_b.squeeze().detach().cpu().numpy().reshape(-1)
            merged_path.parent.mkdir(parents=True, exist_ok=True)
            with merged_path.open("w", encoding="utf-8", newline="\n") as f:
                for ii, (va, vb) in enumerate(zip(da.tolist(), db.tolist())):
                    f.write(f"{ii}\t{ii}\t{va:.10g}\t{vb:.10g}\n")
            exported += 1
            if exported % 50 == 0:
                print(f"[{split}] exported {exported} / {total} ...", flush=True)

        print(f"[{split}] done. exported {exported} dual plot(s) under: {img_dir.resolve()}", flush=True)
        print(f"[{split}] done. exported {exported} merged txt(s) under: {res_dir.resolve()}", flush=True)
        return

    stride = max(1, int(args.stride))
    limit = int(args.max_items)
    export_plan = merged_dataset_export_indices(export_dss, allowed_keys)
    indices = export_plan[::stride]
    if limit > 0:
        indices = indices[:limit]
    total = len(export_plan)

    if nw > 1:
        if device.type == "cuda":
            print(
                f"[{split}] export-workers={nw}: workers use CPU; for single-GPU CUDA try "
                f"--export-workers 1 --infer-batch-size 16.",
                flush=True,
            )
        model_args = {
            "depth": int(args.depth),
            "features": int(args.features),
            "legacy_plain": bool(getattr(args, "legacy_plain", False)),
            "middle_depth": int(args.middle_depth),
            "num_residual": int(args.num_residual),
            "use_attention": bool(args.use_attention),
            "no_attention": bool(args.no_attention),
            "attention_reduction": int(args.attention_reduction),
        }
        pack = {
            "repo": str(_REPO.resolve()),
            "threecnn": str(_CNN.resolve()),
            "ensemble": is_ensemble,
            "ckpt_path": str(ckpt_path.resolve()),
            "ckpt_paths": [str(p.resolve()) for p in ckpt_paths],
            "data_root": data_root,
            "reference_subdir": args.reference_subdir,
            "noisy_subdir": args.noisy_subdir,
            "band": args.band,
            "segment_length": int(args.segment_length),
            **split_kw,
            "resample_mode": args.resample_mode,
            "match_noisy_scale": bool(args.match_noisy_scale),
            "zscore_using_reference": bool(args.zscore_using_reference),
            "manifest_dual": manifest_dual,
            "model_args": model_args,
            "img_dir": str(img_dir.resolve()),
            "res_dir": str(res_dir.resolve()),
        }
        chunks = [c for c in split_into_n_chunks(indices, nw) if c]
        if not chunks:
            print(f"[{split}] no indices to export.", flush=True)
            return
        with ProcessPoolExecutor(max_workers=len(chunks), initializer=_cnn_mp_init, initargs=(pack,)) as ex:
            futs = [ex.submit(_cnn_mp_run_chunk, ch, infer_bs) for ch in chunks]
            exported = sum(f.result() for f in as_completed(futs))
        print(f"[{split}] done. exported {exported} plot(s) under: {img_dir.resolve()}", flush=True)
        print(f"[{split}] done. exported {exported} txt(s) under: {res_dir.resolve()}", flush=True)
        return

    exported = 0
    keys: list[str] = []
    c_list: list[torch.Tensor] = []
    n_list: list[torch.Tensor] = []

    def flush_triplet_batch() -> int:
        nonlocal keys, c_list, n_list
        if not keys:
            return 0
        c_stack = torch.cat(c_list, dim=0).to(device)
        n_stack = torch.cat(n_list, dim=0).to(device)
        d_stack = _denoise_batch(n_stack)
        for bi, k in enumerate(keys):
            c = c_stack[bi : bi + 1]
            n = n_stack[bi : bi + 1]
            d = d_stack[bi : bi + 1]
            save_triplet_figure(
                img_dir / f"{k}.png",
                c.squeeze().detach().cpu().numpy(),
                n.squeeze().detach().cpu().numpy(),
                d.squeeze().detach().cpu().numpy(),
                title=f"{k} (band={args.band}) — reference vs noisy vs denoised",
                figsize=(12.0, 4.0),
                dpi=160,
                xlabel="Sample",
                ylabel="Amplitude (preprocessed)",
                noisy_label="noisy",
                denoised_label="denoised",
                reference_label="reference",
            )
            y = d.squeeze().detach().cpu().float().numpy()
            with (res_dir / f"{k}.txt").open("w", encoding="utf-8", newline="\n") as f:
                for i, v in enumerate(y.tolist()):
                    f.write(f"{i}\t{i}\t{v}\n")
        n_done = len(keys)
        keys, c_list, n_list = [], [], []
        return n_done

    for ds_i, j in indices:
        it = export_dss[int(ds_i)][int(j)]
        keys.append(str(it.get("key", f"idx_{j}")))
        c_list.append(it["reference"].unsqueeze(0))
        n_list.append(it["noisy"].unsqueeze(0))
        if len(keys) >= infer_bs:
            exported += flush_triplet_batch()
            if exported % 50 == 0:
                print(f"[{split}] exported {exported} / {total} ...", flush=True)
    exported += flush_triplet_batch()

    print(f"[{split}] done. exported {exported} plot(s) under: {img_dir.resolve()}", flush=True)
    print(f"[{split}] done. exported {exported} txt(s) under: {res_dir.resolve()}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description="cnn: DnCNN visualization/export (invoked by repo-root visualize_data.py cnn).",
        epilog="Minimal usage: python visualize_data.py cnn --data-root data1\n"
        "Default --split test is the fixed holdout test set (20%); --split all exports train pool + holdout.",
    )
    p.add_argument(
        "--data-root",
        "--our-data-root",
        type=str,
        default=".",
        dest="data_root",
        help="Data root (same as train.py --data-root; --our-data-root alias accepted).",
    )
    p.add_argument("--reference-subdir", type=str, default="reference_signal")
    p.add_argument("--noisy-subdir", type=str, default="noise_signal")
    p.add_argument("--band", type=str, default="all", choices=("low", "middle", "high", "all"))
    add_viz_split_arguments(p)
    add_viz_pooled_arguments(p)
    add_cv_ensemble_arguments(p)
    p.add_argument("--segment-length", type=int, default=1024, help="Same default as common_train_cli / train.py.")
    p.add_argument(
        "--resample-mode",
        type=str,
        default="resample_linear",
        choices=("pad_edge", "pad_zero", "resample_linear"),
        help="Same default as train_dncnn.",
    )
    p.add_argument(
        "--match-noisy-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="match_noisy_scale",
        help="Same as train_dncnn --match-noisy-scale (default on).",
    )
    p.add_argument(
        "--zscore-using-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="zscore_using_reference",
        help="Same as train_dncnn --zscore-using-reference (default on).",
    )
    p.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))

    # model params (must match training)
    p.add_argument("--depth", type=int, default=18)
    p.add_argument("--features", type=int, default=64)
    p.add_argument("--legacy-plain", action="store_true", help="Match legacy single-layer DnCNN training.")
    p.add_argument("--middle-depth", type=int, default=10)
    p.add_argument("--num-residual", type=int, default=5, dest="num_residual")
    p.add_argument("--use-attention", action="store_true", help="Align with attention-enabled training model.")
    p.add_argument("--no-attention", action="store_true", help="Force attention off.")
    p.add_argument("--attention-reduction", type=int, default=8)

    # checkpoint
    p.add_argument(
        "--ckpt",
        type=str,
        default="last",
        help="last (default last.pt, same as loss_eval) / best (best.pt) / path to a specific .pt",
    )
    p.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Weights dir; if omitted, prefer cnn/output/<dataset>/runs, then repo output/<dataset>/runs, etc.",
    )

    # output
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Image output dir; default output/cnn/<dataset>/image",
    )
    p.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Result output dir; default output/cnn/<dataset>/result",
    )

    # batch export control
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-items", type=int, default=0, help="0=all")
    p.add_argument(
        "--export-workers",
        type=int,
        default=default_export_worker_count(),
        help="Parallel export workers (default by CPU count, cap 8; workers always use CPU).",
    )
    p.add_argument(
        "--infer-batch-size",
        type=int,
        default=1,
        help="Batch infer size in single process or worker; for CUDA single GPU use --export-workers 1 and 16~64 here.",
    )
    args = p.parse_args()
    if args.use_attention and args.no_attention:
        raise SystemExit("Cannot specify both --use-attention and --no-attention")

    t0 = time.perf_counter()
    if args.split == "all":
        _run_one_split(args, split="train", clear_outputs=True)
        _run_one_split(args, split="test", clear_outputs=False)
    else:
        _run_one_split(args, split=args.split, clear_outputs=True)
    print(
        f"[cnn-viz] total {time.perf_counter() - t0:.2f}s (split={args.split} export_workers={args.export_workers})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
