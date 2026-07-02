"""
TraMagNet visualization/export (invoked by repo-root ``visualize_data.py tramagnet``).

Weights from ``TraMagNet`` training (``checkpoint`` contains ``generator``); ``UNet`` defined in ``TraMagNet/models/unet.py``.
Data and preprocessing align with the ``dncnn`` pipeline via ``TraMagNet/data/our_data_dataset.py`` (``dncnn`` is not added to ``sys.path``):
``OurDataDataset`` + ``resample_mode`` / ``split_round`` / ``strict_all_bands`` /
``match_noisy_scale_to_reference`` / ``zscore_using_reference``; dual-channel export branch for four-column ``+subway`` files.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
_TRAMAGNET = _REPO / "TraMagNet"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_TRAMAGNET))

from data_common.cv_ensemble import add_cv_ensemble_arguments
from data_common.ensemble_infer import load_unet_ensemble, unet_ensemble_forward
from data_common.viz_ckpt_resolve import resolve_viz_inference_plan
from data_common.viz_split import (
    add_viz_split_arguments,
    chosen_sample_ids_from_specs,
    describe_split_for_log,
    maybe_sync_split_from_runs_config,
    our_data_dataset_split_kwargs,
)
from data_common.viz_export_workers import (  # noqa: E402
    checkpoint_run_candidates,
    default_export_worker_count,
    empty_viz_output_dirs,
    split_into_n_chunks,
)
from data_common.viz_pooled import (
    add_viz_pooled_arguments,
    clear_viz_output_base,
    dataset_export_indices,
    prefixed_export_stem,
    resolve_viz_pool_plan,
    resolve_viz_target_output_dirs,
    spec_in_allowed_keys,
    split_kwargs_for_viz_target,
)

from models.unet import (  # noqa: E402
    UNET_LATENT_CHANNELS,
    UNET_LATENT_LENGTH,
    UNet,
    complete_unet_state_dict,
)

# Multiprocess worker state (one copy per spawn child)
_VIZ4_MP: dict = {}


def _gan5_ckpt_sd(payload: dict | object) -> dict:
    if not isinstance(payload, dict):
        return payload  # type: ignore[return-value]
    if "generator" in payload:
        return payload["generator"]
    if "model" in payload:
        return payload["model"]
    return payload  # type: ignore[return-value]


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


def _affine_match_mean_std(noisy: torch.Tensor, clean: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    mu_c, sig_c = _mean_std(clean, eps=eps)
    mu_n, sig_n = _mean_std(noisy, eps=eps)
    return (noisy - mu_n) * (sig_c / sig_n) + mu_c


def _zscore(x: torch.Tensor, mu: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
    return (x - mu) / sig


def _viz_export_basename(key: str, *, filename_suffix: str) -> str:
    """Export filename stem; when split=all, suffix is ``(train)`` / ``(test)``."""
    s = (filename_suffix or "").strip()
    return f"{key}{s}" if s else str(key)


def _infer_z(
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    mode: str,
) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros(
            batch_size,
            UNET_LATENT_CHANNELS,
            UNET_LATENT_LENGTH,
            device=device,
            dtype=dtype,
        )
    return torch.randn(
        batch_size,
        UNET_LATENT_CHANNELS,
        UNET_LATENT_LENGTH,
        device=device,
        dtype=dtype,
    )


def _tramagnet_mp_init(pack: dict) -> None:
    global _VIZ4_MP
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    sys.path.insert(0, pack["repo"])
    sys.path.insert(0, pack["tramagnet"])

    import torch as _t

    _t.set_num_threads(1)

    from data.our_data_dataset import OurDataConfig, OurDataDataset
    from models.unet import UNet

    device = _t.device("cpu")
    ds = OurDataDataset(
        OurDataConfig(
            root=pack["data_root"],
            reference_subdir=pack["reference_subdir"],
            noisy_subdir=pack["noisy_subdir"],
            band=pack["band"],
            segment_length=int(pack["segment_length"]),
            train=bool(pack["train"]),
            holdout_eval=bool(pack["holdout_eval"]),
            cv_folds=int(pack["cv_folds"]),
            cv_fold=int(pack["cv_fold"]),
            train_ratio=float(pack["train_ratio"]),
            seed=int(pack["seed"]),
            shuffle_split=bool(pack["shuffle_split"]),
            split_round=True,
            resample_mode=pack["resample_mode"],
            strict_all_bands=True,
            match_noisy_scale_to_reference=bool(pack["match_noisy_scale"]),
            zscore_using_reference=bool(pack["zscore_using_reference"]),
        )
    )
    z_mode = str(pack["z_mode"])
    mp_state = {
        "device": device,
        "ds": ds,
        "z_mode": z_mode,
        "img_dir": Path(pack["img_dir"]),
        "res_dir": Path(pack["res_dir"]),
        "band": str(pack["band"]),
        "filename_suffix": str(pack.get("filename_suffix") or ""),
    }
    if pack.get("ensemble"):
        members = load_unet_ensemble(pack["ckpt_paths"], device, gan_generator=True)
        mp_state["members"] = members
    else:
        model = UNet().to(device)
        payload = _t.load(pack["ckpt_path"], map_location=device)
        sd = _gan5_ckpt_sd(payload)
        model.load_state_dict(complete_unet_state_dict(model, sd), strict=True)
        model.eval()
        mp_state["model"] = model
    _VIZ4_MP = mp_state


@torch.no_grad()
def _tramagnet_mp_run_chunk(chunk_indices: list[int], infer_bs: int) -> int:
    from data_common.viz_export import save_triplet_figure

    g = _VIZ4_MP
    device = g["device"]
    ds = g["ds"]
    z_mode = str(g["z_mode"])
    img_dir = g["img_dir"]
    res_dir = g["res_dir"]
    band = g["band"]
    fn_suf = str(g.get("filename_suffix") or "")
    members = g.get("members")
    model = g.get("model")

    def denoise_local(n: torch.Tensor) -> torch.Tensor:
        bsz = n.size(0)
        z = _infer_z(batch_size=bsz, device=device, dtype=n.dtype, mode=z_mode)
        if members is not None:
            return unet_ensemble_forward(members, n, z)
        assert model is not None
        return model(n, z)

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
        d_stack = denoise_local(n_stack)
        for bi, k in enumerate(keys):
            c = c_stack[bi : bi + 1]
            n = n_stack[bi : bi + 1]
            d = d_stack[bi : bi + 1]
            base = prefixed_export_stem(k, dataset_tag=None, filename_suffix=fn_suf)
            save_triplet_figure(
                img_dir / f"{base}.png",
                c.squeeze().detach().cpu().numpy(),
                n.squeeze().detach().cpu().numpy(),
                d.squeeze().detach().cpu().numpy(),
                title=f"{k} (band={band}) — clean vs noisy vs denoised",
                figsize=(12.0, 4.0),
                dpi=160,
                xlabel="Sample",
                ylabel="Amplitude (preprocessed)",
                noisy_label="noisy",
                denoised_label="denoised",
                reference_label="reference",
            )
            y = d.squeeze().detach().cpu().float().numpy()
            with (res_dir / f"{base}.txt").open("w", encoding="utf-8", newline="\n") as f:
                for ii, v in enumerate(y.tolist()):
                    f.write(f"{ii}\t{ii}\t{v}\n")
        n_out = len(keys)
        keys, c_list, n_list = [], [], []
        return n_out

    for j in chunk_indices:
        it = ds[j]
        keys.append(str(it.get("key", f"idx_{j}")))
        c_list.append(it["reference"].unsqueeze(0))
        n_list.append(it["noisy"].unsqueeze(0))
        if len(keys) >= infer_bs:
            exported += flush()
    exported += flush()
    return exported


@torch.no_grad()
def _run_one_split(
    args: argparse.Namespace,
    *,
    split: str,
    clear_outputs: bool = True,
    filename_suffix: str = "",
) -> None:
    from data.our_data_dataset import OurDataConfig, OurDataDataset
    from data_common.flat_pairing import sample_id_from_reference_path
    from data_common.pair_specs import list_pair_specs
    from data_common.txt_io import (
        pad_or_resample_to_length,
        read_one_file_with_meta,
        read_two_channel_file,
        subway_noisy_has_four_value_columns,
    )
    from data_common.viz_export import save_dual_column_triplet_figure, save_triplet_figure

    pool_plan = resolve_viz_pool_plan(args, _REPO, split=split)
    out_tag = pool_plan.output_tag
    first_root = pool_plan.targets[0].data_root
    plan = resolve_viz_inference_plan(
        args, repo=_REPO, data_root=first_root, data_tag=out_tag, nn_dir=_TRAMAGNET
    )
    ckpt_paths = list(plan.ckpt_paths)
    is_ensemble = plan.mode == "ensemble"
    ckpt_path = ckpt_paths[0]

    if not getattr(args, "_viz_tramagnet_runs_split_synced", False):
        maybe_sync_split_from_runs_config(args, runs_dir=plan.config_dir, log_prefix="[TraMagNet-viz]")
        setattr(args, "_viz_tramagnet_runs_split_synced", True)

    print(
        f"[TraMagNet-viz] split={split} -> {describe_split_for_log(split, cv_folds=int(args.cv_folds), cv_fold=int(args.cv_fold))}"
        + (f" pooled={out_tag} n_roots={len(pool_plan.targets)}" if pool_plan.is_pooled else ""),
        flush=True,
    )

    _out_base = Path("output") / _TRAMAGNET.name / out_tag
    user_img_dir = Path(args.output_dir) if args.output_dir else None
    user_res_dir = Path(args.result_dir) if args.result_dir else None
    if clear_outputs and pool_plan.is_pooled:
        clear_viz_output_base(_out_base)
    elif clear_outputs and not pool_plan.is_pooled:
        img0, res0 = resolve_viz_target_output_dirs(
            out_base=_out_base,
            dataset_tag=pool_plan.targets[0].dataset_tag,
            is_pooled=False,
            output_dir=user_img_dir,
            result_dir=user_res_dir,
        )
        empty_viz_output_dirs(img0, res0)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    members = None
    model = None
    if is_ensemble:
        members = load_unet_ensemble(ckpt_paths, device, gan_generator=True)
        print(f"[{split}] cv ensemble ({len(ckpt_paths)} folds)", flush=True)
    else:
        model = UNet().to(device)
        payload = torch.load(ckpt_path, map_location=device)
        sd = _gan5_ckpt_sd(payload)
        model.load_state_dict(complete_unet_state_dict(model, sd), strict=True)
        model.eval()

    def denoise(n: torch.Tensor) -> torch.Tensor:
        bsz = n.size(0)
        z = _infer_z(batch_size=bsz, device=device, dtype=n.dtype, mode=str(args.z_mode))
        if members is not None:
            return unet_ensemble_forward(members, n, z)
        assert model is not None
        return model(n, z)

    for tgt_i, tgt in enumerate(pool_plan.targets):
        data_root = tgt.data_root
        allowed_keys = tgt.allowed_keys
        img_dir, res_dir = resolve_viz_target_output_dirs(
            out_base=_out_base,
            dataset_tag=tgt.dataset_tag,
            is_pooled=pool_plan.is_pooled,
            output_dir=user_img_dir,
            result_dir=user_res_dir,
        )
        img_dir.mkdir(parents=True, exist_ok=True)
        res_dir.mkdir(parents=True, exist_ok=True)
        split_kw = split_kwargs_for_viz_target(
            split,
            allowed_keys=allowed_keys,
            train_ratio=float(args.train_ratio),
            seed=int(args.seed),
            shuffle_split=bool(args.shuffle_split),
            cv_folds=int(getattr(args, "cv_folds", 0)),
            cv_fold=int(getattr(args, "cv_fold", 0)),
        )
        if pool_plan.is_pooled:
            print(f"[TraMagNet-viz] export data_root={tgt.dataset_tag} -> {data_root}", flush=True)

        root_p = Path(data_root)
        noisy_dir = root_p / args.noisy_subdir
        has_subway_dual = False
        if noisy_dir.is_dir():
            for p in noisy_dir.glob("sample*.txt"):
                if subway_noisy_has_four_value_columns(p):
                    has_subway_dual = True
                    break

        ds = OurDataDataset(
            OurDataConfig(
                root=data_root,
                reference_subdir=args.reference_subdir,
                noisy_subdir=args.noisy_subdir,
                band=args.band,  # type: ignore[arg-type]
                segment_length=int(args.segment_length),
                **split_kw,
                split_round=True,
                resample_mode=args.resample_mode,  # type: ignore[arg-type]
                strict_all_bands=True,
                match_noisy_scale_to_reference=bool(args.match_noisy_scale),
                zscore_using_reference=bool(args.zscore_using_reference),
            )
        )

        infer_bs = max(1, int(getattr(args, "infer_batch_size", 1)))
        nw = max(1, int(getattr(args, "export_workers", 1)))

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
                    data_root=root_p,
                )
            total = len(specs)
            stride = max(1, int(args.stride))
            limit = int(args.max_items)
            exported = 0
            for j in range(0, total, stride):
                if limit > 0 and exported >= limit:
                    break
                sp = specs[j]
                if not spec_in_allowed_keys(sp, allowed_keys):
                    continue
                sid = sample_id_from_reference_path(sp.reference_path, data_root=root_p)
                if sid and chosen_sids and sid not in chosen_sids:
                    continue
                noisy_path = Path(sp.noisy_path)
                if not subway_noisy_has_four_value_columns(noisy_path):
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

                rm = str(args.resample_mode)
                c_a_r, _ = pad_or_resample_to_length(c_a, int(args.segment_length), mode=rm)
                c_b_r, _ = pad_or_resample_to_length(c_b, int(args.segment_length), mode=rm)
                a_r, _ = pad_or_resample_to_length(a, int(args.segment_length), mode=rm)
                b_r, _ = pad_or_resample_to_length(b, int(args.segment_length), mode=rm)

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

                den_a = denoise(noisy_a_z)
                den_b = denoise(noisy_b_z)

                stem = noisy_path.stem
                stem_out = prefixed_export_stem(stem, dataset_tag=None, filename_suffix=filename_suffix)
                title = f"{stem} (subway dual) — clean vs noisy vs denoised"
                save_dual_column_triplet_figure(
                    img_dir / f"{stem_out}.jpg",
                    reference_a_z.squeeze().detach().cpu().numpy(),
                    noisy_a_z.squeeze().detach().cpu().numpy(),
                    den_a.squeeze().detach().cpu().numpy(),
                    noisy_b_z.squeeze().detach().cpu().numpy(),
                    den_b.squeeze().detach().cpu().numpy(),
                    title=title,
                    reference_b=reference_b_z.squeeze().detach().cpu().numpy(),
                )
                merged_path = res_dir / f"{stem_out}.txt"
                da = den_a.squeeze().detach().cpu().numpy().reshape(-1)
                db = den_b.squeeze().detach().cpu().numpy().reshape(-1)
                merged_path.parent.mkdir(parents=True, exist_ok=True)
                with merged_path.open("w", encoding="utf-8", newline="\n") as f:
                    for ii, (va, vb) in enumerate(zip(da.tolist(), db.tolist())):
                        f.write(f"{ii}\t{ii}\t{va:.10g}\t{vb:.10g}\n")
                exported += 1
                if exported % 50 == 0:
                    print(f"[{split}] exported {exported} / {total} ...", flush=True)

            print(
                f"[{split}] {tgt.dataset_tag} done. exported {exported} dual plot(s) -> {img_dir.resolve()}",
                flush=True,
            )
            continue

        stride = max(1, int(args.stride))
        limit = int(args.max_items)
        all_indices = dataset_export_indices(ds, allowed_keys)
        total = len(all_indices)
        idx_only = int(getattr(args, "idx_only", -1))
        if idx_only >= 0:
            j0 = idx_only % max(total, 1)
            indices = [all_indices[j0]] if total else []
        else:
            indices = all_indices[::stride]
        if limit > 0:
            indices = indices[:limit]

        if nw > 1:
            if device.type == "cuda":
                print(
                    f"[{split}] export-workers={nw}: workers use CPU; for single-GPU CUDA try "
                    f"--export-workers 1 --infer-batch-size 16 (or larger) for speed.",
                    flush=True,
                )
            if is_ensemble:
                print(
                    f"[{split}] cv ensemble ({len(ckpt_paths)} folds) z_mode={args.z_mode} "
                    f"export_workers={nw} infer_batch_size={infer_bs}",
                    flush=True,
                )
            else:
                print(
                    f"[{split}] ckpt={ckpt_path} z_mode={args.z_mode} "
                    f"export_workers={nw} infer_batch_size={infer_bs}",
                    flush=True,
                )
            pack = {
                "repo": str(_REPO.resolve()),
                "tramagnet": str(_TRAMAGNET.resolve()),
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
                "z_mode": str(args.z_mode),
                "img_dir": str(img_dir.resolve()),
                "res_dir": str(res_dir.resolve()),
                "filename_suffix": str(filename_suffix),
            }
            chunks = [c for c in split_into_n_chunks(indices, nw) if c]
            if not chunks:
                print(f"[{split}] {tgt.dataset_tag} no indices to export.", flush=True)
                continue
            with ProcessPoolExecutor(max_workers=len(chunks), initializer=_tramagnet_mp_init, initargs=(pack,)) as ex:
                futs = [ex.submit(_tramagnet_mp_run_chunk, ch, infer_bs) for ch in chunks]
                exported = sum(f.result() for f in as_completed(futs))
            print(f"[{split}] {tgt.dataset_tag} done. exported {exported} plot(s).", flush=True)
            continue

        if not is_ensemble:
            print(
                f"[{split}] ckpt={ckpt_path} z_mode={args.z_mode} "
                f"infer_batch_size={infer_bs}",
                flush=True,
            )

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
            d_stack = denoise(n_stack)
            for bi, k in enumerate(keys):
                c = c_stack[bi : bi + 1]
                n = n_stack[bi : bi + 1]
                d = d_stack[bi : bi + 1]
                base = prefixed_export_stem(k, dataset_tag=None, filename_suffix=filename_suffix)
                save_triplet_figure(
                    img_dir / f"{base}.png",
                    c.squeeze().detach().cpu().numpy(),
                    n.squeeze().detach().cpu().numpy(),
                    d.squeeze().detach().cpu().numpy(),
                    title=f"{k} (band={args.band}) — clean vs noisy vs denoised",
                    figsize=(12.0, 4.0),
                    dpi=160,
                    xlabel="Sample",
                    ylabel="Amplitude (preprocessed)",
                    noisy_label="noisy",
                    denoised_label="denoised",
                    reference_label="reference",
                )
                y = d.squeeze().detach().cpu().float().numpy()
                with (res_dir / f"{base}.txt").open("w", encoding="utf-8", newline="\n") as f:
                    for i, v in enumerate(y.tolist()):
                        f.write(f"{i}\t{i}\t{v}\n")
            n_done = len(keys)
            keys, c_list, n_list = [], [], []
            return n_done

        for j in indices:
            it = ds[j]
            keys.append(str(it.get("key", f"idx_{j}")))
            c_list.append(it["reference"].unsqueeze(0))
            n_list.append(it["noisy"].unsqueeze(0))
            if len(keys) >= infer_bs:
                exported += flush_triplet_batch()
                if exported % 50 == 0:
                    print(f"[{split}] exported {exported} / {total} ...", flush=True)
        exported += flush_triplet_batch()

        print(f"[{split}] {tgt.dataset_tag} done. exported {exported} plot(s).", flush=True)

    print(f"[TraMagNet-viz] all roots done under: {_out_base.resolve()}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description="TraMagNet UNet generator visualization/export (via visualize_data.py tramagnet).",
    )
    p.add_argument(
        "--data-root",
        "--our-data-root",
        type=str,
        default=".",
        dest="data_root",
        help="Data root; data134 selects pooled mode (data1+data3+data4).",
    )
    p.add_argument("--reference-subdir", type=str, default="reference_signal", dest="reference_subdir")
    p.add_argument("--noisy-subdir", type=str, default="noise_signal")
    p.add_argument("--band", type=str, default="all", choices=("low", "middle", "high", "all"))
    add_viz_split_arguments(p)
    add_viz_pooled_arguments(p)
    add_cv_ensemble_arguments(p)
    p.epilog = (
        "Minimal usage: python visualize_data.py tramagnet --data-root data1\n"
        "Pooled: python visualize_data.py tramagnet --data-root data134 --split test\n"
        "(exports to output/TraMagNet/data134/{data1,data3,data4}/ each with image, result)\n"
        "Default --split test is fixed holdout test set; --split all exports train pool + holdout."
    )
    p.add_argument("--segment-length", type=int, default=1024)
    p.add_argument(
        "--resample-mode",
        type=str,
        default="resample_linear",
        choices=("pad_edge", "pad_zero", "resample_linear"),
        help="Same as train_DnCNN / OurDataDataset.",
    )
    p.add_argument(
        "--match-noisy-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="match_noisy_scale",
        help="Same as train_DnCNN (default on).",
    )
    p.add_argument(
        "--zscore-using-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="zscore_using_reference",
        help="Same as train_DnCNN (default on).",
    )
    p.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    p.add_argument(
        "--z-mode",
        type=str,
        default="zero",
        choices=("zero", "random"),
        dest="z_mode",
        help="Latent: zero matches GAN validation; random resamples each forward.",
    )
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
        help="Weights dir; if omitted, prefer TraMagNet/output/<dataset>/runs, then repo output/<dataset>/runs, etc.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Image output dir; default output/TraMagNet/<dataset>/image",
    )
    p.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Result output dir; default output/TraMagNet/<dataset>/result",
    )
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-items", type=int, default=0, help="0=all")
    p.add_argument(
        "--export-workers",
        type=int,
        default=default_export_worker_count(),
        help="Parallel export workers (default by CPU count, cap 8; workers use CPU). For CUDA single GPU, set 1 and increase --infer-batch-size.",
    )
    p.add_argument(
        "--infer-batch-size",
        type=int,
        default=1,
        help="Infer batch size per process/worker; for CUDA single GPU, 16~64 with --export-workers 1.",
    )
    p.add_argument(
        "--idx-only",
        type=int,
        default=-1,
        dest="idx_only",
        help="When >=0, export only one sample at this dataset index (hooks repo-root viz single preview); default -1 batches by stride/max-items.",
    )
    args = p.parse_args()

    t0 = time.perf_counter()
    if args.split == "all":
        _run_one_split(args, split="train", clear_outputs=True, filename_suffix="(train)")
        _run_one_split(args, split="test", clear_outputs=False, filename_suffix="(test)")
    else:
        _run_one_split(args, split=args.split, clear_outputs=True, filename_suffix="")
    dt = time.perf_counter() - t0
    print(f"[TraMagNet-viz] total {dt:.2f}s (split={args.split} export_workers={args.export_workers})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
