"""
DnCNN-only ablation training: standalone DnCNN supervision split from TraMagNet (no UNet, no latent z, no GAN).

Defaults aligned with TraMagNet data134 5-fold 2000 epochs:
``--loss mse_time --loss-mse-weight 5 --loss-stft-weight 5``.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_DNCNN_ONLY = Path(__file__).resolve().parent
_ABLATION = Path(__file__).resolve().parent.parent
_MAIN = _ABLATION.parent / "main"
_REPO = _MAIN
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_DNCNN_ONLY))
sys.path.insert(0, str(_MAIN))

from common_train_cli import (  # noqa: E402
    add_common_train_arguments,
    resolve_train_out_dir,
    save_torch_checkpoint,
)
from data_common.resolve_dataset_root import resolve_dataset_root  # noqa: E402
from data.our_data_dataset import OurDataConfig, OurDataDataset  # noqa: E402
from models.dncnn import DnCNNDenoiser  # noqa: E402
from models.dncnn_loss import mse_time_frequency_loss, supervised_unet_loss  # noqa: E402


@dataclass(frozen=True)
class TrainCfg:
    root: str = "."
    reference_subdir: str = "reference_signal"
    noisy_subdir: str = "noise_signal"
    band: str = "all"
    segment_length: int = 1024
    batch_size: int = 32
    epochs: int = 2000
    lr: float = 2e-5
    weight_decay: float = 0.0
    seed: int = 42
    train_ratio: float = 0.8
    shuffle_split: bool = True
    cv_folds: int = 5
    cv_fold: int = 0
    num_workers: int = 0
    device: str = "cuda"
    out_dir: str = "runs"
    log_every: int = 50
    resample_mode: str = "resample_linear"
    match_noisy_scale: bool = False
    zscore_using_reference: bool = False
    loss: str = "mse_time"
    loss_mse_weight: float = 5.0
    loss_l1_weight: float = 0.0
    loss_stft_weight: float = 5.0
    subway_dual_channels: bool = True
    strict_all_bands: bool = True


def pick_device(preferred: str) -> torch.device:
    if preferred.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _our_data_config(cfg: TrainCfg, *, train: bool) -> OurDataConfig:
    return OurDataConfig(
        root=cfg.root,
        reference_subdir=cfg.reference_subdir,
        noisy_subdir=cfg.noisy_subdir,
        band=cfg.band,  # type: ignore[arg-type]
        segment_length=cfg.segment_length,
        train=train,
        train_ratio=cfg.train_ratio,
        seed=cfg.seed,
        shuffle_split=cfg.shuffle_split,
        cv_folds=int(cfg.cv_folds),
        cv_fold=int(cfg.cv_fold),
        split_round=True,
        resample_mode=cfg.resample_mode,  # type: ignore[arg-type]
        strict_all_bands=cfg.strict_all_bands,
        subway_dual_channels=cfg.subway_dual_channels,
        match_noisy_scale_to_reference=cfg.match_noisy_scale,
        zscore_using_reference=cfg.zscore_using_reference,
    )


@torch.no_grad()
def eval_mean_l1(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    w = 0.0
    for batch in loader:
        reference = batch["reference"].to(device)
        noisy = batch["noisy"].to(device)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device)
        pred = model(noisy)
        diff = (pred - reference).abs()
        if mask is None:
            total += diff.sum().item()
            w += float(diff.numel())
        else:
            m = mask.float()
            if m.dim() == 2:
                m = m.unsqueeze(1)
            total += (diff * m).sum().item()
            w += float(m.sum().item())
    model.train()
    return total / max(1e-6, w)


def train_one_fold(args: argparse.Namespace, fold: int, out_path: Path) -> None:
    _run_dncnn_training(args, fold=fold, out_path=out_path)


def _mp_cv_fold_train(pack: tuple) -> int:
    """ProcessPool worker: ``(args, fold, out_dir_str)``。"""
    args, fold, out_dir_str = pack
    train_one_fold(args, int(fold), Path(out_dir_str))
    return int(fold)


def run_cv_folds_with_workers(
    args: argparse.Namespace,
    *,
    base_out_dir: Path,
    cv_workers: int,
    log_prefix: str,
) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from data_common.cv_train import iter_cv_folds, resolve_fold_out_dir

    folds = iter_cv_folds(args)
    nf = int(getattr(args, "cv_folds", 0) or 0)
    base = Path(base_out_dir)
    workers = max(1, int(cv_workers))

    if workers <= 1 or len(folds) <= 1:
        for fold in folds:
            out = resolve_fold_out_dir(base, cv_folds=nf, cv_fold=fold)
            train_one_fold(args, fold, out)
        return

    print(f"{log_prefix} CV parallel: {len(folds)} folds, workers={workers}", flush=True)
    packs: list[tuple] = []
    for fold in folds:
        out = resolve_fold_out_dir(base, cv_folds=nf, cv_fold=fold)
        packs.append((args, fold, str(out)))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_mp_cv_fold_train, p): p[1] for p in packs}
        for fut in as_completed(futs):
            fold = futs[fut]
            try:
                fut.result()
                print(f"{log_prefix} fold_{fold} done", flush=True)
            except Exception as e:
                raise RuntimeError(f"fold_{fold} failed: {e}") from e


def _run_dncnn_training(args: argparse.Namespace, *, fold: int, out_path: Path) -> None:
    from data_common.train_data_factory import make_supervised_dataset, training_uses_pooled_data

    pooled = training_uses_pooled_data(args)
    if pooled:
        from data_common.pooled_data_split import parse_data_roots_arg

        entries = parse_data_roots_arg(str(args.data_roots), repo=_REPO)
        data_root = str(entries[0][1])
        print(
            f"[DnCNN-only ablation] pooled data: {[t for t, _ in entries]} "
            f"(split: seed={args.seed}, train_ratio={args.train_ratio})",
            flush=True,
        )
    else:
        data_root = resolve_dataset_root(args.data_root, repo=_REPO)
        if Path(data_root).resolve() != Path(args.data_root).expanduser().resolve():
            print(f"[data-root] resolved {args.data_root!r} -> {data_root}", flush=True)

    if args.loss in ("time", "default") and float(args.loss_l1_weight) == 0.0 and float(args.loss_stft_weight) == 0.0:
        args.loss_l1_weight = 1.0

    lr = float(args.lr) if args.lr is not None else 2e-5
    cfg = TrainCfg(
        root=data_root,
        reference_subdir=args.reference_subdir,
        noisy_subdir=args.noisy_subdir,
        band=args.band,
        segment_length=int(args.segment_length),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=lr,
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        train_ratio=float(args.train_ratio),
        shuffle_split=bool(args.shuffle_split),
        cv_folds=int(args.cv_folds),
        cv_fold=int(fold),
        num_workers=int(args.num_workers),
        device=str(args.device),
        out_dir=str(out_path),
        log_every=int(args.log_every),
        resample_mode=str(args.resample_mode),
        match_noisy_scale=bool(args.match_noisy_scale),
        zscore_using_reference=bool(args.zscore_using_reference),
        loss=str(args.loss),
        loss_mse_weight=float(args.loss_mse_weight),
        loss_l1_weight=float(args.loss_l1_weight),
        loss_stft_weight=float(args.loss_stft_weight),
        subway_dual_channels=bool(args.subway_dual_channels),
        strict_all_bands=bool(args.strict_all_bands),
    )

    if cfg.loss == "mse_time":
        from models.dncnn_loss import resolve_mix_weights

        try:
            resolve_mix_weights(cfg.loss_mse_weight, cfg.loss_l1_weight, cfg.loss_stft_weight)
        except ValueError as e:
            raise SystemExit(f"mse_time: {e}") from e
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "config.txt").write_text(str(asdict(cfg)), encoding="utf-8")

    ds_tr = make_supervised_dataset(
        args, cfg, repo=_REPO, train=True, our_data_config_cls=OurDataConfig, our_data_dataset_cls=OurDataDataset
    )
    ds_va = make_supervised_dataset(
        args, cfg, repo=_REPO, train=False, our_data_config_cls=OurDataConfig, our_data_dataset_cls=OurDataDataset
    )

    if cfg.loss == "2unet":
        loss_banner = "loss=MSE(pred,reference) [2unet]"
    elif cfg.loss in ("time", "default"):
        loss_banner = (
            f"loss=norm(L1)*{cfg.loss_l1_weight}+norm(STFT)*{cfg.loss_stft_weight} (mix→1)"
        )
    else:
        loss_banner = (
            f"loss=norm(MSE)*{cfg.loss_mse_weight}+norm(L1)*{cfg.loss_l1_weight}"
            f"+norm(STFT)*{cfg.loss_stft_weight} [mse_time, mix→1]"
        )
    from data_common.cv_train import cv_fold_log_prefix, format_cv_fold_info

    fold_info = format_cv_fold_info(fold, cfg.cv_folds)
    fp = cv_fold_log_prefix(fold, cfg.cv_folds)
    cv_note = f" {fold_info}" if fold_info else ""

    train_loader = DataLoader(
        ds_tr,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        ds_va,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = DnCNNDenoiser().to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best = float("inf")
    step = 0
    start_epoch = 1
    last_ckpt = out_path / "last.pt"

    def _torch_load_last(path: Path) -> dict:
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=device)

    if args.auto_resume and last_ckpt.is_file():
        ckpt: dict | None = None
        try:
            sz = last_ckpt.stat().st_size
            if sz < 32:
                raise OSError(f"File too small ({sz} bytes); write may be incomplete")
            ckpt = _torch_load_last(last_ckpt)
        except (EOFError, OSError, RuntimeError, pickle.UnpicklingError) as e:
            print(
                f"{fp}[resume] Cannot read {last_ckpt} (corrupt or incomplete write); training from scratch: {type(e).__name__}: {e}",
                flush=True,
            )
            ckpt = None
        if ckpt is not None and isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=True)
            if "opt" in ckpt:
                opt.load_state_dict(ckpt["opt"])
            step = int(ckpt.get("step", 0))
            best = float(ckpt.get("best_val_l1", ckpt.get("val_l1", best)))
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            print(
                f"{fp}[resume] loaded {last_ckpt} (epoch={start_epoch - 1}, step={step}, best_val_l1={best:.6f})",
                flush=True,
            )
            if start_epoch > cfg.epochs:
                print(
                    f"{fp}[resume] Already trained to epoch={start_epoch - 1} (>= target epochs={cfg.epochs}); exiting.",
                    flush=True,
                )
                return
        elif ckpt is not None:
            print(f"{fp}[resume] {last_ckpt} missing 'model' key; training from scratch.", flush=True)

    print(
        f"[DnCNN-only ablation]{cv_note} model=models.dncnn.DnCNNDenoiser | data="
        f"{'pooled' if pooled else 'DnCNN-only ablation.data.OurDataDataset'} | band={cfg.band} T={cfg.segment_length} "
        f"resample_mode={cfg.resample_mode} match_noisy={cfg.match_noisy_scale} zscore_reference={cfg.zscore_using_reference} "
        f"subway_dual={cfg.subway_dual_channels} strict_all_bands={cfg.strict_all_bands} "
        f"{loss_banner}",
        flush=True,
    )
    print(
        f"{fp}[data] total_pairs={ds_tr.n_total_pairs} unique_sample_ids={ds_tr.n_unique_samples} "
        f"train_pairs={len(ds_tr)} val_pairs={len(ds_va)}",
        flush=True,
    )

    for epoch in range(start_epoch, cfg.epochs + 1):
        for batch in train_loader:
            reference = batch["reference"].to(device)
            noisy = batch["noisy"].to(device)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(device)

            pred = model(noisy)
            if cfg.loss == "2unet":
                loss = F.mse_loss(pred, reference)
                l_mse = loss
                l_l1 = l_stft = None
            elif cfg.loss in ("time", "default"):
                loss, l_l1, l_stft = supervised_unet_loss(
                    pred,
                    reference,
                    mask,
                    loss_l1_weight=cfg.loss_l1_weight,
                    loss_stft_weight=cfg.loss_stft_weight,
                )
                l_mse = None
            else:
                loss, l_mse, l_l1, l_stft = mse_time_frequency_loss(
                    pred,
                    reference,
                    mask,
                    loss_mse_weight=cfg.loss_mse_weight,
                    loss_l1_weight=cfg.loss_l1_weight,
                    loss_stft_weight=cfg.loss_stft_weight,
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            if step % cfg.log_every == 0:
                if cfg.loss == "2unet":
                    print(f"{fp}epoch={epoch} step={step} loss_mse={loss.item():.6f}", flush=True)
                elif cfg.loss in ("time", "default"):
                    if cfg.loss_stft_weight != 0.0:
                        print(
                            f"{fp}epoch={epoch} step={step} loss={loss.item():.6f} "
                            f"(l_l1={l_l1.item():.6f} l_stft={l_stft.item():.6f})",
                            flush=True,
                        )
                    else:
                        print(f"{fp}epoch={epoch} step={step} loss={loss.item():.6f} (l_l1 only)", flush=True)
                elif cfg.loss_stft_weight != 0.0:
                    print(
                        f"{fp}epoch={epoch} step={step} loss={loss.item():.6f} "
                        f"(l_mse={l_mse.item():.6f} l_l1={l_l1.item():.6f} l_stft={l_stft.item():.6f})",
                        flush=True,
                    )
                else:
                    print(
                        f"{fp}epoch={epoch} step={step} loss={loss.item():.6f} "
                        f"(l_mse={l_mse.item():.6f} l_l1={l_l1.item():.6f})",
                        flush=True,
                    )
            step += 1

        val_l1 = eval_mean_l1(model, val_loader, device)
        print(f"{fp}epoch={epoch} val_L1_mean={val_l1:.6f}", flush=True)
        if val_l1 < best:
            best = val_l1
            save_torch_checkpoint(
                out_path / "best.pt",
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_l1": val_l1,
                    "resample_mode": cfg.resample_mode,
                    "match_noisy_scale": cfg.match_noisy_scale,
                    "zscore_using_reference": cfg.zscore_using_reference,
                    "loss": cfg.loss,
                    "loss_mse_weight": cfg.loss_mse_weight,
                    "loss_l1_weight": cfg.loss_l1_weight,
                    "loss_stft_weight": cfg.loss_stft_weight,
                    "cv_fold": int(fold),
                },
            )
            print(f"{fp}  -> saved best.pt (val_L1={val_l1:.6f})", flush=True)
        meta = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
            "step": step,
            "val_l1": val_l1,
            "best_val_l1": best,
            "resample_mode": cfg.resample_mode,
            "match_noisy_scale": cfg.match_noisy_scale,
            "zscore_using_reference": cfg.zscore_using_reference,
            "loss": cfg.loss,
            "loss_mse_weight": cfg.loss_mse_weight,
            "loss_l1_weight": cfg.loss_l1_weight,
            "loss_stft_weight": cfg.loss_stft_weight,
            "cv_fold": int(fold),
        }
        save_torch_checkpoint(out_path / "last.pt", meta)

    print(f"{fp}done. outputs at: {out_path.resolve()}")


def main(argv: list[str] | None = None) -> None:
    from data_common.cv_train import collect_sorted_sample_ids, maybe_write_split_manifest

    p = argparse.ArgumentParser(
        description="DnCNN-only ablation: standalone supervised training of the TraMagNet DnCNN branch (no UNet / no GAN)."
    )
    add_common_train_arguments(p)
    p.add_argument("--weight-decay", type=float, default=0.0, dest="weight_decay")
    p.add_argument("--log-every", type=int, default=50, dest="log_every")
    p.add_argument(
        "--resample-mode",
        type=str,
        default="resample_linear",
        choices=("pad_edge", "pad_zero", "resample_linear"),
    )
    p.add_argument(
        "--match-noisy-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="match_noisy_scale",
    )
    p.add_argument(
        "--zscore-using-reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="zscore_using_reference",
    )
    p.add_argument(
        "--loss-mse-weight",
        type=float,
        default=5.0,
        dest="loss_mse_weight",
        metavar="W",
    )
    p.add_argument(
        "--loss-l1-weight",
        type=float,
        default=0.0,
        dest="loss_l1_weight",
        metavar="W",
    )
    p.add_argument(
        "--loss-stft-weight",
        type=float,
        default=5.0,
        dest="loss_stft_weight",
        metavar="W",
    )
    p.add_argument(
        "--loss",
        type=str,
        default="mse_time",
        choices=("mse_time", "time", "default", "2unet"),
        dest="loss",
    )
    p.add_argument(
        "--subway-dual-channels",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="subway_dual_channels",
    )
    p.add_argument(
        "--strict-all-bands",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="strict_all_bands",
    )
    p.add_argument(
        "--cv-workers",
        type=int,
        default=2,
        dest="cv_workers",
        help="Number of CV folds to train concurrently (process pool size; default 2).",
    )
    p.set_defaults(epochs=2000)
    args = p.parse_args(argv)

    from data_common.train_data_factory import training_uses_pooled_data

    if training_uses_pooled_data(args):
        from data_common.pooled_data_split import parse_data_roots_arg

        entries = parse_data_roots_arg(str(args.data_roots), repo=_REPO)
        data_root = str(entries[0][1])
    else:
        data_root = resolve_dataset_root(args.data_root, repo=_REPO)
        if not bool(getattr(args, "no_cv_split_manifest", False)) and int(args.cv_folds) > 0:
            sids = collect_sorted_sample_ids(
                data_root,
                reference_subdir=args.reference_subdir,
                noisy_subdir=args.noisy_subdir,
                band=args.band,
                subway_dual_channels=bool(args.subway_dual_channels),
                strict_all_bands=bool(args.strict_all_bands),
            )
            maybe_write_split_manifest(
                manifest_path=Path(resolve_train_out_dir(args, data_root)) / "split_manifest.json",
                sids=sids,
                train_ratio=float(args.train_ratio),
                seed=int(args.seed),
                shuffle_split=bool(args.shuffle_split),
                cv_folds=int(args.cv_folds),
                skip=False,
            )

    base_out = resolve_train_out_dir(args, data_root)
    base_out_path = Path(base_out)
    base_out_path.mkdir(parents=True, exist_ok=True)
    print(f"[DnCNN-only ablation] checkpoint directory: {base_out_path.resolve()}", flush=True)

    run_cv_folds_with_workers(
        args,
        base_out_dir=base_out_path,
        cv_workers=int(args.cv_workers),
        log_prefix="[DnCNN-only ablation]",
    )


if __name__ == "__main__":
    main(None)
