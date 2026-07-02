"""
TraMagNet: conditional LSGAN + supervised loss aligned with TraMagNet (generator and loss implementations only).

- **G**: ``UNet`` (loaded from ``TraMagNet/models/unet.py``, forward consistent with TraMagNet).
- **D**: ``Discriminator`` from ``TraMagNet/models/discriminator.py``.
- **Supervised terms**: ``mse_time_frequency_loss`` / ``supervised_unet_loss`` (loaded from ``TraMagNet/models/unet_loss.py``).

``UNet.forward``: decoder directly outputs denoised ``(B,1,T)`` (no ``noisy + dx`` output skip connection).

**Data**: ``TraMagNet/data/our_data_dataset.py`` in this directory (fields consistent with TraMagNet pipeline); does not mount ``TraMagNet/`` via ``sys.path``.

Usage (repo root)::

    python TraMagNet/train.py --data-root data1 --epochs 50

Or ``cd TraMagNet && python train.py --data-root ../data1``.

**data1–data4**: ``OurDataDataset``. ``--data-root data3`` is auto-resolved under repo root;
data3 ``*+subway.txt`` with two value columns defaults to ``--subway-dual-channels`` (consistent with visualization).

When ``--out-dir`` is omitted, pooled training (``--data-roots``) writes weights to ``output/<pool-tag>/runs`` (default ``output/data134/runs``); single dataset uses ``output/<data-dir-name>/runs``.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

_FIVE = Path(__file__).resolve().parent
_REPO = _FIVE.parent

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_FIVE))

from common_train_cli import (  # noqa: E402
    add_common_train_arguments,
    resolve_train_out_dir,
    save_torch_checkpoint,
)
from data_common.resolve_dataset_root import resolve_dataset_root  # noqa: E402
from data.our_data_dataset import OurDataConfig, OurDataDataset  # noqa: E402
from models.discriminator import Discriminator  # noqa: E402
from models.unet import UNET_LATENT_CHANNELS, UNET_LATENT_LENGTH, UNet, sample_latent  # noqa: E402
from models.unet_loss import (  # noqa: E402
    LOSS_TERM_SCALE_L1,
    LOSS_TERM_SCALE_MSE,
    LOSS_TERM_SCALE_STFT,
    mix_score_from_raw,
    mse_time_frequency_loss,
    resolve_mix_weights,
    supervised_unet_loss,
)


@dataclass(frozen=True)
class Gan5Cfg:
    root: str
    reference_subdir: str
    noisy_subdir: str
    band: str
    segment_length: int
    batch_size: int
    epochs: int
    lr_g: float
    lr_d: float
    seed: int
    train_ratio: float
    shuffle_split: bool
    cv_folds: int
    cv_fold: int
    num_workers: int
    device: str
    out_dir: str
    log_every: int
    z_train: str
    match_noisy_scale: bool
    zscore_using_reference: bool
    resample_mode: str
    loss: str
    loss_mse_weight: float
    loss_l1_weight: float
    loss_stft_weight: float
    lambda_sup: float
    subway_dual_channels: bool
    strict_all_bands: bool


def pick_device(preferred: str) -> torch.device:
    if preferred.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _our_data_config(cfg: Gan5Cfg, *, train: bool) -> OurDataConfig:
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


def _train_z(batch_size: int, device: torch.device, dtype: torch.dtype, mode: str) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros(
            batch_size,
            UNET_LATENT_CHANNELS,
            UNET_LATENT_LENGTH,
            device=device,
            dtype=dtype,
        )
    return sample_latent(batch_size, device=device, dtype=dtype)


def _mix_weights(cfg: Gan5Cfg) -> tuple[float, float, float]:
    """Normalized MSE / L1 / STFT mix proportions (consistent with ``mse_time_frequency_loss``)."""
    if cfg.loss == "2unet":
        return 1.0, 0.0, 0.0
    return resolve_mix_weights(cfg.loss_mse_weight, cfg.loss_l1_weight, cfg.loss_stft_weight)


def _new_sup_accum() -> dict[str, float]:
    return {"sup": 0.0, "mse": 0.0, "l1": 0.0, "stft": 0.0, "n": 0.0}


def _accum_sup_batch(acc: dict[str, float], cfg: Gan5Cfg, pred: torch.Tensor, reference: torch.Tensor, mask) -> None:
    loss_sup, l_mse, l_l1, l_stft = _supervised_loss(cfg, pred, reference, mask)
    acc["sup"] += float(loss_sup.item())
    wm, wl, ws = _mix_weights(cfg)
    if cfg.loss == "2unet" and l_mse is not None:
        acc["mse"] += float(l_mse.item())
    elif cfg.loss == "mse_time":
        if wm > 0.0 and l_mse is not None:
            acc["mse"] += float(l_mse.item())
        if wl > 0.0 and l_l1 is not None:
            acc["l1"] += float(l_l1.item())
        if ws > 0.0 and l_stft is not None:
            acc["stft"] += float(l_stft.item())
    elif cfg.loss in ("time", "default"):
        if wl > 0.0 and l_l1 is not None:
            acc["l1"] += float(l_l1.item())
        if ws > 0.0 and l_stft is not None:
            acc["stft"] += float(l_stft.item())
    acc["n"] += 1.0


def _mean_sup_metrics(acc: dict[str, float]) -> dict[str, float]:
    n = max(1.0, float(acc["n"]))
    return {
        "loss_sup": float(acc["sup"] / n),
        "l_mse": float(acc["mse"] / n),
        "l_l1": float(acc["l1"] / n),
        "l_stft": float(acc["stft"] / n),
    }


def _mix_score(cfg: Gan5Cfg, metrics: dict[str, float]) -> float:
    """Same formula as training ``loss_sup``: fixed-scale alignment then weighted mix (lower is better)."""
    if cfg.loss == "2unet":
        return float(metrics["l_mse"])
    return mix_score_from_raw(
        float(metrics["l_mse"]),
        float(metrics["l_l1"]),
        float(metrics["l_stft"]),
        w_mse=cfg.loss_mse_weight,
        w_l1=cfg.loss_l1_weight,
        w_stft=cfg.loss_stft_weight,
    )


def _format_metric_log(cfg: Gan5Cfg, metrics: dict[str, float], *, split: str) -> str:
    """``split`` is ``val`` or ``train``; primary metric ``{split}_score`` (fixed-scale mix, lower is better)."""
    wm, wl, ws = _mix_weights(cfg)
    score = _mix_score(cfg, metrics)
    parts = [f"{split}_score={score:.6f}"]
    if cfg.loss == "2unet":
        parts.append(f"l_mse={metrics['l_mse']:.6f}")
    elif cfg.loss == "mse_time":
        if wm > 0.0:
            parts.append(f"l_mse={metrics['l_mse']:.6f}")
        if wl > 0.0:
            parts.append(f"l_l1={metrics['l_l1']:.6f}")
        if ws > 0.0:
            parts.append(f"l_stft={metrics['l_stft']:.6f}")
    else:
        if wl > 0.0:
            parts.append(f"l_l1={metrics['l_l1']:.6f}")
        if ws > 0.0:
            parts.append(f"l_stft={metrics['l_stft']:.6f}")
    return " ".join(parts)


@torch.no_grad()
def eval_val_supervised_metrics(
    G: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Gan5Cfg,
) -> dict[str, float]:
    """Mean supervised loss on validation set (same definition as training ``_supervised_loss``; inference z=0)."""
    G.eval()
    acc = _new_sup_accum()
    for batch in loader:
        reference = batch["reference"].to(device)
        noisy = batch["noisy"].to(device)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device)
        bsz = noisy.size(0)
        z = torch.zeros(
            bsz,
            UNET_LATENT_CHANNELS,
            UNET_LATENT_LENGTH,
            device=device,
            dtype=noisy.dtype,
        )
        pred = G(noisy, z)
        _accum_sup_batch(acc, cfg, pred, reference, mask)
    G.train()
    return _mean_sup_metrics(acc)


def _supervised_loss(
    cfg: Gan5Cfg,
    pred: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if cfg.loss == "2unet":
        t = F.mse_loss(pred, reference)
        return t, t, None, None
    if cfg.loss in ("time", "default"):
        tot, l1, stft = supervised_unet_loss(
            pred,
            reference,
            mask,
            loss_l1_weight=cfg.loss_l1_weight,
            loss_stft_weight=cfg.loss_stft_weight,
        )
        return tot, None, l1, stft
    tot, mse, l1, stft = mse_time_frequency_loss(
        pred,
        reference,
        mask,
        loss_mse_weight=cfg.loss_mse_weight,
        loss_l1_weight=cfg.loss_l1_weight,
        loss_stft_weight=cfg.loss_stft_weight,
    )
    return tot, mse, l1, stft


def train_one_epoch(
    *,
    cfg: Gan5Cfg,
    device: torch.device,
    loader: DataLoader,
    G: torch.nn.Module,
    D: torch.nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    epoch: int,
) -> dict[str, float]:
    G.train()
    D.train()
    last_d = last_g = last_sup = float("nan")
    tr_acc = _new_sup_accum()
    for batch in loader:
        reference = batch["reference"].to(device)
        noisy = batch["noisy"].to(device)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device)
        B = noisy.size(0)
        if not torch.isfinite(reference).all() or not torch.isfinite(noisy).all():
            continue

        # --- D ---
        for p in D.parameters():
            p.requires_grad_(True)
        with torch.no_grad():
            z = _train_z(B, device, noisy.dtype, cfg.z_train)
            fake = G(noisy, z)

        pred_real, _ = D(reference, noisy)
        pred_fake, _ = D(fake, noisy)
        loss_d = 0.5 * (
            F.mse_loss(pred_real, torch.ones_like(pred_real))
            + F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
        )
        if not torch.isfinite(loss_d):
            continue
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        clip_grad_norm_(D.parameters(), max_norm=1.0)
        opt_d.step()

        # --- G ---
        for p in D.parameters():
            p.requires_grad_(False)
        z = _train_z(B, device, noisy.dtype, cfg.z_train)
        fake = G(noisy, z)
        pred_fake_g, _ = D(fake, noisy)
        loss_adv = F.mse_loss(pred_fake_g, torch.ones_like(pred_fake_g))
        loss_sup, _a, _b, _c = _supervised_loss(cfg, fake, reference, mask)
        loss_g = loss_adv + float(cfg.lambda_sup) * loss_sup
        if not torch.isfinite(loss_g):
            continue
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        clip_grad_norm_(G.parameters(), max_norm=1.0)
        opt_g.step()

        last_d = float(loss_d.item())
        last_g = float(loss_g.item())
        last_sup = float(loss_sup.item())
        _accum_sup_batch(tr_acc, cfg, fake, reference, mask)

    out: dict[str, float] = {"loss_d": last_d, "loss_g": last_g, "loss_sup": last_sup}
    out.update(_mean_sup_metrics(tr_acc))
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="TraMagNet: TraMagNet UNet + TraMagNet Discriminator + TraMagNet supervised loss (our_data only)")
    add_common_train_arguments(p)
    p.add_argument("--lr-g", type=float, default=2e-5, dest="lr_g")
    p.add_argument("--lr-d", type=float, default=2e-5, dest="lr_d")
    p.add_argument("--log-every", type=int, default=50, dest="log_every")
    p.add_argument(
        "--z-train",
        type=str,
        default="random",
        choices=("random", "zero"),
        dest="z_train",
        help="Training z: random | zero (default random).",
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
        "--resample-mode",
        type=str,
        default="resample_linear",
        choices=("pad_edge", "pad_zero", "resample_linear"),
        dest="resample_mode",
    )
    p.add_argument(
        "--loss",
        type=str,
        default="mse_time",
        choices=("mse_time", "time", "default", "2unet"),
    )
    p.add_argument(
        "--loss-mse-weight",
        type=float,
        default=0.0,
        dest="loss_mse_weight",
        help="Supervised mix: MSE proportion (default 0; normalized with L1/STFT to sum 1).",
    )
    p.add_argument(
        "--loss-l1-weight",
        type=float,
        default=0.4,
        dest="loss_l1_weight",
        help="Supervised mix: L1 proportion (default 0.4).",
    )
    p.add_argument(
        "--loss-stft-weight",
        type=float,
        default=0.6,
        dest="loss_stft_weight",
        help="Supervised mix: STFT proportion (default 0.6).",
    )
    p.add_argument(
        "--lambda-sup",
        type=float,
        default=1.0,
        dest="lambda_sup",
        help="Total supervised weight (multiplies mse_time / time / 2unet total; added to loss_adv).",
    )
    p.add_argument(
        "--subway-dual-channels",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="subway_dual_channels",
        help="data3: merge two value columns in +subway.txt into dual channels (default on, consistent with TraMagNet / viz).",
    )
    p.add_argument(
        "--strict-all-bands",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="strict_all_bands",
        help="When band=all, drop samples missing a frequency band (default on).",
    )
    args = p.parse_args(argv)

    from data_common.cv_train import collect_sorted_sample_ids, maybe_write_split_manifest, run_per_cv_fold
    from data_common.train_data_factory import training_uses_pooled_data

    if training_uses_pooled_data(args):
        from data_common.pooled_data_split import parse_data_roots_arg

        entries = parse_data_roots_arg(str(args.data_roots), repo=_REPO)
        data_root = str(entries[0][1])
    else:
        data_root = resolve_dataset_root(args.data_root, repo=_REPO)
        if Path(data_root).resolve() != Path(args.data_root).expanduser().resolve():
            print(f"[data-root] resolved {args.data_root!r} -> {data_root}", flush=True)
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

    base_out = Path(resolve_train_out_dir(args, data_root))
    base_out.mkdir(parents=True, exist_ok=True)
    print(f"[TraMagNet] checkpoint directory: {base_out.resolve()}", flush=True)

    run_per_cv_fold(args, base_out_dir=base_out, train_fn=_train_one_fold_impl)


def _train_one_fold_impl(args: argparse.Namespace, fold: int, out_path: Path) -> None:
    from data_common.train_data_factory import training_uses_pooled_data

    if training_uses_pooled_data(args):
        from data_common.pooled_data_split import parse_data_roots_arg

        entries = parse_data_roots_arg(str(args.data_roots), repo=_REPO)
        data_root = str(entries[0][1])
    else:
        data_root = resolve_dataset_root(args.data_root, repo=_REPO)
    out_dir = str(out_path.resolve())

    loss_l1_w = float(args.loss_l1_weight)
    loss_stft_w = float(args.loss_stft_weight)
    if args.loss in ("time", "default") and loss_l1_w == 0.0 and loss_stft_w == 0.0:
        loss_l1_w = 1.0

    cfg = Gan5Cfg(
        root=data_root,
        reference_subdir=args.reference_subdir,
        noisy_subdir=args.noisy_subdir,
        band=args.band,
        segment_length=int(args.segment_length),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr_g=float(args.lr_g),
        lr_d=float(args.lr_d),
        seed=int(args.seed),
        train_ratio=float(args.train_ratio),
        shuffle_split=bool(args.shuffle_split),
        cv_folds=int(args.cv_folds),
        cv_fold=int(fold),
        num_workers=int(args.num_workers),
        device=str(args.device),
        out_dir=str(out_dir),
        log_every=int(args.log_every),
        z_train=str(args.z_train),
        match_noisy_scale=bool(args.match_noisy_scale),
        zscore_using_reference=bool(args.zscore_using_reference),
        resample_mode=str(args.resample_mode),
        loss=str(args.loss),
        loss_mse_weight=float(args.loss_mse_weight),
        loss_l1_weight=loss_l1_w,
        loss_stft_weight=loss_stft_w,
        lambda_sup=float(args.lambda_sup),
        subway_dual_channels=bool(args.subway_dual_channels),
        strict_all_bands=bool(args.strict_all_bands),
    )

    if cfg.loss == "mse_time":
        from models.unet_loss import resolve_mix_weights

        try:
            resolve_mix_weights(cfg.loss_mse_weight, cfg.loss_l1_weight, cfg.loss_stft_weight)
        except ValueError as e:
            raise SystemExit(f"mse_time: {e}") from e

    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    out_path = Path(cfg.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "config.txt").write_text(str(asdict(cfg)), encoding="utf-8")

    pin = device.type == "cuda"
    from data_common.train_data_factory import make_supervised_dataset, training_uses_pooled_data

    pooled = training_uses_pooled_data(args)
    ds_tr = make_supervised_dataset(
        args, cfg, repo=_REPO, train=True, our_data_config_cls=OurDataConfig, our_data_dataset_cls=OurDataDataset
    )
    ds_va = make_supervised_dataset(
        args, cfg, repo=_REPO, train=False, our_data_config_cls=OurDataConfig, our_data_dataset_cls=OurDataDataset
    )
    train_loader = DataLoader(
        ds_tr,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        ds_va,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=False,
    )

    from data_common.cv_train import cv_fold_log_prefix, format_cv_fold_info

    fold_info = format_cv_fold_info(fold, cfg.cv_folds)
    fp = cv_fold_log_prefix(fold, cfg.cv_folds)
    cv_note = f" {fold_info}" if fold_info else ""

    G = UNet().to(device)
    D = Discriminator().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(0.5, 0.9))

    start_epoch = 1
    best_val_score = float("inf")
    last_pt = out_path / "last.pt"
    if bool(args.auto_resume) and last_pt.is_file():
        try:
            try:
                ckpt = torch.load(last_pt, map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load(last_pt, map_location=device)
        except (EOFError, OSError, RuntimeError, pickle.UnpicklingError, TypeError) as e:
            print(f"{fp}[resume] cannot read {last_pt}, training from scratch: {type(e).__name__}: {e}", flush=True)
            ckpt = None
        if ckpt is not None and isinstance(ckpt, dict):
            if "generator" in ckpt:
                G.load_state_dict(ckpt["generator"], strict=True)
            if "discriminator" in ckpt:
                D.load_state_dict(ckpt["discriminator"], strict=True)
            if "opt_g" in ckpt:
                opt_g.load_state_dict(ckpt["opt_g"])
            if "opt_d" in ckpt:
                opt_d.load_state_dict(ckpt["opt_d"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            if "best_val_score" in ckpt:
                best_val_score = float(ckpt["best_val_score"])
            else:
                # Legacy checkpoint: normalized loss_sup≈1 is not comparable; re-select best by val_score after resume
                best_val_score = float("inf")
            print(f"{fp}[resume] {last_pt} epoch->{start_epoch} best_val_score={best_val_score:.6f}", flush=True)
        elif ckpt is not None:
            print(f"{fp}[resume] {last_pt} has unexpected structure, training from scratch.", flush=True)
    elif not bool(args.auto_resume):
        print(f"{fp}[resume] --no-auto-resume: not loading from last.pt.", flush=True)

    print(
        f"[TraMagNet]{cv_note} G=TraMagNet.UNet D=TraMagNet.Discriminator loss={cfg.loss} "
        f"sup_w={cfg.lambda_sup} z_train={cfg.z_train} "
        f"subway_dual={cfg.subway_dual_channels} strict_all_bands={cfg.strict_all_bands}",
        flush=True,
    )
    print(f"{fp}[data] train={len(ds_tr)} val={len(ds_va)}", flush=True)
    wm0, wl0, ws0 = _mix_weights(cfg)
    print(
        f"{fp}[metric] loss_sup / val_score = Σ weight×(l_* / feature_scale); "
        f"scales mse={LOSS_TERM_SCALE_MSE} l1={LOSS_TERM_SCALE_L1} stft={LOSS_TERM_SCALE_STFT}; "
        f"mix weights {wm0:.2f}/{wl0:.2f}/{ws0:.2f}, lower is better.",
        flush=True,
    )

    for epoch in range(start_epoch, cfg.epochs + 1):
        m = train_one_epoch(
            cfg=cfg,
            device=device,
            loader=train_loader,
            G=G,
            D=D,
            opt_g=opt_g,
            opt_d=opt_d,
            epoch=epoch,
        )
        if epoch % max(1, cfg.log_every) == 0 or epoch == cfg.epochs:
            print(
                f"{fp}epoch={epoch} train lossD={m['loss_d']:.6f} lossG={m['loss_g']:.6f} "
                + _format_metric_log(cfg, m, split="train"),
                flush=True,
            )

        val_m = eval_val_supervised_metrics(G, val_loader, device, cfg)
        val_score = _mix_score(cfg, val_m)
        print(f"{fp}epoch={epoch} val " + _format_metric_log(cfg, val_m, split="val"), flush=True)

        if val_score < best_val_score:
            best_val_score = val_score
            save_torch_checkpoint(
                out_path / "best.pt",
                {
                    "generator": G.state_dict(),
                    "discriminator": D.state_dict(),
                    "epoch": epoch,
                    "val_score": val_score,
                    "best_val_score": best_val_score,
                    "val_metrics": val_m,
                    "extra": {
                        "loss": cfg.loss,
                        "loss_mse_weight": cfg.loss_mse_weight,
                        "loss_l1_weight": cfg.loss_l1_weight,
                        "loss_stft_weight": cfg.loss_stft_weight,
                        "lambda_sup": cfg.lambda_sup,
                    },
                },
            )
            print(
                f"{fp}  -> saved best.pt (val_score={val_score:.6f})",
                flush=True,
            )

        save_torch_checkpoint(
            last_pt,
            {
                "generator": G.state_dict(),
                "discriminator": D.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "epoch": epoch,
                "val_score": val_score,
                "best_val_score": best_val_score,
                "val_metrics": val_m,
                "extra": {
                    "loss": cfg.loss,
                    "loss_mse_weight": cfg.loss_mse_weight,
                    "loss_l1_weight": cfg.loss_l1_weight,
                    "loss_stft_weight": cfg.loss_stft_weight,
                    "lambda_sup": cfg.lambda_sup,
                },
            },
        )

    print(f"{fp}done. outputs at: {out_path.resolve()}", flush=True)