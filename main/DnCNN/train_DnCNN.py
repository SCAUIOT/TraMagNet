from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.our_data_dataset import OurDataConfig, OurDataDataset
from models.dncnn_1d import (
    DnCNN1D,
    DnCNN1DConfig,
    dncnn_config_from_argparse,
    masked_l1,
    masked_loss,
    masked_mse,
    masked_temporal_mse,
)


def _is_legacy_plain_dncnn_state_dict(sd: dict) -> bool:
    """Legacy single-stack DnCNN uses net.* keys; enhanced structure uses head./middle./res_blocks. etc."""
    return any(k.startswith("net.") for k in sd.keys())


def pick_device(preferred: str) -> torch.device:
    if preferred.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _term_width() -> int:
    try:
        return max(48, shutil.get_terminal_size(fallback=(100, 24)).columns)
    except Exception:
        return 100


def _effective_log_style(style: str) -> str:
    if style == "dynamic" and not sys.stdout.isatty():
        return "compact"
    if style == "rich" and not sys.stdout.isatty():
        return "full"
    return style


def _rich_available() -> bool:
    try:
        import rich  # noqa: F401

        return True
    except ImportError:
        return False


def _start_rich_live(*, fold_info: str = ""):
    """Rich Live in-place refresh; do not call when rich is not installed."""
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    console = Console(width=min(120, _term_width()))
    sub = "[dim]Same-screen update · not line-by-line append[/dim]"
    if fold_info:
        sub = f"[dim]{fold_info} · same-screen update · not line-by-line append[/dim]"
    init = Panel(
        Text("Initializing…", justify="center"),
        title="[bold cyan]DnCNN training live[/bold cyan]",
        subtitle=sub,
        border_style="cyan",
    )
    live = Live(init, console=console, refresh_per_second=12, transient=False)
    live.start()
    return live


def _rich_make_panel(
    *,
    args: argparse.Namespace,
    ep: int,
    epochs: int,
    elapsed_s: float,
    train_loss: float,
    val_loss: float,
    oracle_id: float,
    gain_pct: float,
    best_display: float,
    ls_log: float,
    lzh: str,
    diag: tuple[float, float, float, float, float, float, str] | None,
    best_history: list[float],
    mse_nc0: float | None,
    fold_info: str = "",
):
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    tr = train_loss * ls_log
    va = val_loss * ls_log
    orb = oracle_id * ls_log
    bv = best_display * ls_log

    header = Text()
    header.append("DnCNN · live refresh (", style="dim")
    header.append("same screen, not line-by-line", style="bold")
    header.append(")", style="dim")

    line1 = Text()
    line1.append("Epochs completed: ", style="dim")
    line1.append(f"{ep}", style="bold cyan")
    line1.append(f" / {epochs}", style="cyan")
    line1.append("  ·  ", style="dim")
    line1.append("Total elapsed: ", style="dim")
    line1.append(f"{elapsed_s:.1f}s", style="green")
    if fold_info:
        line1.append("  ·  ", style="dim")
        line1.append(fold_info, style="bold yellow")

    line2 = Text()
    line2.append("Data root: ", style="dim")
    line2.append(f"{args.data_root}", style="white")
    if mse_nc0 is not None:
        line2.append("  ·  ", style="dim")
        line2.append("First-batch MSE(noisy,reference): ", style="dim")
        line2.append(f"{mse_nc0:.6g}", style="magenta")

    tbl = Table(show_header=True, header_style="bold", show_edge=True, pad_edge=False)
    tbl.add_column("Train loss", justify="right")
    tbl.add_column("Val loss", justify="right")
    tbl.add_column("Identity baseline", justify="right", header_style="dim")
    tbl.add_column("Gain vs identity", justify="right")
    tbl.add_column("Best val so far", justify="right", style="bold green")
    tbl.add_column("Loss fn", justify="left")
    tbl.add_row(
        f"{tr:.6f}",
        f"{va:.6f}",
        f"{orb:.6f}",
        f"{gain_pct:.2f}%",
        f"{bv:.6f}",
        lzh,
    )

    steps = Text()
    if best_history:
        tail = best_history[-14:]
        arrow = " → ".join(f"{x:.6g}" for x in tail)
        steps.append("Best-val «step» history (recorded only on new lows, last ~ ", style="dim")
        steps.append(str(len(tail)), style="cyan")
        steps.append("):\n", style="dim")
        steps.append(arrow, style="yellow")
    else:
        steps.append("No new best recorded yet (or first epoch)", style="dim")

    diag_block = Text()
    if diag:
        mse_a, mse_b, mse_c, ac, bc, ab, tag = diag
        diag_block.append("\n[Diagnostics · MSE, independent of training --loss]\n", style="bold")
        diag_block.append(
            f"  MSE(denoised,noisy)={mse_a * ls_log:.6f}  "
            f"MSE(denoised,reference)={mse_b * ls_log:.6f}  "
            f"MSE(noisy,reference)={mse_c * ls_log:.6f}\n",
            style="white",
        )
        diag_block.append(
            f"  Ratio A/C={ac:.4f}  B/C={bc:.4f}  A/B={ab:.4f}",
            style="cyan",
        )
        if tag:
            diag_block.append(tag, style="red")
        diag_block.append(
            "\n  [dim]Identity copy-noisy ≈ A/C→0, B/C→1; ideal denoise ≈ A/C→1, B/C→0.[/dim]",
            style="dim",
        )
    else:
        diag_block.append("\n[dim](identity diagnostics disabled)[/dim]", style="dim")

    foot = Text(
        "(Exploration yields many non-optimal epochs; steps need not decrease every time; checkpoint val_loss is unscaled)",
        style="dim",
    )

    body = Group(
        header,
        line1,
        line2,
        "",
        tbl,
        "",
        steps,
        diag_block,
        foot,
    )

    title = "[bold]DnCNN training live[/bold]"
    if fold_info:
        title = f"[bold]DnCNN training live[/bold] [yellow]{fold_info}[/yellow]"
    return Panel(
        body,
        title=title,
        border_style="cyan",
        padding=(1, 2),
    )


def _enable_windows_ansi() -> None:
    """Try to enable ANSI escapes on Windows console (needed for multi-line in-place refresh)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        h = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
            kernel32.SetConsoleMode(h, mode.value | 0x0004)
    except Exception:
        pass


def _erase_terminal_lines(n: int) -> None:
    """Move cursor up n lines and clear each (for multi-line in-place refresh)."""
    if n <= 0:
        return
    for _ in range(n):
        sys.stdout.write("\033[1A\033[2K\r")
    sys.stdout.flush()


def _build_full_epoch_lines(
    *,
    ep: int,
    epochs: int,
    tr: float,
    va: float,
    orb: float,
    gain_pct: float,
    bv: float,
    lzh: str,
    ls_log: float,
    diag: tuple[float, float, float, float, float, float, str] | None,
    fold_info: str = "",
) -> list[str]:
    sep = "=" * min(72, _term_width())
    fold_line = f"  {fold_info}" if fold_info else ""
    lines: list[str] = [
        sep,
        f"Epoch {ep} / {epochs}{fold_line}",
    ]
    lines.extend(
        [
            f"  Train loss: {tr:.6f}",
            f"  Val loss: {va:.6f}",
            f"  Identity baseline (val error if output equals noisy): {orb:.6f}",
            f"  Gain vs identity (percent val-loss reduction vs baseline): {gain_pct:.2f}%",
            f"  Best val loss so far (through this epoch): {bv:.6f}",
            f"  Loss function: {lzh}",
        ]
    )
    if diag:
        mse_a, mse_b, mse_c, ac, bc, ab, tag = diag
        lines.extend(
            [
                "  —— Diagnostics (fixed MSE, independent of training --loss) ——",
                f"    MSE(denoised, noisy) = {mse_a * ls_log:.6f}",
                f"    MSE(denoised, reference) = {mse_b * ls_log:.6f}",
                f"    MSE(noisy, reference) = {mse_c * ls_log:.6f}",
                "    Dimensionless ratios: A/C = MSE(denoised,noisy)/MSE(noisy,reference), "
                "B/C = MSE(denoised,reference)/MSE(noisy,reference), A/B = MSE(denoised,noisy)/MSE(denoised,reference)",
                f"    A/C = {ac:.4f}    B/C = {bc:.4f}    A/B = {ab:.4f}{tag}",
                "    Reference: identity copy-noisy ≈ A/C→0, B/C→1; ideal denoise ≈ A/C→1, B/C→0.",
            ]
        )
    return lines


def _pad_line(s: str, width: int) -> str:
    if len(s) >= width:
        return s[: width - 1] + "…"
    return s + " " * (width - len(s))


def _loss_name_zh(loss_name: str) -> str:
    m = {"mse": "MSE", "l1": "L1 (MAE)", "huber": "Smooth L1 (Huber)"}
    return m.get(loss_name.lower().strip(), loss_name)


def _print_epoch_log(
    *,
    style: str,
    ep: int,
    epochs: int,
    train_loss: float,
    val_loss: float,
    oracle_id: float,
    gain_pct: float,
    best_val: float,
    loss_name: str,
    ls_log: float,
    diag: tuple[float, float, float, float, float, float, str] | None,
    prev_full_lines: int = 0,
    live_terminal: bool = True,
    fold_info: str = "",
) -> int:
    """diag: (mse_a, mse_b, mse_c, ac, bc, ab, tag) or None. Returns line count for full mode; 1 for other modes."""
    tr = train_loss * ls_log
    va = val_loss * ls_log
    orb = oracle_id * ls_log
    bv = best_val * ls_log
    lzh = _loss_name_zh(loss_name)

    if style == "dynamic":
        if diag:
            _, _, _, ac, bc, ab, _ = diag
            extra = f" | A/C={ac:.2f} B/C={bc:.2f}"
        else:
            extra = ""
        fold_p = f"[{fold_info}] " if fold_info else ""
        line = (
            f"{fold_p}epoch {ep:04d}/{epochs} "
            f"train={tr:.4f} val={va:.4f} identity={orb:.4f} "
            f"gain={gain_pct:.1f}% best_val={bv:.4f} [{lzh}]{extra}"
        )
        sys.stdout.write("\r" + _pad_line(line, _term_width()))
        sys.stdout.flush()
        return 1
    if style == "compact":
        if diag:
            _, _, _, ac, bc, ab, _ = diag
            extra = f" | A/C={ac:.2f} B/C={bc:.2f} A/B={ab:.2f}"
        else:
            extra = ""
        fold_p = f"[{fold_info}] " if fold_info else ""
        print(
            f"{fold_p}epoch {ep:04d}/{epochs} | "
            f"train={tr:.4f} | val={va:.4f} | identity={orb:.4f} | "
            f"gain={gain_pct:.2f}% | best_val={bv:.4f} | loss={lzh}{extra}",
            flush=True,
        )
        return 1

    lines = _build_full_epoch_lines(
        ep=ep,
        epochs=epochs,
        tr=tr,
        va=va,
        orb=orb,
        gain_pct=gain_pct,
        bv=bv,
        lzh=lzh,
        ls_log=ls_log,
        diag=diag,
        fold_info=fold_info,
    )
    live = bool(live_terminal) and sys.stdout.isatty()
    if live and prev_full_lines > 0:
        _erase_terminal_lines(prev_full_lines)
    for line in lines:
        print(line, flush=True)
    return len(lines)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    loss_kind: str,
    huber_beta: float,
    l1_aux_weight: float,
) -> tuple[float, float]:
    """
    Returns (val metric, oracle identity baseline) using the same masked_loss as training.
    Oracle: assume denoised=noisy, i.e. noisy-vs-reference error (identity mapping baseline).
    """
    model.eval()
    total = 0.0
    total_identity = 0.0
    n = 0
    for batch in loader:
        noisy = batch["noisy"].to(device)  # (B,1,T)
        reference = batch["reference"].to(device)
        mask = batch.get("mask", None)
        if mask is not None:
            mask = mask.to(device)
        denoised = model(noisy)
        loss = masked_loss(
            denoised,
            reference,
            mask,
            kind=loss_kind,
            huber_beta=huber_beta,
            l1_aux_weight=l1_aux_weight,
        )
        id_loss = masked_loss(
            noisy,
            reference,
            mask,
            kind=loss_kind,
            huber_beta=huber_beta,
            l1_aux_weight=l1_aux_weight,
        )
        total += float(loss.item())
        total_identity += float(id_loss.item())
        n += 1
    denom = max(1, n)
    return total / denom, total_identity / denom


@torch.no_grad()
def evaluate_identity_mse(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Identity-mapping diagnostics (fixed MSE, independent of --loss):
    - A = MSE(denoised, noisy): if tiny vs B, output tracks noisy (copy-noisy behavior).
    - B = MSE(denoised, reference): should decrease with training.
    - C = MSE(noisy, reference): identity baseline (error if copying noisy).
    Returns batch-averaged (A, B, C).
    """
    model.eval()
    sa = sb = sc = 0.0
    n = 0
    for batch in loader:
        noisy = batch["noisy"].to(device)
        reference = batch["reference"].to(device)
        mask = batch.get("mask", None)
        if mask is not None:
            mask = mask.to(device)
        denoised = model(noisy)
        sa += float(masked_mse(denoised, noisy, mask).item())
        sb += float(masked_mse(denoised, reference, mask).item())
        sc += float(masked_mse(noisy, reference, mask).item())
        n += 1
    d = max(1, n)
    return sa / d, sb / d, sc / d


def _add_dncnn_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--resample-mode",
        type=str,
        default="resample_linear",
        choices=("pad_edge", "pad_zero", "resample_linear"),
        help="How to resize to segment-length (default matches UNet whole-segment interpolation).",
    )
    p.add_argument("--depth", type=int, default=18, help="Only with --legacy-plain: total conv layers (legacy).")
    p.add_argument("--features", type=int, default=64)
    p.add_argument(
        "--legacy-plain",
        action="store_true",
        help="Use original single-stack DnCNN (matches old checkpoints; default off: Conv×N + residual + attention).",
    )
    p.add_argument("--middle-depth", type=int, default=10, help="Enhanced structure: number of Conv+BN+ReLU blocks.")
    p.add_argument("--num-residual", type=int, default=5, dest="num_residual", help="Enhanced structure: number of residual blocks.")
    p.add_argument(
        "--use-attention",
        action="store_true",
        help="Enhanced structure: enable channel+temporal attention (off by default; gating clamped to [0.5,1]).",
    )
    p.add_argument("--no-attention", action="store_true", help="Mutually exclusive with --use-attention: force attention off.")
    p.add_argument("--attention-reduction", type=int, default=8, help="Channel attention reduction ratio.")
    p.add_argument(
        "--loss",
        type=str,
        default="l1",
        choices=("mse", "l1", "huber"),
        help="Train/val metric: l1 (MAE, default, spike-sensitive), mse (optionally with --l1-aux-weight), huber (SmoothL1).",
    )
    p.add_argument(
        "--l1-aux-weight",
        type=float,
        default=0.35,
        help="Only loss=mse: add coefficient × L1(pred noise error) for MAE emphasis; 0 disables. Default 0.35.",
    )
    p.add_argument(
        "--huber-beta",
        type=float,
        default=0.1,
        help="SmoothL1 beta when loss=huber (smaller is closer to L1).",
    )
    p.add_argument(
        "--loss-log-scale",
        type=float,
        default=1.0,
        dest="loss_log_scale",
        metavar="SCALE",
        help=(
            "Terminal display multiplier for train/val/oracle/diag; default 1 (same scale as checkpoint). "
            "Set 100 etc. if needed. Backprop and checkpoint val_loss stay unscaled."
        ),
    )
    p.add_argument(
        "--reference-l1-weight",
        type=float,
        default=0.0,
        dest="reference_l1_weight",
        help="Add coefficient × L1(denoised, reference) beyond noise supervision; 0 off. Try 0.05~0.1 to curb identity collapse.",
    )
    p.add_argument(
        "--grad-match-weight",
        type=float,
        default=0.0,
        dest="grad_match_weight",
        help="Add coefficient × MSE(first temporal diff denoised vs reference); 0 off. Try 0.05~0.1 to align peaks.",
    )
    p.add_argument(
        "--identity-diag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Each epoch print fixed MSE A/B/C and A/C, B/C (identity≈A/C→0,B/C→1; ideal denoise≈A/C→1,B/C→0).",
    )
    p.add_argument(
        "--match-noisy-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="match_noisy_scale",
        help=(
            "Affine noisy to reference segment mean/std (off by default to avoid test leakage). "
            "Enable with --match-noisy-scale."
        ),
    )
    p.add_argument(
        "--zscore-using-reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="zscore_using_reference",
        help=(
            "Standardize with reference mu,sigma (off by default). Prefer noisy_sample (see OurDataConfig). "
            "Enable with --zscore-using-reference."
        ),
    )
    p.add_argument(
        "--data-sanity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="At startup print first train batch MSE(noisy,reference) (same definition as diag C).",
    )
    p.add_argument(
        "--log-style",
        type=str,
        default="rich",
        choices=("rich", "dynamic", "compact", "full"),
        help=(
            "rich (default, pip install rich): sectioned table + Live refresh, Optuna-style; "
            "full: multi-line + ANSI in-place refresh; "
            "compact: one line per epoch; "
            "dynamic: single-line in-place refresh. "
            "Falls back to full if rich missing; rich→full when not a TTY."
        ),
    )
    p.add_argument(
        "--live-terminal",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="live_terminal",
        help=(
            "Only log-style=full: ANSI multi-line in-place refresh on TTY (default on). "
            "--no-live-terminal appends each epoch (scrollback friendly)."
        ),
    )


def train_one_fold(args: argparse.Namespace, fold: int, out_path: Path) -> None:
    from data_common.cv_train import format_cv_fold_info

    args.out_dir = str(out_path)
    fold_info = format_cv_fold_info(fold, int(args.cv_folds))

    device = pick_device(args.device)
    _enable_windows_ansi()
    torch.manual_seed(int(args.seed))

    lr = float(args.lr) if args.lr is not None else 1e-4

    train_ds = OurDataDataset(
        OurDataConfig(
            root=args.data_root,
            reference_subdir=args.reference_subdir,
            noisy_subdir=args.noisy_subdir,
            band=args.band,  # type: ignore[arg-type]
            segment_length=int(args.segment_length),
            train=True,
            train_ratio=float(args.train_ratio),
            seed=int(args.seed),
            shuffle_split=bool(args.shuffle_split),
            cv_folds=int(args.cv_folds),
            cv_fold=int(fold),
            resample_mode=args.resample_mode,  # type: ignore[arg-type]
            strict_all_bands=True,
            match_noisy_scale_to_reference=bool(args.match_noisy_scale),
            zscore_using_reference=bool(args.zscore_using_reference),
        )
    )
    val_ds = OurDataDataset(
        OurDataConfig(
            root=args.data_root,
            reference_subdir=args.reference_subdir,
            noisy_subdir=args.noisy_subdir,
            band=args.band,  # type: ignore[arg-type]
            segment_length=int(args.segment_length),
            train=False,
            train_ratio=float(args.train_ratio),
            seed=int(args.seed),
            shuffle_split=bool(args.shuffle_split),
            cv_folds=int(args.cv_folds),
            cv_fold=int(fold),
            resample_mode=args.resample_mode,  # type: ignore[arg-type]
            strict_all_bands=True,
            match_noisy_scale_to_reference=bool(args.match_noisy_scale),
            zscore_using_reference=bool(args.zscore_using_reference),
        )
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    mse_nc0: float | None = None
    if bool(args.data_sanity):
        with torch.no_grad():
            b0 = next(iter(train_loader))
            n0 = b0["noisy"]
            c0 = b0["reference"]
            m0 = b0.get("mask", None)
            mse_nc0 = float(masked_mse(n0, c0, m0).item())

    cfg: DnCNN1DConfig = dncnn_config_from_argparse(args)
    model = DnCNN1D(cfg).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    save_dir = Path(args.out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best.pt"
    last_path = save_dir / "last.pt"

    best_val = float("inf")
    start_ep = 1
    if args.auto_resume and last_path.is_file():
        ckpt = torch.load(last_path, map_location=device)
        sd: dict = ckpt["model"]
        try:
            model.load_state_dict(sd, strict=True)
        except RuntimeError as err:
            if _is_legacy_plain_dncnn_state_dict(sd) and not cfg.legacy_plain:
                print(
                    "[resume] checkpoint is legacy plain DnCNN (keys net.*), incompatible with enhanced structure; "
                    "auto-switched to legacy_plain for resume.",
                    flush=True,
                )
                print(
                    "[resume] To train enhanced structure from scratch, delete/move last.pt / best.pt or use --no-auto-resume.",
                    flush=True,
                )
                cfg = replace(cfg, legacy_plain=True)
                model = DnCNN1D(cfg).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=lr)
                model.load_state_dict(sd, strict=True)
            else:
                raise err
        if "opt" in ckpt:
            try:
                opt.load_state_dict(ckpt["opt"])
            except Exception:
                print(
                    "[resume] Optimizer state incompatible with current model; skipped loading opt (fresh Adam state)",
                    flush=True,
                )
        best_val = float(ckpt.get("best_val_loss", ckpt.get("val_loss", best_val)))
        start_ep = int(ckpt.get("epoch", 0)) + 1
        print(f"[resume] loaded {last_path} (epoch={start_ep - 1})", flush=True)
        if start_ep > int(args.epochs):
            print(
                f"[resume] Already at epoch={start_ep - 1}, not less than target epochs={int(args.epochs)}; nothing to do.",
                flush=True,
            )
            return

    ls_log = float(args.loss_log_scale)
    log_style = _effective_log_style(str(args.log_style))

    if log_style == "full":
        if mse_nc0 is not None:
            print(
                f"[data sanity] first batch MSE(noisy, reference) = {mse_nc0:.6f} | "
                f"noisy aligned to reference amplitude = {args.match_noisy_scale} | "
                f"z-score using reference segment = {args.zscore_using_reference}",
                flush=True,
            )
        print(
            "[task] Supervision: predict noise pred_noise ≈ (noisy-reference); "
            "validation: masked_loss(denoised, reference).",
            flush=True,
        )
        if float(args.reference_l1_weight) > 0 or float(args.grad_match_weight) > 0:
            print(
                f"[extra losses] reference L1 weight = {args.reference_l1_weight}, "
                f"first-diff match weight = {args.grad_match_weight}",
                flush=True,
            )
    elif log_style == "rich":
        if float(args.reference_l1_weight) > 0 or float(args.grad_match_weight) > 0:
            print(
                f"[train] extra losses: reference_l1={args.reference_l1_weight} grad_match={args.grad_match_weight}",
                flush=True,
            )
    else:
        parts = [
            "DnCNN",
            *( [fold_info] if fold_info else [] ),
            f"data_root={args.data_root}",
            f"samples train/val={len(train_ds)}/{len(val_ds)}",
            str(device),
            f"lr={lr}",
            f"loss={args.loss}",
        ]
        if mse_nc0 is not None:
            parts.append(f"MSE(noisy,reference) first_batch={mse_nc0:.4g}")
        parts.extend(
            [
                f"amp_align={args.match_noisy_scale}",
                f"zscore={args.zscore_using_reference}",
                f"out_dir={args.out_dir}",
            ]
        )
        if float(args.reference_l1_weight) > 0 or float(args.grad_match_weight) > 0:
            parts.append(f"extra_l1={args.reference_l1_weight} grad_match={args.grad_match_weight}")
        print(" | ".join(parts), flush=True)

    meta_path = save_dir / "run_config.txt"
    meta_path.write_text(
        "\n".join(
            [
                f"train_ds={len(train_ds)} val_ds={len(val_ds)} device={device}",
                f"OurDataConfig(train)={asdict(train_ds.cfg)}",
                f"DnCNN1DConfig={asdict(model.cfg)}",
                f"args={repr(args)}",
            ]
        ),
        encoding="utf-8",
    )

    prev_full_lines = 0
    best_history: list[float] = []
    t_train0 = time.perf_counter()
    rich_live = (
        _start_rich_live(fold_info=fold_info) if log_style == "rich" and _rich_available() else None
    )
    try:
        for ep in range(start_ep, int(args.epochs) + 1):
            model.train()
            running = 0.0
            n = 0
            for batch in train_loader:
                noisy = batch["noisy"].to(device)
                reference = batch["reference"].to(device)
                mask = batch.get("mask", None)
                if mask is not None:
                    mask = mask.to(device)
    
                # Explicit noise supervision; masked_loss(denoised, reference) is the same objective family (mse / l1 / huber)
                noise_tgt = noisy - reference
                pred_noise = model.predict_noise(noisy)
                denoised = noisy - pred_noise
                l1w = float(args.l1_aux_weight) if args.loss == "mse" else 0.0
                loss = masked_loss(
                    pred_noise,
                    noise_tgt,
                    mask,
                    kind=args.loss,
                    huber_beta=float(args.huber_beta),
                    l1_aux_weight=l1w,
                )
                clw = float(args.reference_l1_weight)
                if clw > 0:
                    loss = loss + clw * masked_l1(denoised, reference, mask)
                gmw = float(args.grad_match_weight)
                if gmw > 0:
                    loss = loss + gmw * masked_temporal_mse(denoised, reference, mask)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                running += float(loss.item())
                n += 1
    
            train_loss = running / max(1, n)
            val_loss, val_if_identity = evaluate(
                model,
                val_loader,
                device,
                loss_kind=args.loss,
                huber_beta=float(args.huber_beta),
                l1_aux_weight=float(args.l1_aux_weight) if args.loss == "mse" else 0.0,
            )
            gain = (val_if_identity - val_loss) / max(val_if_identity, 1e-12) * 100.0
            display_best = float(min(best_val, val_loss))
    
            diag: tuple[float, float, float, float, float, float, str] | None = None
            if bool(args.identity_diag):
                mse_a, mse_b, mse_c = evaluate_identity_mse(model, val_loader, device)
                den_c = max(mse_c, 1e-12)
                ac = mse_a / den_c
                bc = mse_b / den_c
                ab = mse_a / max(mse_b, 1e-12)
                tag = ""
                if log_style in ("full", "rich"):
                    if ac < 0.15 and bc > 0.85:
                        tag = " [warn: near identity noisy→output]"
                    elif bc < 0.35 and ac > 0.5:
                        tag = " [trend: moving away from noisy, toward reference]"
                diag = (mse_a, mse_b, mse_c, ac, bc, ab, tag)
    
            if val_loss < best_val:
                best_history.append(float(val_loss * ls_log))
                if len(best_history) > 18:
                    best_history[:] = best_history[-18:]
    
            if rich_live is not None:
                rich_live.update(
                    _rich_make_panel(
                        args=args,
                        ep=ep,
                        epochs=int(args.epochs),
                        elapsed_s=time.perf_counter() - t_train0,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        oracle_id=val_if_identity,
                        gain_pct=gain,
                        best_display=display_best,
                        ls_log=ls_log,
                        lzh=_loss_name_zh(str(args.loss)),
                        diag=diag,
                        best_history=best_history,
                        mse_nc0=mse_nc0,
                        fold_info=fold_info,
                    )
                )
            else:
                nlines = _print_epoch_log(
                    style=log_style,
                    ep=ep,
                    epochs=int(args.epochs),
                    train_loss=train_loss,
                    val_loss=val_loss,
                    oracle_id=val_if_identity,
                    gain_pct=gain,
                    best_val=display_best,
                    loss_name=str(args.loss),
                    ls_log=ls_log,
                    diag=diag,
                    prev_full_lines=prev_full_lines,
                    live_terminal=bool(args.live_terminal),
                    fold_info=fold_info,
                )
                if log_style == "full":
                    prev_full_lines = nlines
    
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
            payload = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "epoch": ep,
                "val_loss": val_loss,
                "val_oracle_identity_mse": val_if_identity,
                "train_loss": train_loss,
                "best_val_loss": best_val,
            }
            torch.save(payload, last_path)
            if improved:
                torch.save(payload, best_path)
    finally:
        if rich_live is not None:
            rich_live.stop()

    if log_style == "dynamic":
        sys.stdout.write("\n")
        sys.stdout.flush()
    elif log_style == "full" and bool(args.live_terminal) and sys.stdout.isatty() and prev_full_lines > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()


def main(argv: list[str] | None = None) -> None:
    _repo = Path(__file__).resolve().parents[1]
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
    from common_train_cli import add_common_train_arguments, default_train_out_dir
    from data_common.cv_train import collect_sorted_sample_ids, maybe_write_split_manifest, run_per_cv_fold

    p = argparse.ArgumentParser(description="Directory 3: 1D DnCNN, shared args + network/resample options.")
    add_common_train_arguments(p)
    _add_dncnn_arguments(p)
    args = p.parse_args(argv)
    if getattr(args, "log_style", None) == "rich" and not _rich_available():
        print("[train] rich not installed; using --log-style full (under directory 3: pip install -r requirements.txt)", flush=True)
        args.log_style = "full"
    if getattr(args, "no_attention", False) and getattr(args, "use_attention", False):
        raise SystemExit("Cannot specify both --use-attention and --no-attention")

    base_out = Path(args.out_dir if args.out_dir is not None else default_train_out_dir(args.data_root))
    base_out.mkdir(parents=True, exist_ok=True)
    from data_common.resolve_dataset_root import resolve_dataset_root

    args.data_root = resolve_dataset_root(args.data_root, repo=_repo)
    sids = collect_sorted_sample_ids(
        args.data_root,
        reference_subdir=args.reference_subdir,
        noisy_subdir=args.noisy_subdir,
        band=args.band,
    )
    maybe_write_split_manifest(
        manifest_path=base_out / "split_manifest.json",
        sids=sids,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
        shuffle_split=bool(args.shuffle_split),
        cv_folds=int(args.cv_folds),
        skip=bool(getattr(args, "no_cv_split_manifest", False)),
    )
    run_per_cv_fold(args, base_out_dir=base_out, train_fn=train_one_fold)


if __name__ == "__main__":
    main(None)

