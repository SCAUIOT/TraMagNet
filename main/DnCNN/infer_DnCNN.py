from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_common.viz_export import save_triplet_figure
from data.our_data_dataset import OurDataConfig, OurDataDataset
from models.dncnn_1d import DnCNN1D, dncnn_config_from_argparse


def pick_device(preferred: str) -> torch.device:
    if preferred.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _data_tag_from_root(data_root: str) -> str:
    s = (data_root or ".").strip()
    if s in (".", ""):
        return "data"
    try:
        return Path(s).expanduser().resolve().name
    except OSError:
        return Path(s).name or "data"


def _resolve_data_root(path_str: str) -> str:
    s = (path_str or ".").strip()
    if s in (".", ""):
        return str(Path(".").resolve())
    p = Path(s).expanduser()
    if p.is_absolute():
        return str(p)
    repo_p = (_REPO / p).resolve()
    if repo_p.exists():
        return str(repo_p)
    return str(p)


def save_denoised_txt(out_path: Path, denoised: torch.Tensor) -> None:
    y = denoised.squeeze().detach().cpu().float().numpy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for i, v in enumerate(y.tolist()):
            f.write(f"{i}\t{i}\t{v}\n")


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    payload = torch.load(ckpt_path, map_location=device)
    if isinstance(payload, dict) and "model" in payload:
        model.load_state_dict(payload["model"], strict=True)
        return
    if isinstance(payload, dict):
        model.load_state_dict(payload, strict=True)
        return
    raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description="Run DnCNN1D inference and optionally export a plot.")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--band", type=str, default="low", choices=("low", "middle", "high", "all"))
    p.add_argument("--segment-length", type=int, default=512)
    p.add_argument("--resample-mode", type=str, default="pad_edge", choices=("pad_edge", "pad_zero", "resample_linear"))
    p.add_argument("--split", type=str, default="test", choices=("test", "train"))
    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--depth", type=int, default=18)
    p.add_argument("--features", type=int, default=64)
    p.add_argument("--legacy-plain", action="store_true", help="Match the legacy single-stack DnCNN used during training.")
    p.add_argument("--middle-depth", type=int, default=10)
    p.add_argument("--num-residual", type=int, default=5, dest="num_residual")
    p.add_argument("--use-attention", action="store_true", help="Align with a model trained with attention enabled.")
    p.add_argument("--no-attention", action="store_true", help="Force attention off.")
    p.add_argument("--attention-reduction", type=int, default=8)
    p.add_argument(
        "--save-plot",
        type=str,
        default="",
        help="If set to a file path, save the plot there; otherwise save to --output-image-dir/{key}.png",
    )
    p.add_argument("--output-image-dir", type=str, default=None, help="Plot output directory; default output/<data-name>/image")
    p.add_argument("--result-dir", type=str, default=None, help="Denoised numeric .txt output directory; default output/<data-name>/result")
    p.add_argument(
        "--data-root",
        type=str,
        default=".",
        help="Data root (contains reference_signal / noise_signal); default is the current directory.",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.8, dest="train_ratio")
    p.add_argument("--shuffle-split", action="store_true")
    p.add_argument(
        "--match-noisy-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="match_noisy_scale",
        help="Affine-align noisy to reference (off by default to avoid test leakage).",
    )
    p.add_argument(
        "--zscore-using-reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="zscore_using_reference",
        help="Z-score using reference statistics (off by default; noisy_sample normalization is recommended).",
    )
    args = p.parse_args()
    if args.use_attention and args.no_attention:
        raise SystemExit("Cannot specify both --use-attention and --no-attention")

    device = pick_device(args.device)

    data_root = _resolve_data_root(args.data_root)
    tag = _data_tag_from_root(data_root)
    output_image_dir = Path(args.output_image_dir) if args.output_image_dir else (Path("output") / tag / "image")
    result_dir = Path(args.result_dir) if args.result_dir else (Path("output") / tag / "result")

    ds = OurDataDataset(
        OurDataConfig(
            root=data_root,
            band=args.band,  # type: ignore[arg-type]
            segment_length=int(args.segment_length),
            train=(args.split.lower() == "train"),
            train_ratio=float(args.train_ratio),
            seed=int(args.seed),
            shuffle_split=bool(args.shuffle_split),
            resample_mode=args.resample_mode,  # type: ignore[arg-type]
            strict_all_bands=True,
            match_noisy_scale_to_reference=bool(args.match_noisy_scale),
            zscore_using_reference=bool(args.zscore_using_reference),
        )
    )
    i = int(args.idx) % len(ds)
    item = ds[i]
    noisy = item["noisy"].unsqueeze(0).to(device)  # (1,1,T)
    reference = item["reference"].unsqueeze(0).to(device)
    key = str(item.get("key", f"idx_{i}"))

    model = DnCNN1D(dncnn_config_from_argparse(args)).to(device)
    model.eval()
    load_checkpoint(model, Path(args.ckpt), device)

    denoised = model(noisy)
    print(f"key={key} idx={i} noisy_shape={tuple(noisy.shape)}", flush=True)

    n = noisy.squeeze().detach().cpu().numpy()
    d = denoised.squeeze().detach().cpu().numpy()
    c = reference.squeeze().detach().cpu().numpy()
    if args.save_plot:
        out = Path(args.save_plot)
    else:
        out = output_image_dir / f"{key}.png"
    save_triplet_figure(
        out,
        c,
        n,
        d,
        title=key,
        figsize=(12.0, 4.0),
        dpi=160,
        xlabel="Sample",
        ylabel="Amplitude (preprocessed)",
        noisy_label="noisy",
        denoised_label="denoised",
        reference_label="reference",
    )
    print(f"saved_plot={out.resolve()}", flush=True)

    txt_out = result_dir / f"{key}.txt"
    save_denoised_txt(txt_out, denoised)
    print(f"saved_denoised_txt={txt_out.resolve()}", flush=True)


if __name__ == "__main__":
    main()

