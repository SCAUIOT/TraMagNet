"""Reproducibility helpers: seeds, experiment metadata."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def dataloader_worker_init_fn(base_seed: int):
    def _init(worker_id: int) -> None:
        s = int(base_seed) + int(worker_id)
        random.seed(s)
        np.random.seed(s)
        try:
            import torch

            torch.manual_seed(s)
        except ImportError:
            pass

    return _init


def torch_generator(seed: int):
    import torch

    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def git_commit_hash(repo: Path | None = None) -> str:
    root = repo or Path(__file__).resolve().parents[2]
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def file_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_experiment_artifacts(
    out_dir: Path,
    *,
    config: dict[str, Any],
    command: list[str] | None = None,
    split_manifest: Path | None = None,
    normalization: dict[str, Any] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(_to_yaml(config), encoding="utf-8")
    cmd = command or sys.argv
    (out_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    env_lines = [
        f"python={sys.version.split()[0]}",
        f"platform={sys.platform}",
        f"timestamp={datetime.now(timezone.utc).isoformat()}",
    ]
    try:
        import torch

        env_lines.append(f"torch={torch.__version__}")
        env_lines.append(f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            env_lines.append(f"cuda_device={torch.cuda.get_device_name(0)}")
    except ImportError:
        env_lines.append("torch=not_installed")
    for key in ("CUDA_VISIBLE_DEVICES", "LOSS_EVAL_DEVICE"):
        if key in os.environ:
            env_lines.append(f"{key}={os.environ[key]}")
    (out_dir / "environment.txt").write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    (out_dir / "git_commit.txt").write_text(git_commit_hash() + "\n", encoding="utf-8")
    if split_manifest is not None and split_manifest.is_file():
        dest = out_dir / "split_manifest.json"
        dest.write_text(split_manifest.read_text(encoding="utf-8"), encoding="utf-8")
        (out_dir / "split_manifest.sha256").write_text(file_sha256(split_manifest) + "\n")
    if normalization is not None:
        (out_dir / "normalization.json").write_text(
            json.dumps(normalization, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _to_yaml(obj: Any, indent: int = 0) -> str:
    sp = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return "{}\n"
        lines = []
        for k, v in obj.items():
            key = str(k)
            if isinstance(v, (dict, list)):
                lines.append(f"{sp}{key}:")
                lines.append(_to_yaml(v, indent + 1).rstrip("\n"))
            else:
                lines.append(f"{sp}{key}: {_yaml_scalar(v)}")
        return "\n".join(lines) + "\n"
    if isinstance(obj, list):
        if not obj:
            return "[]\n"
        lines = []
        for v in obj:
            if isinstance(v, (dict, list)):
                lines.append(f"{sp}-")
                lines.append(_to_yaml(v, indent + 1).rstrip("\n"))
            else:
                lines.append(f"{sp}- {_yaml_scalar(v)}")
        return "\n".join(lines) + "\n"
    return f"{sp}{_yaml_scalar(obj)}\n"


def _yaml_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v).replace("\n", " ")
    if any(c in s for c in (":", "#", "{", "}", "[", "]", ",")):
        return json.dumps(s, ensure_ascii=False)
    return s if s else '""'
