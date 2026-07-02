"""public/main dataset paths: ``../datasets/`` and legacy ``data1/data3/data4`` aliases."""

from __future__ import annotations

from pathlib import Path

MAIN_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = MAIN_ROOT.parent
DATASETS_DIR = (PUBLIC_ROOT / "datasets").resolve()

# CLI / manifest may still use legacy tags; physical dirs live under public/datasets/.
PATH_ALIASES: dict[str, str] = {
    "data1": "high-voltage_cable",
    "data2": "high-voltage_cable",
    "data3": "subway",
    "data4": "gaussian_noise",
    "high-voltage_cable": "high-voltage_cable",
    "subway": "subway",
    "gaussian_noise": "gaussian_noise",
}

DEFAULT_SINGLE_ROOT = "data1"
DEFAULT_POOL_ROOTS = ("data1", "data3", "data4")
DEFAULT_POOL_TAG = "data134"


def physical_dataset_name(name: str) -> str:
    key = str(name).strip().replace("\\", "/").rstrip("/")
    if not key:
        return key
    leaf = Path(key).name
    return PATH_ALIASES.get(leaf, leaf)


def is_legacy_alias(name: str) -> bool:
    leaf = Path(str(name).strip()).name
    return leaf in PATH_ALIASES and leaf.startswith("data")


def dataset_tag_for_path(path: Path) -> str:
    """Output directory tag: physical name high-voltage_cable → still exposed as data1 (matches legacy manifests)."""
    name = path.resolve().name
    for legacy, physical in PATH_ALIASES.items():
        if physical == name and legacy.startswith("data"):
            return legacy
    return name


def _dataset_candidates(raw: str, physical: str, repo: Path) -> list[Path]:
    p = Path(raw).expanduser()
    cands = [
        DATASETS_DIR / physical,
        repo.parent / "datasets" / physical,
        repo / "datasets" / physical,
    ]
    if len(p.parts) > 1:
        cands.insert(0, (PUBLIC_ROOT / p).resolve())
        cands.insert(0, (repo / p).resolve())
    cands.extend(
        [
            repo / raw,
            repo / physical,
            Path.cwd() / raw,
            Path.cwd() / physical,
        ]
    )
    if p.is_absolute():
        cands.insert(0, p.resolve())
    return cands


def resolve_dataset_root(data_root: str | Path, *, repo: Path | None = None) -> str:
    """
    Resolve a data root directory.

    For aliases such as ``data1`` / ``data3`` / ``data4``, **prefer** ``public/datasets/<physical name>``
    to avoid accidentally picking up legacy ``data1`` dirs outside the repo.
    """
    repo = MAIN_ROOT if repo is None else Path(repo)
    raw = str(data_root).strip()
    p = Path(raw).expanduser()
    leaf = p.name if p.name else raw
    physical = physical_dataset_name(leaf)

    if is_legacy_alias(raw) or is_legacy_alias(leaf):
        for cand in _dataset_candidates(raw, physical, repo):
            try:
                c = cand.resolve()
            except OSError:
                continue
            if c.is_dir() and "datasets" in c.parts:
                return str(c)

    if p.is_dir():
        return str(p.resolve())

    for cand in _dataset_candidates(raw, physical, repo):
        try:
            c = cand.resolve()
        except OSError:
            continue
        if c.is_dir():
            return str(c)

    return str((DATASETS_DIR / physical).resolve())


def resolve_data_root_entry(part: str, *, repo: Path | None = None) -> tuple[str, Path]:
    """
    Parse one ``--data-roots`` entry; return ``(tag, path)``.

    ``tag`` keeps the CLI name (e.g. ``data1``) for pooled group keys ``{tag}/{sample_id}``.
    ``path`` points at the actual ``public/datasets/…`` directory.
    """
    repo = MAIN_ROOT if repo is None else Path(repo)
    part = str(part).strip()
    if not part:
        raise ValueError("empty data root entry")

    p = Path(part).expanduser()
    if p.is_absolute() and p.is_dir():
        tag = part.split("/")[-1].split("\\")[-1]
        if tag in PATH_ALIASES:
            return tag, p.resolve()
        return dataset_tag_for_path(p), p.resolve()

    if len(p.parts) > 1 and not is_legacy_alias(part):
        for base in (repo, PUBLIC_ROOT, Path.cwd()):
            cand = (base / p).resolve()
            if cand.is_dir():
                tag = part.split("/")[-1].split("\\")[-1]
                return (tag if tag in PATH_ALIASES else dataset_tag_for_path(cand)), cand

    tag = p.name if p.name else part
    if is_legacy_alias(tag):
        tag = Path(part).name if Path(part).name.startswith("data") else tag
    resolved = Path(resolve_dataset_root(part if is_legacy_alias(part) else tag, repo=repo))
    if not resolved.is_dir():
        raise FileNotFoundError(f"data root not found: {part} -> {resolved}")
    legacy_tag = tag if tag in PATH_ALIASES and tag.startswith("data") else dataset_tag_for_path(resolved)
    return legacy_tag, resolved
