"""
Unified visualization entry point (public/main).

Usage::

    python visualize_data.py <cnn|tramagnet> [backend-specific args...]

Examples::

    python visualize_data.py cnn --data-root ../datasets/high-voltage_cable
    python visualize_data.py tramagnet --data-root ../datasets/high-voltage_cable --split test

Default ``--split test`` uses the fixed held-out test set (20%) from the same 8:2 split as training.
Default ``--ckpt last``; for K-fold, ensemble over each fold's ``last.pt``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_BACKENDS = {
    "cnn": "data_common.viz_cnn_runner",
    "tramagnet": "data_common.viz_tramagnet_runner",
}


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        print(
            "Usage: python visualize_data.py <cnn|tramagnet> [backend args...]\n\n"
            "Examples:\n"
            "  python visualize_data.py cnn --data-root ../datasets/high-voltage_cable\n"
            "  python visualize_data.py tramagnet --data-root ../datasets/high-voltage_cable\n"
            "  (default --split test = held-out test set 20%%; --ckpt last)\n\n"
            "See all args for a backend:\n"
            "  python visualize_data.py cnn --help",
            flush=True,
        )
        return 2
    if argv[0] in ("-h", "--help"):
        print(
            "First argument must be a backend name: cnn | tramagnet\n"
            "Example: python visualize_data.py cnn --help",
            flush=True,
        )
        return 0
    backend = argv[0].lower().strip()
    if backend not in _BACKENDS:
        print(f"Unknown backend {argv[0]!r}. Options: {', '.join(sorted(_BACKENDS))}", flush=True)
        return 2
    sys.argv = [f"{Path(sys.argv[0]).name} {backend}"] + argv[1:]
    mod = importlib.import_module(_BACKENDS[backend])
    return int(mod.main())


if __name__ == "__main__":
    raise SystemExit(main())
