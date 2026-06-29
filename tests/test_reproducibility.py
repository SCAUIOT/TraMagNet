from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main"
sys.path.insert(0, str(MAIN))

from data_common.normalization import NormalizationConfig, normalize_pair  # noqa: E402


def test_normalization_uses_noisy_stats_only_by_default():
    reference = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    noisy = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    cfg = NormalizationConfig()
    c_n, n_n = normalize_pair(reference, noisy, cfg)
    mu = noisy.mean()
    sig = noisy.std(unbiased=False)
    expected_n = (noisy - mu) / sig
    assert torch.allclose(n_n, expected_n, atol=1e-5)
    assert not torch.allclose(c_n, torch.zeros_like(c_n))


def test_no_reference_stats_in_noisy_input():
    reference = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    noisy = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
    cfg = NormalizationConfig(normalization="noisy_sample")
    _, n_n = normalize_pair(reference, noisy, cfg)
    assert torch.allclose(n_n, torch.zeros_like(n_n), atol=1e-5)


def _tiny_dncnn():
    sys.path.insert(0, str(MAIN / "cnn"))
    from models.dncnn_1d import DnCNN1D, DnCNN1DConfig

    return DnCNN1D(
        DnCNN1DConfig(
            features=4,
            middle_depth=1,
            num_residual_blocks=1,
            use_bn=False,
        )
    )


def test_dncnn_forward_shape():
    m = _tiny_dncnn()
    x = torch.randn(1, 1, 64)
    y = m(x)
    assert tuple(y.shape) == (1, 1, 64)


def test_loss_backward():
    m = _tiny_dncnn()
    pred = m(torch.randn(1, 1, 64))
    target = torch.randn(1, 1, 64)
    loss = torch.nn.functional.l1_loss(pred, target)
    loss.backward()
    assert any(p.grad is not None for p in m.parameters())


def test_eval_metrics_on_toy_data():
    from eval_metrics import compute_five_metrics

    reference = np.linspace(0, 1, 256)
    den = reference + 0.01 * np.random.default_rng(0).standard_normal(256)
    noisy = reference + 0.1 * np.random.default_rng(1).standard_normal(256)
    m = compute_five_metrics(reference, noisy, den)
    assert m.snr_db > m.snr_noisy_db


def test_pooled_config_defaults_no_reference_leakage():
    from data_common.pooled_our_data_dataset import PooledOurDataConfig

    cfg = PooledOurDataConfig(root_entries=[])
    assert cfg.match_noisy_scale_to_reference is False
    assert cfg.zscore_using_reference is False


def test_split_manifest_no_absolute_paths():
    import json

    manifest = json.loads((MAIN / "splits" / "ztest5_data134_manifest.json").read_text(encoding="utf-8"))
    roots = manifest.get("data_roots") or {}
    for tag, path in roots.items():
        assert tag == path or not Path(str(path)).is_absolute(), f"absolute path in manifest: {path}"


def test_compileall_public():
    import compileall

    ok = compileall.compile_dir(str(ROOT), quiet=1)
    assert ok
