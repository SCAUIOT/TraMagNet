# TraMagNet — Magnetic Signal Denoising (Paper Reproduction)

Deep learning and traditional methods for denoising magnetic measurement signals under HV cable, subway, and white-noise interference.

**Release tag:** `v1.0-paper` (configure after GitHub release)  
**Code version:** commit hash recorded in each run directory as `git_commit.txt`

## Layout note (paper reproduction scripts)

This repository uses a **paper reproduction script layout**, not a full installable library API yet.
Training and evaluation are invoked via `scripts/*.py` wrappers and `main/` entry points.
The `src/tramagnet/` package is a placeholder for future library packaging; use the commands below for reproduction.

## Methods

| Name in paper | Directory / CLI | Config |
|---------------|-----------------|--------|
| **TraMagNet** (proposed) | `scripts/train.py tramagnet` | `configs/tramagnet.yaml` |
| **DnCNN baseline** | `scripts/train.py dncnn` | `configs/dncnn.yaml` |
| **DnCNN-only ablation** | `scripts/train.py dncnn_ablation` | `configs/dncnn_ablation.yaml` |
| **UNet-only ablation** | `scripts/train.py unet_ablation` | `configs/unet_ablation.yaml` |
| Gradient wavelet morphological filter | `scripts/infer_traditional.py` | — |
| Multi-SE morphological filter | `scripts/infer_traditional.py` | — |

## Data

- **Public release:** dataset files are **not** redistributed in this repository (see `data/README.md`).
- Place data under `datasets/` using tags `data1` (high-voltage_cable), `data3` (subway), `data4` (gaussian_noise).
- Splits: `splits/ztest5_data134_manifest.json` (pooled 8:2 + 5-fold CV), `splits/data1_split8020_seed42.json`.

## Environment

- Python **3.10+**
- PyTorch **2.1+** with CUDA optional (CPU supported for tests)
- Install:

```bash
pip install -r requirements.txt
# or
pip install -e ".[dev]"
```

Tested on Windows 10/11 and Linux; GPU: NVIDIA RTX series (CUDA 11.8+).

## Normalization (no test leakage)

Default preprocessing (**since v1.0-paper**):

```yaml
  match_noisy_scale_to_reference: false
  zscore_using_reference: false
normalization: noisy_sample   # z-score using noisy segment statistics only
```

Models trained with older defaults must be **re-trained** before comparing to paper tables.

**Deep learning models only.** The defaults above apply to CNN / TraMagNet / ablation training and evaluation.

### Traditional morphological filter baselines (different protocol)

The scripts under `main/traditional/` (`run_py_denoise_methods.py`, invoked via `scripts/infer_traditional.py`) follow the **paired reference/noisy offline evaluation protocol** used in this project’s traditional-method reproduction:

1. Each noisy segment is processed together with its **paired reference** for visualization and metric export.
2. Preprocessing may use **reference-derived statistics** (e.g. scale matching / z-score referenced to clean). This is **not** the same as the deep models’ default noisy-only / no-leakage settings.
3. These baselines must **not** be described as a blind denoising deployment setting—they assume access to paired references for the evaluation pipeline.
4. If cited in a paper, state explicitly that traditional filters were compared under this **preprocessing protocol**, separate from the learning-based no-leakage protocol.

## Training

```bash
# TraMagNet on pooled data134, 5-fold CV
cd main/TraMagNet
python train.py --data-roots data1,data3,data4 --split-manifest ../splits/ztest5_data134_manifest.json --epochs 2000

# DnCNN baseline on data1
cd main/cnn
python train.py --data-root data1 --epochs 500

# Via unified script
python scripts/train.py tramagnet -- --data-root data1 --epochs 10
python scripts/train.py dncnn -- --data-root data1 --epochs 10
```

Checkpoints go to `output/<dataset>/runs/fold_*` (gitignored).

## Inference / traditional filters

```bash
python scripts/infer_traditional.py --data-roots data1 --max-pairs 10
python main/visualize_data.py cnn --data-root data1 --split test
python main/visualize_data.py tramagnet --data-root data1 --split test
```

## Evaluation

```bash
cd main
python eval_metrics.py --data-root data1 --split test --methods cnn,TraMagNet
python loss_eval.py --mode report --data-root data1 --split test
# or
python ../scripts/evaluate.py metrics --data-root data1 --split test
```

**Metrics:** time-domain SNR (dB), frequency-domain SNR, joint SNR — see docstrings in `main/eval_metrics.py`.

## Reproduce paper tables

1. Prepare data (`data/README.md`)
2. Regenerate splits if needed: `python tools/regenerate_splits.py`
3. Train all methods using configs in `configs/`
4. Run `python scripts/evaluate.py metrics --data-root data1 --split test`
5. Compare `metrics.json` / console SNR tables with paper

Each training fold saves: `config.yaml`, `command.txt`, `environment.txt`, `git_commit.txt`, `split_manifest.json`, `normalization.json` (when enabled in trainer).

## Tests

```bash
pytest -q
python -m compileall -q .
```

## Release zip (no cache)

After tests pass, pack without `__pycache__` / `.pytest_cache`:

```bash
python tools/pack_release.py
# creates ../public.zip next to public/
```

Do **not** run `pytest` after packing (it recreates cache dirs). Re-run tests before the next pack if needed.

## Citation

If you use this code, please cite the paper (update DOI/arXiv in `CITATION.cff` when the preprint is available):

```bibtex
@article{shen2026tramagnet,
  title   = {TraMagNet: Conditional Adversarial Deep Learning for One-Dimensional Magnetic Signal Denoising},
  author  = {Shen},
  year    = {2026},
  note    = {Manuscript under preparation. Code: \url{https://github.com/witness/tramagnet-denoising}}
}
```

Release tag for paper reproduction: `v1.0-paper` (see GitHub Releases).

## License

MIT — see [LICENSE](LICENSE).

## Legacy migration scripts

One-off import tools live under `tools/legacy/` and are **not** required for paper reproduction.
