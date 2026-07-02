# TraMagNet — Magnetic Signal Denoising (Paper Reproduction)

Deep learning and traditional methods for denoising magnetic measurement signals under high-voltage noise, subway noise, and gaussian noise.

**Release tag:** `v1.0-paper` (configure after GitHub release)  
**Code version:** commit hash recorded in each run directory as `git_commit.txt`

## Layout note (paper reproduction scripts)

This repository uses a **paper reproduction script layout**: training and evaluation via `scripts/*.py` and `main/` entry points.

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
- Train/test splits are computed in code from `--seed` (default 42) and `--train-ratio` (default 0.8); no external split files.

## Environment

- Python **3.10+**
- PyTorch **2.1+** with CUDA optional (CPU supported)
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

**Deep learning models only.** The defaults above apply to DnCNN / TraMagNet / ablation training and evaluation.

### Traditional morphological filter baselines (different protocol)

The scripts under `main/traditional/` (`run_py_denoise_methods.py`, invoked via `scripts/infer_traditional.py`) follow the **paired reference/noisy offline evaluation protocol** used in this project’s traditional-method reproduction:

1. Each noisy segment is processed together with its **paired reference reference** for visualization and metric export.
2. Preprocessing may use **clean-derived statistics** (e.g. scale matching / z-score referenced to clean). This is **not** the same as the deep models’ default noisy-only / no-leakage settings.
3. These baselines must **not** be described as a blind denoising deployment setting—they assume access to paired references for the evaluation pipeline.
4. If cited in a paper, state explicitly that traditional filters were compared under this **preprocessing protocol**, separate from the learning-based no-leakage protocol.

## Training

```bash
# TraMagNet on pooled data134, 5-fold CV
cd main/TraMagNet
python train.py --data-roots data1,data3,data4 --epochs 2000

# DnCNN baseline on data1
cd main/DnCNN
python train.py --data-root data1 --epochs 500

# Via unified script
python scripts/train.py tramagnet -- --data-root data1 --epochs 10
python scripts/train.py dncnn -- --data-root data1 --epochs 10
```

Checkpoints go to `output/<dataset>/runs/fold_*` (gitignored).

## Inference / traditional filters

```bash
python scripts/infer_traditional.py --data-roots data1 --max-pairs 10
python main/visualize_data.py dncnn --data-root data1 --split test
python main/visualize_data.py tramagnet --data-root data1 --split test
```

## Evaluation

```bash
cd main
python eval_metrics.py --data-root data1 --split test --methods dncnn,TraMagNet
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

## Sanity check

```bash
python -m compileall -q .
```

For data pairing verification (repo root, not in this package): `python test10.py`.

## Release zip (no cache)

Pack without `__pycache__`:

```bash
python tools/pack_release.py
# creates ../public.zip next to public/
```

## Citation

If you use this code, please cite the paper (update DOI/arXiv in `CITATION.cff` when the preprint is available):

```bibtex
@article{,
  title   = 
  author  = 
  year    = 
  note    = 
}
```

Release tag for paper reproduction: `v1.0-paper` (see GitHub Releases).

## Contributors

<table>
  <tr>
    <td align="center" width="140">
      <img
        src="./assets/jc_hu.jpg"
        width="100"
        height="100"
        alt="Jingcheng Hu"
      />
      <br />
      <sub><b>Jingcheng Hu</b></sub>
    </td>
    <td align="center" width="140">
      <img
        src="./assets/jj_wu.jpg"
        width="100"
        height="100"
        alt="Jiajun Wu"
      />
      <br />
      <sub><b>Jiajun Wu</b></sub>
    </td>
    <td align="center" width="140">
      <img
        src="./assets/zs_zhang.jpg"
        width="100"
        height="100"
        alt="Zusheng Zhang"
      />
      <br />
      <sub><b>Zusheng Zhang</b></sub>
    </td>
  </tr>
</table>

Please let us know of any bugs found in the code. Suggestions and collaborations are welcomed

## License

MIT — see [LICENSE](LICENSE).

## Legacy migration scripts

One-off import tools live under `tools/legacy/` and are **not** required for paper reproduction.
