# Data layout

This repository expects datasets under `datasets/` (not committed). Legacy CLI tags map as:

| Tag | Directory | Description |
|-----|-----------|-------------|
| `data1` | `datasets/high-voltage_cable` | high-voltage noise |
| `data3` | `datasets/subway` | subway noise |
| `data4` | `datasets/gaussian_noise` | gaussian noise |

Each dataset directory contains matching pairs under two folders (same basename = same index):

```
datasets/high-voltage_cable/
├── reference_signal/
│   └── sample1.txt, sample2.txt, ...
└── noise_signal/
    └── sample1.txt, sample2.txt, ...
```

Reference and noisy files are paired **by filename** (`sample{i}.txt` in both directories).

## If data cannot be published 

Place your own paired `.txt` files following the same layout and naming (`sample{id}.txt` or band-axis naming documented in `main/data_common/pair_specs.py`). Pass `--data-root data1` or an absolute path to your copy.

Do **not** commit raw measurement data if your license prohibits redistribution.
