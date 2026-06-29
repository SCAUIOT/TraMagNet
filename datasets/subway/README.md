## subway noise paired sample

This dataset contains a paired signal sample for subway noise denoising. The pair consists of two files with the same sample name, `sample1.txt`, stored under different subdirectories according to the dataset layout:

```text
datasets/white_noise/
├── reference_signal/
│   └── sample1.txt      # reference signal, 3 columns
├── noise_signal/
│   └── sample1.txt      # noisy signal, 3 columns
```

### Noisy signal file: `noise_signal/sample1.txt`

Each row in the noisy signal file contains three columns:

| Column   | Meaning            | Description                                                            |
| -------- | ------------------ | ---------------------------------------------------------------------- |
| Column 1 | Sample index       | Sequential sample identifier                                           |
| Column 2 | Timestamp          | Sampling timestamp                                                     |
| Column 3 | Noisy signal value | Measured signal amplitude after High-voltage cable noise contamination |

An example row has the following structure:

```text
%%%%%%%%%%449    1610678462805    753
```

In this row, `449` is the sample index, `1610678462805` is the timestamp, and `147.55097867` is the noisy signal amplitude at that sampling point.

The corresponding reference signal file structure is the same as the noisy signal file structure.

### Pairing rule

The noisy and reference files are paired by filename and row alignment. For example:

```text
noise_signal/sample1.txt  ↔  reference_signal/sample1.txt
```

For each row, the first two columns should be identical or directly corresponding between the two files. The third column of the noisy file is the model input, while the third column of the reference file is the expected reference target. 
