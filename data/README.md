# Data (not in git)

PCam files go here. **Never** `git add` images or `.h5` files.

```text
data/raw/          # original download (.h5.gz then unzipped .h5)
data/processed/    # optional smaller subset later
```

Each patch is 96×96 RGB. Label `0` = no metastasis, `1` = metastasis.

Published sizes:

| Split | Images |
| --- | ---: |
| train | 262,144 |
| valid | 32,768 |
| test | 32,768 |

## Day 1 on the lab server (recommended)

```bash
cd /data_sde/ikrame/pcam-mets-explain
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,download]"
```

Copy the updated `src/` from your laptop first if the server still has only Day 0 files.

**Light download** (skip ~6 GB train images; enough for Day 1 plots + split counts):

```bash
python -m pcam_mets_explain.download --light
python -m pcam_mets_explain.preview
```

**Full download** (needed before Day 2 training):

```bash
python -m pcam_mets_explain.download
python -m pcam_mets_explain.preview
```

Check:

- `reports/split_counts.json` — train/valid/test counts
- `reports/figures/pcam_examples.png` — 4 normal + 4 metastasis

Downloads come from Hugging Face (`1aurent/PatchCamelyon`), not Google Drive. The first `--light` run pulls **valid + test** images (about 1.5 GB) plus all labels. Full `train` images are only fetched without `--light`.
