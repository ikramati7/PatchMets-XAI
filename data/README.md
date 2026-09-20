# Data (not in git)

Put PatchCamelyon (PCam) files here. Nothing under `raw/` or `processed/` is committed.

Suggested layout after Day 1:

```text
data/raw/          # original download
data/processed/    # optional smaller subset
```

PCam is a public patch dataset derived from Camelyon16 lymph-node slides. Each image is 96×96 pixels with a binary label (metastasis vs no metastasis).

Download options (pick one on Day 1):

- [PatchCamelyon on GitHub / papers with code](https://github.com/basveeling/pcam)
- TensorFlow Datasets `patch_camelyon` if you already use TF
- Kaggle mirrors of PCam / Camelyon patches

If the full set is too large for your laptop, keep a **subset** (for example 10k train / 2k val / 2k test) and write the counts in `reports/metrics.json` later.
