# Docs assets

Figures linked from the root README (ROC, Grad-CAM grids, demo strip).

Regenerate plots on a machine with the trained checkpoint and PCam data:

```bash
python -m pcam_mets_explain.explain --split valid --max-samples 4000
# then copy selected PNGs from reports/figures/ into docs/figures/
```
