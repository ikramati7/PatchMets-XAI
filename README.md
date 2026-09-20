# pcam-mets-explain

**Research demo (not a diagnosis).** A small model looks at a **tiny histology patch** from a lymph node and guesses whether it contains **metastasis**. It also draws a heatmap of the pixels it used.

This is **patch-level**, not whole-slide analysis. One square is not a patient diagnosis.

> Day 0 status: repo layout, install, and environment check. Training starts on Day 1.

## Setup (Day 0)

```powershell
cd C:\Users\hp\Projects\pcam-mets-explain
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev]"
python -m pcam_mets_explain.check_env
```

PyTorch is **not** installed yet (it is large). On Day 2, install the CPU build if you do not have a GPU:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Folder layout

| Path | Role |
| --- | --- |
| `data/` | PCam files you download (not committed) |
| `src/pcam_mets_explain/` | Code |
| `app/` | Streamlit demo (Day 4) |
| `reports/` | Metrics, plots, example heatmaps (Day 3–5) |
| `tests/` | Small tests that do not need the full dataset |

## Later this week

- Day 1: download PCam (or a subset) and plot example patches
- Day 2: fine-tune ResNet18
- Day 3: ROC, confusion matrix, Grad-CAM on successes and failures
- Day 4: Streamlit app
- Day 5: README screenshots and GitHub polish

## License

MIT. PCam data has its own terms; do not commit the dataset.
