# Model card — PatchMets-XAI

**Research and education only. Not a medical device. Not for diagnosis.**

## What is PatchMets-XAI?

| Part | Meaning |
| --- | --- |
| **Patch** | The model looks at one small square of tissue (96×96 pixels), not a whole glass slide. |
| **Mets** | Short for **metastasis** — cancer that has spread to a lymph node. |
| **XAI** | Short for **explainable AI** — we show a heatmap (Grad-CAM) of which pixels the model used. |

So the name means: *an explained patch-level metastasis detector*.

(Folder name in this repo may still be `pcam-mets-explain`; the public project name is **PatchMets-XAI**.)

## What the model does

- **Input:** one RGB histology patch (ideally 96×96; other sizes are resized).
- **Output:** a score **P(metastasis)** between 0 and 1, plus a Grad-CAM overlay.
- **Classes:** `no_metastasis` (0) vs `metastasis` (1).

## Data

- **Dataset:** [PatchCamelyon (PCam)](https://github.com/basveeling/pcam) — public lymph-node patches derived from Camelyon16.
- **Do not** put hospital / private patient slides in this repository.
- The large `.h5` files are **not** pushed to GitHub (see `.gitignore`).

## Training setup (this repo’s reported run)

| Item | Value |
| --- | --- |
| Model | ResNet18 (ImageNet init, binary head) |
| Train / valid size | 20 000 / 4 000 balanced subset |
| Epochs | 3 |
| Device | CUDA GPU |
| Checkpoint | `reports/checkpoints/best.pt` (local only; not in git) |

## Metrics

| Split | AUC | Accuracy (threshold 0.5) |
| --- | ---: | ---: |
| Validation | ~0.946 | ~0.87 |
| Held-out test | ~0.927 | ~0.82 |

See also `reports/metrics.json`, `reports/explain_report.json`, and figures under `docs/figures/`.

## Leakage and fairness (honest limits)

| Question | Answer for this project |
| --- | --- |
| Can the same **slide (WSI)** appear in train and test? | **No** for official PCam splits — they are WSI-disjoint. |
| Can the same **patient** appear in more than one split? | **Unknown** — public PCam does not give patient IDs here. |
| Does a high AUC mean it works in any hospital? | **No.** Everything is still **in-domain PCam**. Other scanners, stains, or labs can look different. |
| Is one patch a patient diagnosis? | **No.** Patch-level only. |

## Intended use

- Learning computational pathology
- Portfolio / research demo
- Exploring mistakes with Grad-CAM

## Out of scope

- Clinical decisions
- Whole-slide triage
- Replacing a pathologist

## How to reproduce

See the root [README](../README.md): download PCam → train → `explain` → Streamlit demo.
