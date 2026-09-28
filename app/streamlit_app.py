"""Upload a histology patch → P(metastasis) + Grad-CAM overlay."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from pcam_mets_explain.config import CHECKPOINTS, CLASS_NAMES, IMAGE_SIZE
from pcam_mets_explain.infer import default_checkpoint, explain_patch, load_model_for_demo, load_rgb_image

st.set_page_config(page_title="pcam-mets-explain", layout="centered")
st.title("PatchMets-XAI")
st.caption("Research demo only. Not a diagnosis. Patch-level, not whole-slide. (pcam-mets-explain)")

st.warning(
    "This is a **research demo**, not a medical device. "
    "It scores one small tissue square (~96×96 px). "
    "It cannot diagnose a patient or replace a pathologist."
)

_devices = ["cpu"]
if torch.cuda.is_available():
    _devices.append("cuda")

with st.sidebar:
    st.header("Model")
    checkpoint_str = st.text_input(
        "Checkpoint path",
        value=str(default_checkpoint()),
        help=f"Default: {CHECKPOINTS / 'best.pt'}",
    )
    device_choice = st.selectbox("Device", _devices, index=0)
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
    st.markdown(
        f"Expected input: **RGB** PNG/JPEG, ideally **{IMAGE_SIZE}×{IMAGE_SIZE}**. "
        "Other sizes are resized automatically."
    )


@st.cache_resource(show_spinner="Loading model…")
def _cached_model(checkpoint: str, device: str):
    return load_model_for_demo(Path(checkpoint), device=device)


uploaded = st.file_uploader("Upload a lymph-node histology patch", type=["png", "jpg", "jpeg"])

if uploaded is None:
    st.info("Upload a PNG or JPEG patch to see P(metastasis) and a Grad-CAM heatmap.")
    st.stop()

rgb, was_resized = load_rgb_image(Image.open(uploaded))

try:
    model, payload, checkpoint_path = _cached_model(checkpoint_str, device_choice)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001 — show friendly errors in the UI
    st.error(f"Could not load checkpoint on {device_choice}: {exc}")
    st.stop()

result = explain_patch(model, rgb)
probability = float(result["probability"])
label_name = CLASS_NAMES[int(probability >= threshold)]

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Patch")
    st.image(
        rgb,
        caption=f"{IMAGE_SIZE}×{IMAGE_SIZE}" + (" (resized)" if was_resized else ""),
        use_container_width=True,
    )
with col_b:
    st.subheader("Grad-CAM")
    st.image(result["overlay"], caption="Warmer = higher model attention", use_container_width=True)

st.metric("P(metastasis)", f"{probability:.3f}")
st.write(f"**Call at threshold {threshold:.2f}:** `{label_name}`")

meta = []
if isinstance(payload, dict):
    auc = payload.get("val_auc", payload.get("best_val_auc"))
    if auc is not None:
        meta.append(f"checkpoint val AUC ≈ {float(auc):.3f}")
meta.append(f"weights: `{checkpoint_path}`")
st.caption(" · ".join(meta))