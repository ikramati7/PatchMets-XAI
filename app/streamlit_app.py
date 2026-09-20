"""Upload a patch and show P(metastasis) + heatmap. Implemented on Day 4."""

import streamlit as st

st.set_page_config(page_title="pcam-mets-explain", layout="centered")
st.title("pcam-mets-explain")
st.caption("Research demo only. Not a diagnosis. Patch-level, not whole-slide.")
st.info("Day 0: app skeleton. The upload → probability → Grad-CAM flow lands on Day 4.")
