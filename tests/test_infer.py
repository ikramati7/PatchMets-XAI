"""Tests for single-patch inference helpers (no Streamlit, no real checkpoint)."""

from pathlib import Path

import numpy as np
import torch

from pcam_mets_explain.config import IMAGE_SIZE
from pcam_mets_explain.infer import (
    explain_patch,
    load_model_for_demo,
    load_rgb_image,
    predict_probability,
    preprocess_rgb,
)
from pcam_mets_explain.model import build_resnet18


def test_load_rgb_image_resizes():
    from PIL import Image

    big = Image.fromarray(np.zeros((128, 64, 3), dtype=np.uint8))
    rgb, resized = load_rgb_image(big)
    assert resized is True
    assert rgb.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert rgb.dtype == np.uint8


def test_preprocess_and_predict_shapes(tmp_path: Path):
    model = build_resnet18(pretrained=False)
    model.eval()
    checkpoint = tmp_path / "toy.pt"
    torch.save({"model": model.state_dict(), "best_val_auc": 0.5}, checkpoint)

    loaded, payload, path = load_model_for_demo(checkpoint, device="cpu")
    assert path == checkpoint
    assert payload["best_val_auc"] == 0.5

    rgb = np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    batch = preprocess_rgb(rgb)
    assert batch.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)

    prob = predict_probability(loaded, batch, torch.device("cpu"))
    assert 0.0 <= prob <= 1.0

    out = explain_patch(loaded, rgb, device=torch.device("cpu"))
    assert out["overlay"].shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert out["predicted_label"] in (0, 1)
    assert abs(out["probability"] - prob) < 1e-5
