import numpy as np
import torch
from torch import nn

from pcam_mets_explain.explain import pick_case_indices
from pcam_mets_explain.gradcam import GradCAM, overlay_cam_on_rgb
from pcam_mets_explain.model import build_resnet18


def test_pick_case_indices_covers_four_buckets():
    labels = np.array([0, 0, 1, 1, 0, 1])
    preds = np.array([0, 1, 1, 0, 0, 1])
    probs = np.array([0.1, 0.9, 0.8, 0.2, 0.05, 0.95])
    chosen = pick_case_indices(labels, preds, probs, per_bucket=1)
    assert chosen["true_negative"]
    assert chosen["true_positive"]
    assert chosen["false_positive"]
    assert chosen["false_negative"]


def test_gradcam_runs_on_resnet18():
    model = build_resnet18(pretrained=False)
    model.eval()
    cam = GradCAM(model, model.layer4[-1])
    images = torch.zeros(2, 3, 96, 96)
    maps = cam(images)
    assert maps.shape == (2, 3, 3) or maps.shape[0] == 2
    assert maps.min() >= 0.0
    cam.close()


def test_overlay_cam_on_rgb_shape():
    rgb = np.zeros((96, 96, 3), dtype=np.uint8)
    cam = np.linspace(0, 1, 12 * 12, dtype=np.float32).reshape(12, 12)
    out = overlay_cam_on_rgb(rgb, cam)
    assert out.shape == (96, 96, 3)
    assert out.dtype == np.uint8
