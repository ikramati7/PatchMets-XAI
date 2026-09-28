"""Single-patch inference + Grad-CAM for the Streamlit demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pcam_mets_explain.config import CHECKPOINTS, IMAGE_SIZE
from pcam_mets_explain.gradcam import GradCAM, overlay_cam_on_rgb, resnet18_target_layer
from pcam_mets_explain.model import load_checkpoint

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_checkpoint() -> Path:
    return CHECKPOINTS / "best.pt"


def load_rgb_image(source) -> tuple[np.ndarray, bool]:
    """Load an RGB uint8 array and whether it was resized to IMAGE_SIZE."""
    if isinstance(source, (str, Path)):
        image = Image.open(source)
    elif isinstance(source, Image.Image):
        image = source
    elif isinstance(source, np.ndarray):
        image = Image.fromarray(source)
    else:
        raise TypeError(f"Unsupported image source type: {type(source)!r}")

    image = image.convert("RGB")
    resized = image.size != (IMAGE_SIZE, IMAGE_SIZE)
    if resized:
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.uint8)
    return rgb, resized


def preprocess_rgb(rgb: np.ndarray) -> torch.Tensor:
    """Convert HxWx3 uint8 RGB to a normalized 1x3xHxW tensor."""
    if rgb.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Expected {(IMAGE_SIZE, IMAGE_SIZE, 3)}, got {rgb.shape}")
    tensor = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)


@torch.no_grad()
def predict_probability(model: torch.nn.Module, batch: torch.Tensor, device: torch.device) -> float:
    model.eval()
    logits = model(batch.to(device)).squeeze()
    return float(torch.sigmoid(logits).item())


def explain_patch(
    model: torch.nn.Module,
    rgb: np.ndarray,
    *,
    device: torch.device | None = None,
    alpha: float = 0.45,
) -> dict:
    """Return probability, CAM, and overlay for one RGB patch."""
    if device is None:
        device = next(model.parameters()).device
    batch = preprocess_rgb(rgb)
    probability = predict_probability(model, batch, device)

    model.eval()
    cam_helper = GradCAM(model, resnet18_target_layer(model))
    try:
        cam = cam_helper(batch.to(device))[0]
    finally:
        cam_helper.close()

    overlay = overlay_cam_on_rgb(rgb, cam, alpha=alpha)
    return {
        "probability": probability,
        "cam": cam,
        "overlay": overlay,
        "predicted_label": int(probability >= 0.5),
    }


def load_model_for_demo(checkpoint: Path | None = None, device: str | torch.device = "cpu"):
    path = Path(checkpoint) if checkpoint is not None else default_checkpoint()
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Train first: python -m pcam_mets_explain.train"
        )
    torch_device = torch.device(device)
    model, payload = load_checkpoint(path, map_location=torch_device)
    model.to(torch_device)
    model.eval()
    return model, payload, path
