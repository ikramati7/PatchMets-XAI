"""Grad-CAM for the ResNet18 binary PCam classifier."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class GradCAM:
    """Capture activations and gradients from one convolutional layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()

    def __call__(self, images: torch.Tensor) -> np.ndarray:
        """Return Grad-CAM maps shaped (N, H, W) in [0, 1]."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(images).squeeze(1)
        logits.sum().backward()
        assert self.activations is not None and self.gradients is not None
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        denom = cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        cam = cam / denom
        return cam.detach().cpu().numpy()


def resnet18_target_layer(model: nn.Module) -> nn.Module:
    return model.layer4[-1]


def overlay_cam_on_rgb(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a CAM heatmap onto an RGB uint8 image."""
    from PIL import Image

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    height, width = rgb.shape[:2]
    heat = Image.fromarray(np.uint8(255 * cam), mode="L").resize((width, height), Image.Resampling.BILINEAR)
    heat_arr = np.asarray(heat, dtype=np.float32) / 255.0
    # Simple matplotlib-like hot colormap without importing pyplot here.
    r = np.clip(1.5 * heat_arr, 0, 1)
    g = np.clip(1.5 * heat_arr - 0.5, 0, 1)
    b = np.clip(1.5 * heat_arr - 1.0, 0, 1)
    color = np.stack([r, g, b], axis=-1)
    blended = (1.0 - alpha) * (rgb.astype(np.float32) / 255.0) + alpha * color
    return np.uint8(np.clip(blended * 255.0, 0, 255))
