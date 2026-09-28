"""ResNet18 binary classifier for PCam patches."""

from __future__ import annotations

import torch
from torch import nn


def build_resnet18(pretrained: bool = True) -> nn.Module:
    from torchvision import models
    from torchvision.models import ResNet18_Weights

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def load_checkpoint(path, map_location="cpu") -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model = build_resnet18(pretrained=False)
    model.load_state_dict(payload["model"])
    return model, payload
