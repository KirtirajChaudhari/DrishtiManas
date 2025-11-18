"""Lightweight ocular backbone placeholder built with PyTorch."""
from __future__ import annotations

import torch
from torch import nn


class OcularBackbone(nn.Module):
    """Tiny CNN for prototyping ocular disease classification."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        feats = self.features(x)
        feats = feats.view(feats.size(0), -1)
        return torch.sigmoid(self.classifier(feats))
