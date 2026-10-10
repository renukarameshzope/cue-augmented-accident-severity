"""Seven-class weather classifier (clear ... stormy) used for the weather cue."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from .scoring import WEATHER_CLASSES

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_weather_model(pretrained: bool = False) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    m = models.mobilenet_v3_small(weights=weights)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, len(WEATHER_CLASSES))
    return m


def load_weather_model(path: str, device: str = "cpu") -> nn.Module:
    m = build_weather_model(pretrained=False)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device).eval()


@torch.no_grad()
def predict_weather(model: nn.Module, image_bgr: np.ndarray, device: str = "cpu") -> str:
    rgb = cv2.cvtColor(cv2.resize(image_bgr, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(((rgb - _MEAN) / _STD).transpose(2, 0, 1)).unsqueeze(0).to(device)
    return WEATHER_CLASSES[int(model(x).argmax(1).item())]
