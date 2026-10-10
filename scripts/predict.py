"""Predict severity for one image and print the cue evidence behind it.

    python scripts/predict.py --image crash.jpg --checkpoint results/model.pt \
        --tire weights/tire.pt --debris weights/debris.pt --objects weights/objects.pt
"""
from __future__ import annotations

import argparse
import json

import cv2
import torch
from PIL import Image

import _path  # noqa: F401
from casev.config import load_config
from casev.cues.detectors import CueConfig, CueExtractor
from casev.cues.scoring import CUE_NAMES
from casev.data import build_transforms
from casev.model import CLASS_NAMES, severity_score
from casev.train import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tire"); ap.add_argument("--debris"); ap.add_argument("--objects")
    ap.add_argument("--vehicles", default="yolov8n.pt"); ap.add_argument("--weather")
    a = ap.parse_args()

    cfg = load_config(a.config, pretrained=False, device="cpu")
    model = build_model(cfg)
    model.load_state_dict(torch.load(a.checkpoint, map_location="cpu"))
    model.eval()

    bgr = cv2.imread(a.image)
    cues = CueExtractor(CueConfig(a.tire, a.debris, a.objects, (), a.vehicles, a.weather))(bgr)
    x = build_transforms(cfg.image_size, train=False)(Image.open(a.image).convert("RGB")).unsqueeze(0)
    c = torch.tensor([[cues[k] for k in CUE_NAMES]], dtype=torch.float32)
    with torch.no_grad():
        p = torch.softmax(model(x, c), 1)
    print(json.dumps({
        "cue_scores": {k: round(v, 2) for k, v in cues.items()},
        "class_probabilities": {n: round(float(v), 3) for n, v in zip(CLASS_NAMES, p[0])},
        "predicted_class": CLASS_NAMES[int(p.argmax())],
        "severity_score": round(float(severity_score(p)[0]), 3),
    }, indent=2))


if __name__ == "__main__":
    main()
