"""Cue extraction: turns one image (or a short clip) into six cue scores.

Detectors (paper Section 4.3):
  * tire marks, debris, scattered objects -> YOLOv8 object detectors
  * traffic load -> YOLOv8 vehicle detection + multi-object tracking
  * weather -> 7-class CNN classifier
  * red-region cue -> colour heuristic (red_region.py)

The YOLO models are optional (``pip install ultralytics``). Weights are NOT
shipped; train them on your own annotations and pass the paths in the config.
Any detector that is not configured returns 0.0, so the pipeline still runs
for testing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .red_region import red_region
from .scoring import (
    CUE_NAMES,
    WEATHER_CLASSES,
    LevelThresholds,
    level_to_score,
    tire_marks_to_score,
    value_to_level,
    weather_to_score,
)

VEHICLE_COCO_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck


def _load_yolo(path: Optional[str]):
    if not path:
        return None
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install ultralytics to use YOLO-based cue detectors") from e
    return YOLO(path)


def _box_area_fraction(result) -> float:
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return 0.0
    h, w = result.orig_shape
    xyxy = result.boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return float(min(1.0, areas.sum() / (h * w)))


@dataclass
class CueConfig:
    tire_marks_weights: Optional[str] = None
    debris_weights: Optional[str] = None
    objects_weights: Optional[str] = None
    obstructive_class_ids: Sequence[int] = ()
    vehicle_weights: Optional[str] = "yolov8n.pt"
    weather_weights: Optional[str] = None
    conf: float = 0.25


class CueExtractor:
    """Runs all six cue detectors and returns a dict of scores in [0, 1]."""

    def __init__(self, cfg: CueConfig, thresholds: Optional[LevelThresholds] = None, device: str = "cpu"):
        self.cfg = cfg
        self.th = thresholds or LevelThresholds()
        self.device = device
        self.tire = _load_yolo(cfg.tire_marks_weights)
        self.debris = _load_yolo(cfg.debris_weights)
        self.objects = _load_yolo(cfg.objects_weights)
        self.vehicles = _load_yolo(cfg.vehicle_weights) if cfg.vehicle_weights else None
        self.weather = None
        if cfg.weather_weights:
            from .weather import load_weather_model

            self.weather = load_weather_model(cfg.weather_weights, device)

    def _predict(self, model, image):
        if model is None:
            return None
        return model.predict(image, conf=self.cfg.conf, verbose=False, device=self.device)[0]

    def traffic_load_level(self, frames: List[np.ndarray]) -> int:
        """Count unique tracked vehicles across frames (single image = detection count)."""
        if self.vehicles is None:
            return 0
        if len(frames) == 1:
            r = self._predict(self.vehicles, frames[0])
            cls = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.array([])
            n = int(np.isin(cls, list(VEHICLE_COCO_IDS)).sum())
        else:
            ids = set()
            for f in frames:  # ultralytics tracker; swap in DeepSORT if preferred
                r = self.vehicles.track(f, persist=True, verbose=False, conf=self.cfg.conf)[0]
                if r.boxes is None or r.boxes.id is None:
                    continue
                cls = r.boxes.cls.cpu().numpy().astype(int)
                tid = r.boxes.id.cpu().numpy().astype(int)
                ids.update(t for t, c in zip(tid, cls) if c in VEHICLE_COCO_IDS)
            n = len(ids)
        return value_to_level(n, self.th.vehicle_count)

    def __call__(self, image_bgr: np.ndarray, clip: Optional[List[np.ndarray]] = None) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        tire_frac = _box_area_fraction(self._predict(self.tire, image_bgr))
        scores["tire_marks"] = tire_marks_to_score(tire_frac, self.th.tire_area_saturation)

        debris_frac = _box_area_fraction(self._predict(self.debris, image_bgr))
        scores["debris"] = level_to_score(value_to_level(debris_frac, self.th.debris_area))

        r = self._predict(self.objects, image_bgr)
        if r is None or r.boxes is None:
            obj_level = 0
        else:
            cls = r.boxes.cls.cpu().numpy().astype(int)
            obj_level = value_to_level(len(cls), self.th.objects_count)
            if np.isin(cls, list(self.cfg.obstructive_class_ids)).any():
                obj_level = 4
        scores["scattered_objects"] = level_to_score(obj_level)

        scores["traffic_load"] = level_to_score(self.traffic_load_level(clip or [image_bgr]))

        if self.weather is not None:
            from .weather import predict_weather

            scores["weather"] = weather_to_score(predict_weather(self.weather, image_bgr, self.device))
        else:
            scores["weather"] = weather_to_score("clear")

        rr = red_region(image_bgr)
        red_level = value_to_level(rr.area_fraction, self.th.red_area)
        if rr.n_large_blobs >= 2:
            red_level = 4
        scores["red_region"] = level_to_score(red_level)

        return {k: scores[k] for k in CUE_NAMES}


__all__ = ["CueConfig", "CueExtractor", "WEATHER_CLASSES"]
