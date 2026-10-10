"""Mapping from cue-detector outputs to normalised scores in [0, 1].

Implements the scoring scheme described in the paper (Table 2 and Section 4.4).
Every cue produces a single float so that the six scores can be concatenated
with the image embedding.

Cue order (used everywhere in this package):
    0 tire_marks, 1 debris, 2 scattered_objects, 3 traffic_load, 4 weather, 5 red_region
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

CUE_NAMES: List[str] = [
    "tire_marks",
    "debris",
    "scattered_objects",
    "traffic_load",
    "weather",
    "red_region",
]

# Five ordinal levels shared by debris, scattered objects, traffic load and the
# red-region cue (paper Table 2: levels 1/3/5/7/9 -> 0.00/0.25/0.50/0.75/1.00).
LEVEL_SCORES: Dict[int, float] = {0: 0.00, 1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00}

LEVEL_NAMES: Dict[str, List[str]] = {
    "debris": ["none", "light", "moderate", "heavy", "extensive"],
    "scattered_objects": ["none", "1-2 small", "3-5", "6+ or large", "obstructive"],
    "traffic_load": ["empty", "light", "moderate", "high congestion", "gridlock"],
    "red_region": ["none", "small spots", "noticeable patches", "large pool", "multiple large pools"],
}

# Seven weather classes (paper Section 4.4). Clear -> 1.00, stormy -> 0.00.
WEATHER_SCORES: Dict[str, float] = {
    "clear": 1.00,
    "partly_cloudy": 0.78,
    "cloudy": 0.56,
    "foggy": 0.33,
    "rainy": 0.22,
    "snowy": 0.11,
    "stormy": 0.00,
}
WEATHER_CLASSES: List[str] = list(WEATHER_SCORES.keys())


@dataclass
class LevelThresholds:
    """Rules that turn raw detector measurements into ordinal levels 0-4.

    Each list holds the four lower bounds for levels 1, 2, 3 and 4. The default
    values are illustrative; calibrate them on your own annotated data and
    report the values you use.
    """

    # debris: fraction of image area covered by debris boxes
    debris_area: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.04, 0.10])
    # scattered objects: number of detected objects (an obstructive object forces level 4)
    objects_count: List[float] = field(default_factory=lambda: [1, 3, 6, 10])
    # traffic load: number of tracked vehicles in the frame / clip
    vehicle_count: List[float] = field(default_factory=lambda: [1, 6, 15, 30])
    # red region: fraction of image area flagged by the red-region heuristic
    red_area: List[float] = field(default_factory=lambda: [0.0005, 0.003, 0.01, 0.03])
    # tire marks: fraction of image area covered by tire-mark boxes -> saturates at this value
    tire_area_saturation: float = 0.05


def value_to_level(value: float, bounds: List[float]) -> int:
    """Return the ordinal level (0-4) for a measurement given four lower bounds."""
    level = 0
    for i, b in enumerate(bounds, start=1):
        if value >= b:
            level = i
    return level


def level_to_score(level: int) -> float:
    if level not in LEVEL_SCORES:
        raise ValueError(f"level must be 0-4, got {level}")
    return LEVEL_SCORES[level]


def weather_to_score(weather_class: str) -> float:
    key = weather_class.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in WEATHER_SCORES:
        raise ValueError(f"unknown weather class '{weather_class}'")
    return WEATHER_SCORES[key]


def tire_marks_to_score(area_fraction: float, saturation: float) -> float:
    """Continuous score: 0 = no tire marks, 1 = prominent tire marks."""
    if area_fraction <= 0:
        return 0.0
    return float(min(1.0, area_fraction / saturation))
