from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict

import yaml

from .train import TrainConfig


def load_config(path: str | None, **overrides: Any) -> TrainConfig:
    raw: Dict[str, Any] = {}
    if path:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    raw = {**raw.get("train", raw), **{k: v for k, v in overrides.items() if v is not None}}
    names = {f.name for f in fields(TrainConfig)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return TrainConfig(**raw)
