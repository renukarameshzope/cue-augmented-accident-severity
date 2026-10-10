"""Dataset and leakage-safe splitting.

Expected manifest CSV columns:
    image_path   path to the image (absolute or relative to --image-root)
    label        low | moderate | high  (or 0 | 1 | 2)
    group_id     identifier of the source video / scene. Frames from one video
                 MUST share a group_id so that they never appear in both the
                 training and test sets.
    source       (optional) dataset name, e.g. DAD, A3D, CADP
    tire_marks, debris, scattered_objects, traffic_load, weather, red_region
                 cue scores in [0, 1] (produced by scripts/extract_cues.py)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import transforms as T

from .cues.scoring import CUE_NAMES
from .model import CLASS_NAMES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_manifest(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"image_path", "label", "group_id", *CUE_NAMES} - set(df.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    if not pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].astype(str).str.lower().map({n: i for i, n in enumerate(CLASS_NAMES)})
    if df["label"].isna().any():
        raise ValueError("unknown label values in manifest")
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def build_transforms(size: int = 240, train: bool = True):
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.75, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.1),  # no hue jitter: it would distort the red-region evidence
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class SeverityDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: Optional[str] = None, transform=None, load_images: bool = True):
        self.df = df.reset_index(drop=True)
        self.root = Path(image_root) if image_root else None
        self.transform = transform
        self.load_images = load_images
        self.cues = torch.tensor(self.df[CUE_NAMES].values, dtype=torch.float32)
        self.labels = torch.tensor(self.df["label"].values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        img = torch.zeros(1)
        if self.load_images:
            p = Path(self.df.at[i, "image_path"])
            if self.root is not None and not p.is_absolute():
                p = self.root / p
            img = Image.open(p).convert("RGB")
            img = self.transform(img) if self.transform else T.ToTensor()(img)
        return img, self.cues[i], self.labels[i]


def grouped_holdout_split(df: pd.DataFrame, test_size: float = 0.4, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified, group-aware hold-out split (default 60/40 as in the paper)."""
    n_splits = max(2, int(round(1.0 / test_size)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(df, df["label"], groups=df["group_id"]))
    return train_idx, test_idx


def grouped_kfold(df: pd.DataFrame, k: int = 10, seed: int = 42) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    yield from sgkf.split(df, df["label"], groups=df["group_id"])


def assert_no_group_leakage(df: pd.DataFrame, train_idx, test_idx) -> None:
    overlap = set(df.iloc[train_idx]["group_id"]) & set(df.iloc[test_idx]["group_id"])
    if overlap:
        raise AssertionError(f"{len(overlap)} groups appear in both train and test, e.g. {list(overlap)[:3]}")
