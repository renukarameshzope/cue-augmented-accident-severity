"""Red-region cue: a colour heuristic for possible blood / injury evidence.

Strongly red pixels are isolated by suppressing the blue and green channels,
then cleaned with morphological opening and closing. The output is the fraction
of the image covered by red blobs and the number of large blobs.

Known limitation (stated in the paper): the heuristic also responds to red
vehicles, brake lights and signs. Treat it as a weak cue and test it in the
ablation.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RedRegionResult:
    area_fraction: float
    n_blobs: int
    n_large_blobs: int
    mask: np.ndarray


def red_region(
    image_bgr: np.ndarray,
    red_min: int = 120,
    dominance: float = 1.6,
    kernel_size: int = 5,
    min_blob_frac: float = 2e-4,
    large_blob_frac: float = 5e-3,
) -> RedRegionResult:
    """Detect strongly red regions in a BGR image.

    A pixel is red when R >= red_min and R >= dominance * max(G, B).
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("expected a BGR image of shape (H, W, 3)")
    img = image_bgr.astype(np.float32)
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    mask = ((r >= red_min) & (r >= dominance * np.maximum(g, b))).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    h, w = mask.shape
    total = float(h * w)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] / total if n > 1 else np.array([])
    keep = areas >= min_blob_frac
    clean = np.zeros_like(mask)
    if n > 1:
        labels = cv2.connectedComponents(mask, connectivity=8)[1]
        for idx in np.where(keep)[0]:
            clean[labels == idx + 1] = 255
    return RedRegionResult(
        area_fraction=float(areas[keep].sum()) if areas.size else 0.0,
        n_blobs=int(keep.sum()),
        n_large_blobs=int((areas >= large_blob_frac).sum()) if areas.size else 0,
        mask=clean,
    )
