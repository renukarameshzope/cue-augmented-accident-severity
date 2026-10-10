"""Evidence that the model actually uses the cues.

* cue_permutation_importance: drop in macro F1 when one cue column is shuffled
  across the test set (model is not retrained). Complements the retraining
  ablation in scripts/run_ablation.py.
* cue_occlusion: for one sample, change in predicted probability of the
  predicted class when each cue is replaced by its training-set mean.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from .cues.scoring import CUE_NAMES
from .evaluate import classification_metrics


@torch.no_grad()
def _forward(model, images, cues, device, batch_size=64):
    model.eval()
    out = []
    for i in range(0, len(cues), batch_size):
        img = images[i:i + batch_size].to(device) if model.use_image else None
        out.append(torch.softmax(model(img, cues[i:i + batch_size].to(device)), 1).cpu())
    return torch.cat(out).numpy()


def cue_permutation_importance(model, images: torch.Tensor, cues: torch.Tensor, y: np.ndarray,
                               device: str = "cpu", n_repeats: int = 10, seed: int = 42) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    base = classification_metrics(y, _forward(model, images, cues, device))["macro_f1"]
    res = {}
    for j, name in enumerate(CUE_NAMES):
        drops = []
        for _ in range(n_repeats):
            c = cues.clone()
            c[:, j] = c[rng.permutation(len(c)), j]
            drops.append(base - classification_metrics(y, _forward(model, images, c, device))["macro_f1"])
        res[name] = {"mean_drop": float(np.mean(drops)), "std": float(np.std(drops))}
    return res


@torch.no_grad()
def cue_occlusion(model, image: torch.Tensor, cues: torch.Tensor, cue_means: torch.Tensor, device: str = "cpu") -> Dict[str, float]:
    """Per-cue contribution for a single image (positive = cue supports the prediction)."""
    model.eval()
    img = image.unsqueeze(0).to(device) if model.use_image else None
    c0 = cues.unsqueeze(0).to(device)
    p0 = torch.softmax(model(img, c0), 1)[0]
    k = int(p0.argmax())
    contrib = {}
    for j, name in enumerate(CUE_NAMES):
        c = c0.clone()
        c[0, j] = cue_means[j]
        contrib[name] = float(p0[k] - torch.softmax(model(img, c), 1)[0, k])
    return contrib
