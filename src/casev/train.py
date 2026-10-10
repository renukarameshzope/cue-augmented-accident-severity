"""Training loop for the fusion model and its baselines / ablation variants."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import CueFusionNet


@dataclass
class TrainConfig:
    backbone: str = "efficientnet_b1"
    pretrained: bool = True
    n_frozen_blocks: int = 3
    hidden: int = 256
    dropout: float = 0.3
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 20
    image_size: int = 240
    num_workers: int = 2
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_image: bool = True
    use_cues: bool = True
    cue_mask: Optional[List[float]] = None
    class_weights: Optional[List[float]] = None
    early_stopping_patience: int = 0  # 0 = train for all epochs (paper setting)
    extra: Dict = field(default_factory=dict)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: TrainConfig) -> CueFusionNet:
    return CueFusionNet(
        backbone=cfg.backbone,
        pretrained=cfg.pretrained,
        n_frozen_blocks=cfg.n_frozen_blocks,
        hidden=cfg.hidden,
        dropout=cfg.dropout,
        use_image=cfg.use_image,
        use_cues=cfg.use_cues,
        cue_mask=cfg.cue_mask,
    )


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for img, cues, y in loader:
        img = img.to(device) if model.use_image else None
        logits = model(img, cues.to(device))
        probs.append(torch.softmax(logits, 1).cpu())
        labels.append(y)
    return {"proba": torch.cat(probs).numpy(), "y_true": torch.cat(labels).numpy()}


def train_model(cfg: TrainConfig, train_ds, val_ds=None, log=print) -> CueFusionNet:
    set_seed(cfg.seed)
    device = cfg.device
    model = build_model(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
    weight = torch.tensor(cfg.class_weights, device=device) if cfg.class_weights else None
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    g = torch.Generator().manual_seed(cfg.seed)
    tl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, generator=g)
    vl = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers) if val_ds is not None else None

    best_state, best_f1, bad = None, -1.0, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for img, cues, y in tl:
            img = img.to(device) if cfg.use_image else None
            cues, y = cues.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(img, cues), y)
            loss.backward()
            opt.step()
            total += loss.item() * y.size(0)
            n += y.size(0)
        sched.step()
        msg = f"epoch {epoch:02d}  train_loss {total / n:.4f}"
        if vl is not None:
            from .evaluate import classification_metrics

            out = predict(model, vl, device)
            f1 = classification_metrics(out["y_true"], out["proba"])["macro_f1"]
            msg += f"  val_macro_f1 {f1:.4f}"
            if cfg.early_stopping_patience:
                if f1 > best_f1:
                    best_f1, best_state, bad = f1, copy.deepcopy(model.state_dict()), 0
                else:
                    bad += 1
                    if bad >= cfg.early_stopping_patience:
                        log(msg + "  (early stop)")
                        break
        log(msg)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model
