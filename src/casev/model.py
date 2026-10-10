"""Cue-augmented fusion model (paper Section 4.5).

EfficientNet-B1 (ImageNet-pretrained, early layers frozen) produces a global
image embedding. The six cue scores are concatenated with that embedding and the
fused vector goes through fully connected layers to a 3-class softmax
(low / moderate / high).

The same class also builds the baselines and ablation variants used in the
paper:
  * use_cues=False            -> image-only EfficientNet-B1 baseline
  * use_image=False           -> cue-scores-only MLP baseline
  * cue_mask=[1,1,0,1,1,1]    -> "without <cue>" ablation (cue set to zero)
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torchvision import models

CLASS_NAMES = ["low", "moderate", "high"]
# Anchors used to turn class probabilities into a scalar severity score in [0, 1]
SEVERITY_ANCHORS = torch.tensor([0.0, 0.5, 1.0])
# Paper Table 4 thresholds for the scalar score
SCORE_THRESHOLDS = (0.30, 0.60)


def build_backbone(name: str = "efficientnet_b1", pretrained: bool = True):
    """Return (feature_extractor, embedding_dim)."""
    if name == "efficientnet_b1":
        w = models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b1(weights=w)
        dim = m.classifier[1].in_features
        m.classifier = nn.Identity()
    elif name == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=w)
        dim = m.fc.in_features
        m.fc = nn.Identity()
    elif name == "vit_b_16":
        w = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.vit_b_16(weights=w)
        dim = m.heads.head.in_features
        m.heads = nn.Identity()
    else:
        raise ValueError(f"unknown backbone {name}")
    return m, dim


def freeze_early_layers(backbone: nn.Module, name: str, n_frozen_blocks: int) -> None:
    """Freeze the stem and the first ``n_frozen_blocks`` stages of the backbone."""
    if n_frozen_blocks <= 0:
        return
    if name == "efficientnet_b1":
        for block in list(backbone.features.children())[: n_frozen_blocks + 1]:  # +1 = stem
            for p in block.parameters():
                p.requires_grad = False
    elif name == "resnet50":
        stages = [backbone.conv1, backbone.bn1, backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]
        for s in stages[: n_frozen_blocks + 2]:
            for p in s.parameters():
                p.requires_grad = False
    elif name == "vit_b_16":
        for p in backbone.conv_proj.parameters():
            p.requires_grad = False
        for blk in list(backbone.encoder.layers.children())[:n_frozen_blocks]:
            for p in blk.parameters():
                p.requires_grad = False


class CueFusionNet(nn.Module):
    def __init__(
        self,
        num_cues: int = 6,
        num_classes: int = 3,
        backbone: str = "efficientnet_b1",
        pretrained: bool = True,
        n_frozen_blocks: int = 3,
        hidden: int = 256,
        dropout: float = 0.3,
        use_image: bool = True,
        use_cues: bool = True,
        cue_mask: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        if not (use_image or use_cues):
            raise ValueError("at least one of use_image / use_cues must be True")
        self.use_image, self.use_cues = use_image, use_cues
        self.num_cues = num_cues
        emb_dim = 0
        if use_image:
            self.backbone, emb_dim = build_backbone(backbone, pretrained)
            freeze_early_layers(self.backbone, backbone, n_frozen_blocks)
        mask = torch.ones(num_cues) if cue_mask is None else torch.as_tensor(cue_mask, dtype=torch.float32)
        self.register_buffer("cue_mask", mask)
        in_dim = emb_dim + (num_cues if use_cues else 0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, image: Optional[torch.Tensor], cues: Optional[torch.Tensor]) -> torch.Tensor:
        parts = []
        if self.use_image:
            parts.append(self.backbone(image))
        if self.use_cues:
            parts.append(cues * self.cue_mask)
        return self.head(torch.cat(parts, dim=1))


def severity_score(probs: torch.Tensor) -> torch.Tensor:
    """Scalar severity in [0, 1] = expected value of the class probabilities."""
    return probs @ SEVERITY_ANCHORS.to(probs.device)


def score_to_class(score: float) -> int:
    lo, hi = SCORE_THRESHOLDS
    return 0 if score <= lo else (1 if score <= hi else 2)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
