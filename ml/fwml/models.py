"""Model architectures for on-device fish classification.

Architecture choice is treated as an experiment, not an assumption; see
`docs/MODEL.md` for the comparison table this module is used to produce.
What is fixed is the *interface*: every model here exposes

    logits_species, logits_genus, logits_family, embedding

so that the training loop, the exporter and the on-device engine do not change
when the backbone does.

Hierarchical heads
------------------
The species head is the product feature. The genus and family heads exist for
two reasons that matter more than a small accuracy gain:

1. **Honest fallback.** When species-level evidence is weak the app should say
   "some kind of *Sebastes*" rather than guess a species. That claim is far more
   defensible when a head was actually trained to make it than when it is
   synthesised by summing species probabilities - although we measure both, and
   `evaluate.py` reports the comparison, because summation is free.
2. **Gradient signal for the tail.** A species with 40 images contributes little
   on its own, but it also trains its genus and family, and that shared signal
   measurably helps the backbone learn features that transfer to the rare
   classes.

The auxiliary losses are weighted well below the species loss (see
`TrainConfig.genus_loss_weight`), so they shape features without dominating.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Backbones we evaluate. All are available in torchvision without downloading
#: anything exotic, and all export cleanly to ONNX.
BACKBONES = {
    # name: (torchvision factory, feature dim, rough param count in millions)
    "mobilenet_v3_large": ("mobilenet_v3_large", 960, 5.5),
    "mobilenet_v3_small": ("mobilenet_v3_small", 576, 2.5),
    "efficientnet_b0": ("efficientnet_b0", 1280, 5.3),
    "efficientnet_b1": ("efficientnet_b1", 1280, 7.8),
    "convnext_tiny": ("convnext_tiny", 768, 28.6),
    "resnet50": ("resnet50", 2048, 25.6),
}


@dataclass
class ModelSpec:
    backbone: str = "mobilenet_v3_large"
    num_species: int = 1000
    num_genera: int = 0
    num_families: int = 0
    embedding_dim: int = 256
    dropout: float = 0.2
    pretrained: bool = True

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class FishNetModel(nn.Module):
    """Backbone + pooled embedding + species/genus/family heads."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.backbone, feat_dim = _build_backbone(spec.backbone, spec.pretrained)
        self.feature_dim = feat_dim

        # A bottleneck embedding gives us three things at once: a smaller
        # classifier matrix (important at ~2,000 classes on a phone), an
        # L2-normalisable vector for nearest-prototype and few-shot work, and a
        # place to put dropout that is not the backbone.
        self.embed = nn.Sequential(
            nn.Linear(feat_dim, spec.embedding_dim),
            nn.BatchNorm1d(spec.embedding_dim),
            nn.Hardswish(inplace=True),
        )
        self.dropout = nn.Dropout(spec.dropout)
        self.species_head = nn.Linear(spec.embedding_dim, spec.num_species)
        self.genus_head = (
            nn.Linear(spec.embedding_dim, spec.num_genera) if spec.num_genera else None
        )
        self.family_head = (
            nn.Linear(spec.embedding_dim, spec.num_families) if spec.num_families else None
        )

    def forward(self, x: torch.Tensor):
        f = self.backbone(x)
        if f.ndim > 2:
            f = torch.flatten(F.adaptive_avg_pool2d(f, 1), 1)
        e = self.embed(f)
        d = self.dropout(e)
        out = {
            "species": self.species_head(d),
            "embedding": e,
        }
        if self.genus_head is not None:
            out["genus"] = self.genus_head(d)
        if self.family_head is not None:
            out["family"] = self.family_head(d)
        return out


class ExportWrapper(nn.Module):
    """Inference-only view of the model, for ONNX export.

    Returns exactly the tensors the on-device engine consumes, with fixed
    names, and emits **logits** rather than probabilities: temperature scaling
    happens on-device using the value in the pack manifest, so the same exported
    graph can be re-calibrated without re-exporting.
    """

    def __init__(self, model: FishNetModel, with_embedding: bool = True) -> None:
        super().__init__()
        self.model = model
        self.with_embedding = with_embedding

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        if self.with_embedding:
            return out["species"], F.normalize(out["embedding"], dim=1)
        return out["species"]


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    import torchvision.models as tvm

    if name not in BACKBONES:
        raise SystemExit(f"unknown backbone {name!r}; have {sorted(BACKBONES)}")
    factory_name, feat_dim, _ = BACKBONES[name]
    weights = "DEFAULT" if pretrained else None
    net = getattr(tvm, factory_name)(weights=weights)

    # Strip the classifier; we supply our own heads. Each family stores it
    # differently, hence the explicit handling rather than a generic hack.
    if name.startswith("mobilenet_v3"):
        net.classifier = nn.Identity()
        # mobilenet_v3 pools inside forward and returns (N, feat_dim)
        return net, feat_dim
    if name.startswith("efficientnet"):
        net.classifier = nn.Identity()
        return net, feat_dim
    if name == "convnext_tiny":
        net.classifier = nn.Sequential(
            tvm.convnext.LayerNorm2d(feat_dim, eps=1e-6), nn.Flatten(1)
        )
        return net, feat_dim
    if name == "resnet50":
        net.fc = nn.Identity()
        return net, feat_dim
    raise SystemExit(f"no head-stripping rule for {name!r}")


def build(spec: ModelSpec) -> FishNetModel:
    return FishNetModel(spec)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class HierarchicalLoss(nn.Module):
    """Species loss plus down-weighted genus and family auxiliaries.

    Label smoothing is applied to the species head only. Smoothing mainly buys
    calibration, and we calibrate explicitly with temperature scaling later, so
    a large value would just blur the signal; 0.1 is the standard compromise.
    """

    def __init__(
        self,
        genus_weight: float = 0.2,
        family_weight: float = 0.1,
        label_smoothing: float = 0.1,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.genus_weight = genus_weight
        self.family_weight = family_weight
        self.species_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, weight=class_weights
        )
        self.aux_loss = nn.CrossEntropyLoss()

    def forward(self, out: dict, species: torch.Tensor,
                genus: torch.Tensor | None = None,
                family: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        loss = self.species_loss(out["species"], species)
        parts = {"species": float(loss.detach())}

        if genus is not None and "genus" in out and self.genus_weight > 0:
            lg = self.aux_loss(out["genus"], genus)
            loss = loss + self.genus_weight * lg
            parts["genus"] = float(lg.detach())
        if family is not None and "family" in out and self.family_weight > 0:
            lf = self.aux_loss(out["family"], family)
            loss = loss + self.family_weight * lf
            parts["family"] = float(lf.detach())
        return loss, parts
