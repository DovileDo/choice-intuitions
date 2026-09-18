"""
ResNet-50 source models: ImageNet, RadImageNet, Ecoset (DVD baseline),
Ecoset (DVD-S, Lu et al. 2026), and a randomly initialised baseline (scratch).

Models from load_source_model() take RGB tensors in [0, 1] (plain ToTensor()
output) and apply their source's own input normalization inside forward(), so
target pipelines use identical transforms for every source.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models.resnet import Bottleneck, ResNet

from intuitions.paths import MODELS_DIR


def _read_torchvision_imagenet(_):
    return ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=False)


RADIMAGENET_MODULES = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}


def _read_radimagenet(path):
    # Keys are "backbone.<i>.<rest>" from nn.Sequential(*resnet50.children()).
    state = torch.load(path, map_location="cpu", weights_only=True)
    renamed = {}
    for key, value in state.items():
        _, idx, rest = key.split(".", 2)
        renamed[f"{RADIMAGENET_MODULES[idx]}.{rest}"] = value
    return renamed


def _read_dvd(path):
    # Saved from DDP + torch.compile, so keys carry "module._orig_mod.".
    state = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
    return {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in state.items()}


@dataclass(frozen=True)
class SourceSpec:
    read: Optional[Callable[[Optional[Path]], dict]]  # None: random initialisation
    checkpoint: Optional[Path]
    mean: tuple
    std: tuple
    bgr: bool
    pretrain_resolution: Optional[int]


SOURCES = {
    "imagenet": SourceSpec(
        _read_torchvision_imagenet, None,
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), bgr=False, pretrain_resolution=224,
    ),
    # RadImageNet was trained on cv2-loaded (BGR) images scaled to [-1, 1].
    "radimagenet": SourceSpec(
        _read_radimagenet, MODELS_DIR / "radimagenet" / "RadImageNet_ResNet50_pytorch.pt",
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), bgr=True, pretrain_resolution=224,
    ),
    # DVD models were trained on RGB in [0, 1] with no mean/std normalization.
    "ecoset_baseline": SourceSpec(
        _read_dvd, MODELS_DIR / "ecoset" / "dvd_baseline_checkpoint_best.pth",
        mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), bgr=False, pretrain_resolution=256,
    ),
    "ecoset_dvd_s": SourceSpec(
        _read_dvd, MODELS_DIR / "ecoset" / "dvd_s_checkpoint_best.pth",
        mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), bgr=False, pretrain_resolution=256,
    ),
    # No pretraining: torchvision's default ResNet-50 initialisation.
    "scratch": SourceSpec(
        None, None,
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), bgr=False, pretrain_resolution=None,
    ),
}


class SourceResNet50(ResNet):
    def __init__(self, spec, num_classes):
        super().__init__(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
        self.bgr = spec.bgr
        self.register_buffer("input_mean", torch.tensor(spec.mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("input_std", torch.tensor(spec.std).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        if self.bgr:
            x = x.flip(1)
        return super().forward((x - self.input_mean) / self.input_std)


def is_pretrained(source):
    return SOURCES[source].read is not None


def read_checkpoint(source):
    """Full pretrained state dict (including the source-task fc head)."""
    spec = SOURCES[source]
    if spec.read is None:
        raise ValueError(f"{source!r} is randomly initialised and has no checkpoint")
    return spec.read(spec.checkpoint)


def load_backbone_state(source):
    return {k: v for k, v in read_checkpoint(source).items() if not k.startswith("fc.")}


def load_source_model(source, num_classes, dropout=0.0):
    """Backbone from `source` (pretrained, or random for "scratch") with a fresh (Dropout +) Linear head."""
    if source not in SOURCES:
        raise ValueError(f"Unknown source {source!r}; choose from {list(SOURCES)}")
    model = SourceResNet50(SOURCES[source], num_classes)
    if is_pretrained(source):
        missing, unexpected = model.load_state_dict(load_backbone_state(source), strict=False)
        if unexpected or set(missing) != {"fc.weight", "fc.bias"}:
            raise RuntimeError(f"{source}: backbone mismatch (missing={missing}, unexpected={unexpected})")

    head = nn.Linear(model.fc.in_features, num_classes)
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)
    model.fc = nn.Sequential(nn.Dropout(p=dropout), head) if dropout > 0 else head
    return model
