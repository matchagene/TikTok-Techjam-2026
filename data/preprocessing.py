"""Canonical image preprocessing shared by training and evaluation.

Robustness corruptions must be applied to PIL images *before* these transforms.
The constants below match the pretrained timm DINOv2 ViT-B/14 configuration
used by the historical M0 baseline.
"""

from __future__ import annotations

from typing import Final

from torchvision import transforms

DINOV2_INPUT_SIZE: Final[int] = 518
IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)


def get_dinov2_preprocess() -> transforms.Compose:
    """Return the canonical preprocessing for M0--M4 DINOv2 experiments.

    This function intentionally contains no robustness augmentation. Apply
    JPEG/blur/resize/noise/crop/color-jitter to the PIL image first, then call
    this preprocessing function.
    """

    return transforms.Compose(
        [
            transforms.Resize(
                (DINOV2_INPUT_SIZE, DINOV2_INPUT_SIZE),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
