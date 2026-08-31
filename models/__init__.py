"""Model definitions for Track 5 experiments.

Imports are lazy so utility/unit-test environments do not require ``timm``
unless a pretrained DINO model is actually instantiated.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DINOBaseline", "RobustDINODetector"]


def __getattr__(name: str) -> Any:
    if name == "DINOBaseline":
        from .baseline import DINOBaseline

        return DINOBaseline
    if name == "RobustDINODetector":
        from .robust_detector import RobustDINODetector

        return RobustDINODetector
    raise AttributeError(name)
