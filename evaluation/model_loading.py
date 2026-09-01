"""Centralized loading of M0--M4 checkpoints for evaluation/inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from evaluation.model_adapter import ModelAdapter


def _load_state_dict(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        state: Any = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch <2.0
        state = torch.load(path, map_location="cpu")
    if isinstance(state, Mapping) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint must contain a PyTorch state-dict mapping")
    return state


def build_model_for_id(
    model_id: str,
    *,
    model_name: str = "vit_base_patch14_dinov2.lvd142m",
    pretrained: bool = True,
) -> nn.Module:
    """Construct the architecture associated with a ladder model ID."""

    if model_id in {"M0", "M1"}:
        # Lazy import avoids requiring timm in unit-test-only environments.
        from models.baseline import DINOBaseline

        return DINOBaseline(
            model_name=model_name,
            freeze_backbone=True,
            pretrained=pretrained,
        )
    if model_id in {"M2", "M3", "M4"}:
        from models.robust_detector import RobustDINODetector

        return RobustDINODetector(
            model_name=model_name,
            freeze_backbone=True,
            pretrained=pretrained,
        )
    raise ValueError("model_id must be one of M0, M1, M2, M3, M4")


def load_adapter(
    model_id: str,
    checkpoint: str | Path,
    *,
    device: torch.device,
    model_name: str = "vit_base_patch14_dinov2.lvd142m",
) -> ModelAdapter:
    """Load a strict checkpoint and return the canonical P(fake) adapter."""

    # Checkpoints contain the full backbone state. Avoid a network download
    # during evaluation/inference by constructing the architecture uninitialized.
    model = build_model_for_id(model_id, model_name=model_name, pretrained=False)
    state = _load_state_dict(Path(checkpoint))
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint {checkpoint} is incompatible with {model_id} architecture"
        ) from exc
    model.to(device)
    model.eval()
    return ModelAdapter(model=model, model_id=model_id)
