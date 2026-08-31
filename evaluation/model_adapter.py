"""One canonical adapter for model outputs and fake-probability semantics.

M0 is historically inverted because ``ImageFolder`` alphabetically assigned
``fake=0, real=1``. M1--M4 use the corrected global convention ``fake=1``.
Keeping the conversion here prevents silent probability-direction bugs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

CORRECTED_MODEL_IDS = frozenset({"M1", "M2", "M3", "M4"})
SUPPORTED_MODEL_IDS = frozenset({"M0", *CORRECTED_MODEL_IDS})


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        logits = output
    elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        logits = output[0]
    elif isinstance(output, dict) and isinstance(output.get("logits"), torch.Tensor):
        logits = output["logits"]
    else:
        raise TypeError(
            "Model output must be logits tensor, (logits, ...), or {'logits': tensor}."
        )
    if logits.ndim == 1:
        logits = logits.unsqueeze(1)
    if logits.ndim != 2 or logits.shape[1] != 1:
        raise ValueError(f"Expected binary logits shape [B, 1], got {tuple(logits.shape)}")
    return logits


def logits_to_fake_probability(logits: torch.Tensor, *, model_id: str) -> torch.Tensor:
    """Convert raw binary logits to canonical ``P(fake)`` for M0--M4."""

    if model_id not in SUPPORTED_MODEL_IDS:
        raise ValueError(f"Unsupported model_id {model_id!r}; expected one of {sorted(SUPPORTED_MODEL_IDS)}")
    if logits.ndim == 1:
        logits = logits.unsqueeze(1)
    if logits.ndim != 2 or logits.shape[1] != 1:
        raise ValueError(f"Expected binary logits shape [B, 1], got {tuple(logits.shape)}")

    positive_probability = torch.sigmoid(logits)
    if model_id == "M0":
        # Historical ImageFolder classes: fake=0, real=1, therefore the
        # sigmoid is P(real), not P(fake).
        return 1.0 - positive_probability
    return positive_probability


@dataclass
class ModelAdapter:
    """Thin inference wrapper exposing canonical logits and ``P(fake)``."""

    model: nn.Module
    model_id: str

    def __post_init__(self) -> None:
        if self.model_id not in SUPPORTED_MODEL_IDS:
            raise ValueError(
                f"Unsupported model_id {self.model_id!r}; expected one of {sorted(SUPPORTED_MODEL_IDS)}"
            )

    def predict_logits(self, images: torch.Tensor) -> torch.Tensor:
        return _extract_logits(self.model(images))

    def predict_fake_probability(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.predict_logits(images)
        return logits_to_fake_probability(logits, model_id=self.model_id)
