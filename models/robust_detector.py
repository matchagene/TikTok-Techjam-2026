"""DINOv2 detector exposing a pre-dropout representation for consistency loss."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

DEFAULT_DINOV2_MODEL = "vit_base_patch14_dinov2.lvd142m"


class RobustDINODetector(nn.Module):
    """DINOv2 + 256-d robust representation + binary fake classifier.

    ``backbone`` injection exists primarily for tests and future backbone
    experiments. Production M2--M4 should leave it as ``None`` and use the
    configured pretrained timm DINOv2 backbone.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_DINOV2_MODEL,
        freeze_backbone: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        backbone: nn.Module | None = None,
        embedding_dim: int | None = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        if backbone is None:
            try:
                import timm
            except ImportError as exc:  # pragma: no cover - real runtime only
                raise RuntimeError(
                    "timm is required to construct the pretrained DINOv2 backbone"
                ) from exc
            backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        self.model_name = model_name
        self.backbone = backbone
        if embedding_dim is None:
            embedding_dim = getattr(backbone, "num_features", None)
        if embedding_dim is None:
            raise ValueError(
                "embedding_dim must be supplied when the injected backbone has no num_features"
            )
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)

        self.projector = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_dim, 1)
        self._backbone_frozen = False
        self.set_backbone_frozen(freeze_backbone)

    @property
    def backbone_frozen(self) -> bool:
        return self._backbone_frozen

    def set_backbone_frozen(self, frozen: bool) -> None:
        self._backbone_frozen = bool(frozen)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not self._backbone_frozen
        if self._backbone_frozen:
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen foundation models should remain deterministic feature
        # extractors. This prevents train-mode stochasticity (e.g. drop path)
        # from being penalized as if it were corruption sensitivity.
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        backbone_features = self.backbone(x)
        if isinstance(backbone_features, (tuple, list)):
            if not backbone_features:
                raise ValueError("Backbone returned an empty tuple/list")
            backbone_features = backbone_features[0]
        if not isinstance(backbone_features, torch.Tensor):
            raise TypeError("Backbone must return a tensor or tuple/list whose first item is a tensor")
        if backbone_features.ndim != 2:
            raise ValueError(
                f"Expected pooled backbone features [B, D], got {tuple(backbone_features.shape)}"
            )

        representation = self.projector(backbone_features)
        logits = self.classifier(self.dropout(representation))
        if return_features:
            return logits, representation
        return logits

    def initialize_head_from_m1(self, checkpoint: str | Path | Mapping[str, Any]) -> None:
        """Copy compatible M1 MLP head weights into projector/classifier.

        M1 keys from ``DINOBaseline`` are:
        ``head.0.{weight,bias}`` and ``head.3.{weight,bias}``.
        """

        if isinstance(checkpoint, (str, Path)):
            state: Any = torch.load(checkpoint, map_location="cpu", weights_only=True)
        else:
            state = checkpoint
        if isinstance(state, Mapping) and "model_state_dict" in state:
            state = state["model_state_dict"]
        if not isinstance(state, Mapping):
            raise TypeError("M1 checkpoint must contain a state-dict mapping")

        mapping = {
            "head.0.weight": self.projector[0].weight,
            "head.0.bias": self.projector[0].bias,
            "head.3.weight": self.classifier.weight,
            "head.3.bias": self.classifier.bias,
        }
        missing = [key for key in mapping if key not in state]
        if missing:
            raise KeyError(f"M1 checkpoint is missing expected head keys: {missing}")

        with torch.no_grad():
            for source_key, target_parameter in mapping.items():
                source = state[source_key]
                if not isinstance(source, torch.Tensor):
                    raise TypeError(f"Checkpoint value {source_key} is not a tensor")
                if source.shape != target_parameter.shape:
                    raise ValueError(
                        f"Shape mismatch for {source_key}: checkpoint {tuple(source.shape)} "
                        f"vs target {tuple(target_parameter.shape)}"
                    )
                target_parameter.copy_(source)

    def unfreeze_last_blocks(self, num_blocks: int = 2) -> None:
        """Optional stretch mode: train only the last transformer blocks + norm."""

        if num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            raise AttributeError("Backbone does not expose a 'blocks' sequence")
        if num_blocks > len(blocks):
            raise ValueError(f"Requested {num_blocks} blocks but backbone has {len(blocks)}")

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for block in blocks[-num_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        norm = getattr(self.backbone, "norm", None)
        if norm is not None:
            for parameter in norm.parameters():
                parameter.requires_grad = True
        self._backbone_frozen = False
