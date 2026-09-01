from __future__ import annotations

import torch
import torch.nn as nn

class DINOBaseline(nn.Module):
    def __init__(
        self,
        model_name='vit_base_patch14_dinov2.lvd142m',
        num_classes=1,
        freeze_backbone=True,
        hidden_dim=256,
        dropout=0.3,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        # Import timm only when a pretrained model is actually constructed.
        # This keeps lightweight utilities/tests importable on minimal environments.
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError(
                "timm is required to construct the pretrained DINOv2 backbone"
            ) from exc
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        self._backbone_frozen = bool(freeze_backbone)
        if self._backbone_frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
                
        # DINOv2 ViT-B outputs a 768-dimensional embedding
        embedding_dim = self.backbone.num_features 
        
        # Classification Head (MLP)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes) # Outputs raw logits (BCELossWithLogits)
        )

    def train(self, mode=True):
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        # Pass through classification head
        logits = self.head(features)
        return logits

    def unfreeze_backbone(self):
        self._backbone_frozen = False
        for param in self.backbone.parameters():
            param.requires_grad = True
