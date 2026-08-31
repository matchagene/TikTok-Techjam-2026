import torch
import torch.nn as nn
import timm

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
        # Load pretrained DINOv2 from timm
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # DINOv2 ViT-B outputs a 768-dimensional embedding
        embedding_dim = self.backbone.num_features 
        
        # Classification Head (MLP)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes) # Outputs raw logits (BCELossWithLogits)
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        # Pass through classification head
        logits = self.head(features)
        return logits

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
