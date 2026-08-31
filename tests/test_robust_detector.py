import unittest

import torch
import torch.nn as nn

from models.robust_detector import RobustDINODetector


class TinyBackbone(nn.Module):
    num_features = 8

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(3, self.num_features)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        pooled = self.pool(x).flatten(1)
        return self.dropout(self.linear(pooled))


class RobustDetectorTests(unittest.TestCase):
    def test_forward_shapes(self):
        model = RobustDINODetector(backbone=TinyBackbone(), hidden_dim=256, freeze_backbone=True)
        logits, representation = model(torch.randn(3, 3, 16, 16), return_features=True)
        self.assertEqual(tuple(logits.shape), (3, 1))
        self.assertEqual(tuple(representation.shape), (3, 256))

    def test_representation_is_before_head_dropout(self):
        model = RobustDINODetector(
            backbone=TinyBackbone(), hidden_dim=16, dropout=0.9, freeze_backbone=True
        )
        model.train()
        x = torch.randn(2, 3, 8, 8)
        _, z1 = model(x, return_features=True)
        _, z2 = model(x, return_features=True)
        # Frozen backbone is forced to eval and representation is pre-dropout.
        torch.testing.assert_close(z1, z2)

    def test_frozen_backbone_has_no_gradients(self):
        model = RobustDINODetector(backbone=TinyBackbone(), hidden_dim=16, freeze_backbone=True)
        logits = model(torch.randn(2, 3, 8, 8))
        logits.sum().backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.backbone.parameters()))
        self.assertIsNotNone(model.projector[0].weight.grad)
        self.assertIsNotNone(model.classifier.weight.grad)

    def test_m1_head_initialization(self):
        model = RobustDINODetector(backbone=TinyBackbone(), hidden_dim=16, freeze_backbone=True)
        state = {
            "head.0.weight": torch.randn_like(model.projector[0].weight),
            "head.0.bias": torch.randn_like(model.projector[0].bias),
            "head.3.weight": torch.randn_like(model.classifier.weight),
            "head.3.bias": torch.randn_like(model.classifier.bias),
        }
        model.initialize_head_from_m1(state)
        torch.testing.assert_close(model.projector[0].weight, state["head.0.weight"])
        torch.testing.assert_close(model.classifier.weight, state["head.3.weight"])


if __name__ == "__main__":
    unittest.main()
