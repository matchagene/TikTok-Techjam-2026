import unittest

import torch
import torch.nn as nn

from models.robust_detector import RobustDINODetector
from training.pairwise_engine import compute_pairwise_batch_loss, paired_forward


class TinyBackbone(nn.Module):
    num_features = 6

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, self.num_features)

    def forward(self, x):
        return self.fc(self.pool(x).flatten(1))


class PairwiseEngineTests(unittest.TestCase):
    def setUp(self):
        self.model = RobustDINODetector(
            backbone=TinyBackbone(), hidden_dim=8, dropout=0.0, freeze_backbone=True
        )

    def test_paired_forward_shapes(self):
        clean = torch.randn(4, 3, 8, 8)
        corrupt = torch.randn(4, 3, 8, 8)
        outputs = paired_forward(self.model, clean, corrupt)
        self.assertEqual(tuple(outputs[0].shape), (4, 1))
        self.assertEqual(tuple(outputs[1].shape), (4, 1))
        self.assertEqual(tuple(outputs[2].shape), (4, 8))
        self.assertEqual(tuple(outputs[3].shape), (4, 8))

    def test_identical_views_have_zero_consistency_in_deterministic_model(self):
        clean = torch.randn(4, 3, 8, 8)
        batch = {"clean": clean, "corrupt": clean.clone(), "label": torch.tensor([0, 1, 0, 1])}
        losses = compute_pairwise_batch_loss(
            self.model,
            batch,
            device=torch.device("cpu"),
            lambda_pred=0.5,
            lambda_repr=0.25,
        )
        self.assertAlmostEqual(losses.prediction.item(), 0.0, places=7)
        self.assertAlmostEqual(losses.representation.item(), 0.0, places=7)

    def test_gradient_flow_head_only(self):
        batch = {
            "clean": torch.randn(4, 3, 8, 8),
            "corrupt": torch.randn(4, 3, 8, 8),
            "label": torch.tensor([0, 1, 0, 1]),
        }
        losses = compute_pairwise_batch_loss(
            self.model,
            batch,
            device=torch.device("cpu"),
            lambda_pred=0.5,
            lambda_repr=0.25,
        )
        losses.total.backward()
        self.assertIsNotNone(self.model.projector[0].weight.grad)
        self.assertIsNotNone(self.model.classifier.weight.grad)
        self.assertTrue(all(p.grad is None for p in self.model.backbone.parameters()))


if __name__ == "__main__":
    unittest.main()
