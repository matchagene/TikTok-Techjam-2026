import unittest

import torch

from losses.consistency import binary_symmetric_kl, pairwise_training_loss, representation_mse


class ConsistencyLossTests(unittest.TestCase):
    def test_prediction_consistency_zero_for_identical_logits(self):
        logits = torch.tensor([[0.2], [-1.1], [2.3]])
        self.assertAlmostEqual(binary_symmetric_kl(logits, logits).item(), 0.0, places=7)

    def test_representation_consistency_zero_for_identical_features(self):
        features = torch.randn(4, 256)
        self.assertAlmostEqual(representation_mse(features, features).item(), 0.0, places=7)

    def test_symmetric_kl_is_symmetric(self):
        a = torch.tensor([[-2.0], [0.3], [1.7]])
        b = torch.tensor([[1.2], [-0.8], [0.5]])
        self.assertAlmostEqual(
            binary_symmetric_kl(a, b).item(), binary_symmetric_kl(b, a).item(), places=7
        )

    def test_lambda_zero_total_equals_classification(self):
        clean_logits = torch.tensor([[0.3], [-0.4]], requires_grad=True)
        corrupt_logits = torch.tensor([[1.0], [-1.2]], requires_grad=True)
        clean_z = torch.randn(2, 4, requires_grad=True)
        corrupt_z = torch.randn(2, 4, requires_grad=True)
        labels = torch.tensor([[1.0], [0.0]])
        losses = pairwise_training_loss(
            clean_logits=clean_logits,
            corrupt_logits=corrupt_logits,
            labels=labels,
            clean_representation=clean_z,
            corrupt_representation=corrupt_z,
            lambda_pred=0.0,
            lambda_repr=0.0,
        )
        torch.testing.assert_close(losses.total, losses.classification)

    def test_default_project_weights_are_composable(self):
        losses = pairwise_training_loss(
            clean_logits=torch.zeros(2, 1),
            corrupt_logits=torch.ones(2, 1),
            labels=torch.tensor([0.0, 1.0]),
            clean_representation=torch.zeros(2, 3),
            corrupt_representation=torch.ones(2, 3),
            lambda_pred=0.5,
            lambda_repr=0.25,
        )
        expected = losses.classification + 0.5 * losses.prediction + 0.25 * losses.representation
        torch.testing.assert_close(losses.total, expected)


if __name__ == "__main__":
    unittest.main()
