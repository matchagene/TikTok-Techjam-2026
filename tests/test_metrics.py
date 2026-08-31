import unittest

from evaluation.metrics import compute_binary_metrics


class MetricsTests(unittest.TestCase):
    def test_perfect_ranking(self):
        metrics = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["n_real"], 2)
        self.assertEqual(metrics["n_fake"], 2)

    def test_requires_both_classes(self):
        with self.assertRaisesRegex(ValueError, "both real and fake"):
            compute_binary_metrics([1, 1], [0.7, 0.8])

    def test_rejects_invalid_probabilities(self):
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            compute_binary_metrics([0, 1], [0.2, 1.2])

    def test_rejects_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            compute_binary_metrics([0, 1], [0.2])


if __name__ == "__main__":
    unittest.main()
