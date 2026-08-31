import unittest

import torch
import torch.nn as nn

from evaluation.model_adapter import ModelAdapter, logits_to_fake_probability


class ConstantModel(nn.Module):
    def __init__(self, value: float, *, tuple_output: bool = False):
        super().__init__()
        self.value = value
        self.tuple_output = tuple_output

    def forward(self, images):
        logits = torch.full((images.shape[0], 1), self.value, dtype=images.dtype)
        if self.tuple_output:
            return logits, torch.zeros(images.shape[0], 256)
        return logits


class ModelAdapterTests(unittest.TestCase):
    def test_m0_probability_is_inverted(self):
        logits = torch.tensor([[2.0], [-2.0]])
        expected = 1.0 - torch.sigmoid(logits)
        actual = logits_to_fake_probability(logits, model_id="M0")
        torch.testing.assert_close(actual, expected)

    def test_m1_to_m4_use_sigmoid_as_fake_probability(self):
        logits = torch.tensor([[2.0], [-2.0]])
        for model_id in ("M1", "M2", "M3", "M4"):
            with self.subTest(model_id=model_id):
                torch.testing.assert_close(
                    logits_to_fake_probability(logits, model_id=model_id),
                    torch.sigmoid(logits),
                )

    def test_adapter_accepts_robust_tuple_output(self):
        adapter = ModelAdapter(ConstantModel(0.0, tuple_output=True), "M3")
        images = torch.zeros(3, 3, 4, 4)
        self.assertEqual(tuple(adapter.predict_logits(images).shape), (3, 1))
        torch.testing.assert_close(
            adapter.predict_fake_probability(images), torch.full((3, 1), 0.5)
        )

    def test_unknown_model_id_rejected(self):
        with self.assertRaises(ValueError):
            ModelAdapter(ConstantModel(0.0), "M9")


if __name__ == "__main__":
    unittest.main()
