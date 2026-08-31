import unittest

import torch
from PIL import Image

from data.preprocessing import DINOV2_INPUT_SIZE, get_dinov2_preprocess


class PreprocessingTests(unittest.TestCase):
    def test_output_shape_and_finiteness(self):
        image = Image.new("RGB", (100, 80), (127, 128, 129))
        tensor = get_dinov2_preprocess()(image)
        self.assertEqual(tuple(tensor.shape), (3, DINOV2_INPUT_SIZE, DINOV2_INPUT_SIZE))
        self.assertTrue(torch.isfinite(tensor).all())


if __name__ == "__main__":
    unittest.main()
