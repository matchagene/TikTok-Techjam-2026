import unittest

import numpy as np
from PIL import Image

from augmentations import apply_pipeline
from augmentations.sampler import OPERATION_COUNT_RANGES, sample_corruption, sample_pipeline


def make_test_image(width: int = 31, height: int = 23) -> Image.Image:
    y, x = np.mgrid[:height, :width]
    pixels = np.stack(
        ((x * 7 + y * 2) % 256, (x * 3 + y * 11) % 256, (x * 13 + y * 5) % 256),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(pixels)


class TrainingSamplerTests(unittest.TestCase):
    def setUp(self):
        self.image = make_test_image()

    def test_same_seed_is_deterministic(self):
        for difficulty in ("mild", "medium", "severe"):
            with self.subTest(difficulty=difficulty):
                first, first_trace = sample_corruption(self.image, difficulty, seed=123)
                second, second_trace = sample_corruption(self.image, difficulty, seed=123)
                self.assertEqual(first_trace, second_trace)
                np.testing.assert_array_equal(np.asarray(first), np.asarray(second))

    def test_different_seeds_can_differ(self):
        traces = [sample_corruption(self.image, "medium", seed=seed)[1] for seed in range(10)]
        serialized = {repr(trace["operations"]) for trace in traces}
        self.assertGreater(len(serialized), 1)

    def test_output_is_rgb(self):
        output, _ = sample_corruption(self.image.convert("L"), "severe", seed=9)
        self.assertEqual(output.mode, "RGB")

    def test_input_not_mutated(self):
        before = np.asarray(self.image).copy()
        sample_corruption(self.image, "severe", seed=99)
        np.testing.assert_array_equal(np.asarray(self.image), before)

    def test_operation_count_ranges(self):
        for difficulty, (minimum, maximum) in OPERATION_COUNT_RANGES.items():
            for seed in range(100):
                pipeline = sample_pipeline(difficulty=difficulty, seed=seed)
                self.assertGreaterEqual(len(pipeline), minimum)
                self.assertLessEqual(len(pipeline), maximum)
                # Transform families are sampled without replacement.
                names = [step[0] for step in pipeline]
                self.assertEqual(len(names), len(set(names)))

    def test_trace_matches_output_pipeline(self):
        output, trace = sample_corruption(self.image, "medium", seed=90125)
        pipeline = tuple((name, severity) for name, severity in trace["operations"])
        expected = apply_pipeline(self.image, pipeline, seed=90125)
        np.testing.assert_array_equal(np.asarray(output), np.asarray(expected))

    def test_invalid_difficulty_errors(self):
        with self.assertRaisesRegex(ValueError, "Invalid difficulty"):
            sample_corruption(self.image, "extreme", seed=1)

    def test_invalid_seed_errors(self):
        for seed in (-1, True, 1.5, "1"):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    sample_corruption(self.image, "mild", seed=seed)


if __name__ == "__main__":
    unittest.main()
