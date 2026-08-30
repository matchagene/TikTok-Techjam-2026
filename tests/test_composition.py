import unittest

import numpy as np
from PIL import Image

from augmentations import apply_pipeline
from augmentations.transforms import apply_transform


def make_test_image(width: int = 23, height: int = 19) -> Image.Image:
    """Create a non-uniform synthetic RGB image without external data."""
    y, x = np.mgrid[:height, :width]
    pixels = np.stack(
        (
            (x * 11 + y * 3) % 256,
            (x * 5 + y * 13) % 256,
            (x * 17 + y * 7) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(pixels)


class ApplyPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = make_test_image()

    def assert_same_pixels(self, first: Image.Image, second: Image.Image) -> None:
        self.assertEqual(first.size, second.size)
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))

    def test_empty_pipeline_returns_separate_rgb_copy(self) -> None:
        grayscale = self.image.convert("L")

        result = apply_pipeline(grayscale, [])

        self.assertIsNot(result, grayscale)
        self.assertEqual(result.mode, "RGB")
        self.assert_same_pixels(result, grayscale.convert("RGB"))

    def test_single_step_pipeline_equals_direct_apply_transform(self) -> None:
        pipeline_result = apply_pipeline(
            self.image,
            [("color_jitter", None)],
            seed=123,
        )
        direct_result = apply_transform(
            self.image,
            "color_jitter",
            severity=None,
            seed=123,
        )

        self.assert_same_pixels(pipeline_result, direct_result)

    def test_multi_step_deterministic_pipeline_matches_manual_order(self) -> None:
        pipeline = [
            ("center_crop", 0.8),
            ("resize", 0.5),
            ("jpeg", 70),
        ]

        result = apply_pipeline(self.image, pipeline, seed=42)
        expected = apply_transform(self.image, "center_crop", 0.8, seed=42)
        expected = apply_transform(expected, "resize", 0.5, seed=100)
        expected = apply_transform(expected, "jpeg", 70, seed=200)

        self.assert_same_pixels(result, expected)

    def test_same_stochastic_pipeline_and_seed_is_deterministic(self) -> None:
        pipeline = [("gaussian_noise", 0.05), ("color_jitter", None)]

        first = apply_pipeline(self.image, pipeline, seed=7)
        second = apply_pipeline(self.image, pipeline, seed=7)

        self.assert_same_pixels(first, second)

    def test_different_seeds_change_stochastic_pipeline(self) -> None:
        pipeline = [("gaussian_noise", 0.05), ("color_jitter", None)]

        first = apply_pipeline(self.image, pipeline, seed=7)
        second = apply_pipeline(self.image, pipeline, seed=8)

        self.assertFalse(np.array_equal(np.asarray(first), np.asarray(second)))

    def test_seed_does_not_change_deterministic_only_pipeline(self) -> None:
        pipeline = [("gaussian_blur", 1.0), ("resize", 0.5), ("jpeg", 70)]

        first = apply_pipeline(self.image, pipeline, seed=7)
        second = apply_pipeline(self.image, pipeline, seed=999)

        self.assert_same_pixels(first, second)

    def test_order_is_respected_and_different_order_changes_pixels(self) -> None:
        first_pipeline = [("gaussian_blur", 2.0), ("jpeg", 30)]
        second_pipeline = list(reversed(first_pipeline))

        first = apply_pipeline(self.image, first_pipeline)
        second = apply_pipeline(self.image, second_pipeline)

        self.assertFalse(np.array_equal(np.asarray(first), np.asarray(second)))

    def test_pipeline_does_not_mutate_input(self) -> None:
        original_pixels = np.asarray(self.image).copy()

        result = apply_pipeline(
            self.image,
            [("center_crop", 0.8), ("gaussian_noise", 0.05), ("jpeg", 70)],
            seed=31,
        )

        self.assertIsNot(result, self.image)
        np.testing.assert_array_equal(np.asarray(self.image), original_pixels)

    def test_invalid_pipeline_structure_has_useful_errors(self) -> None:
        invalid_pipelines = (
            ["jpeg"],
            [("jpeg",)],
            [("jpeg", 70, "extra")],
            [42],
        )
        for pipeline in invalid_pipelines:
            with self.subTest(pipeline=pipeline):
                with self.assertRaisesRegex(ValueError, "pipeline step 0"):
                    apply_pipeline(self.image, pipeline)

        for pipeline in (None, "jpeg"):
            with self.subTest(pipeline=pipeline):
                with self.assertRaisesRegex(TypeError, "pipeline must be an ordered sequence"):
                    apply_pipeline(self.image, pipeline)

    def test_invalid_transform_and_severity_errors_propagate(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported transform"):
            apply_pipeline(self.image, [("unknown", None)])
        with self.assertRaisesRegex(ValueError, "Invalid severity"):
            apply_pipeline(self.image, [("jpeg", 80)])
        with self.assertRaisesRegex(ValueError, "Invalid severity"):
            apply_pipeline(self.image, [("center_crop", None)])

    def test_invalid_seed_raises_value_error(self) -> None:
        for seed in (-1, 1.5, True, "42"):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    apply_pipeline(self.image, [], seed=seed)

    def test_center_crop_dimensions_are_preserved_by_composition(self) -> None:
        result = apply_pipeline(self.image, [("center_crop", 0.8)])

        expected_size = (
            int(round(self.image.width * 0.8)),
            int(round(self.image.height * 0.8)),
        )
        self.assertEqual(result.size, expected_size)
        self.assertEqual(result.mode, "RGB")

    def test_pipeline_does_not_modify_global_numpy_rng_state(self) -> None:
        np.random.seed(1234)
        expected = np.random.random(4)
        np.random.seed(1234)

        apply_pipeline(self.image, [("gaussian_noise", 0.05)], seed=99)
        actual = np.random.random(4)

        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
