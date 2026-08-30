import random
import unittest

import numpy as np
from PIL import Image

from augmentations.transforms import BENCHMARK_SEVERITIES, apply_transform


def make_test_image(width: int = 23, height: int = 19) -> Image.Image:
    """Create a non-uniform synthetic RGB image without external fixtures."""
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


class ApplyTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = make_test_image()

    def assert_same_pixels(self, first: Image.Image, second: Image.Image) -> None:
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))

    def test_clean_preserves_pixels_but_returns_separate_image(self) -> None:
        result = apply_transform(self.image, "clean")

        self.assertIsNot(result, self.image)
        self.assert_same_pixels(result, self.image)

    def test_every_official_severity_is_accepted(self) -> None:
        for transform, severities in BENCHMARK_SEVERITIES.items():
            for severity in severities:
                with self.subTest(transform=transform, severity=severity):
                    result = apply_transform(self.image, transform, severity)
                    self.assertIsInstance(result, Image.Image)
                    self.assertEqual(result.mode, "RGB")

    def test_missing_and_invalid_severities_raise_value_error(self) -> None:
        invalid_values = {
            "jpeg": 80,
            "gaussian_blur": 1.5,
            "resize": 0.75,
            "gaussian_noise": 0.01,
            "center_crop": 0.75,
        }
        for transform, invalid in invalid_values.items():
            with self.subTest(transform=transform, severity=None):
                with self.assertRaises(ValueError):
                    apply_transform(self.image, transform)
            with self.subTest(transform=transform, severity=invalid):
                with self.assertRaises(ValueError):
                    apply_transform(self.image, transform, invalid)

    def test_fixed_transform_rejects_a_severity(self) -> None:
        for transform in ("clean", "color_jitter"):
            with self.subTest(transform=transform):
                with self.assertRaises(ValueError):
                    apply_transform(self.image, transform, 1)

    def test_unknown_transform_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            apply_transform(self.image, "unknown")

    def test_noise_is_deterministic_for_same_seed(self) -> None:
        first = apply_transform(self.image, "gaussian_noise", 0.05, seed=7)
        second = apply_transform(self.image, "gaussian_noise", 0.05, seed=7)

        self.assert_same_pixels(first, second)

    def test_different_noise_seeds_change_pixels(self) -> None:
        first = apply_transform(self.image, "gaussian_noise", 0.05, seed=7)
        second = apply_transform(self.image, "gaussian_noise", 0.05, seed=8)

        self.assertFalse(np.array_equal(np.asarray(first), np.asarray(second)))

    def test_color_jitter_is_deterministic_for_same_seed(self) -> None:
        first = apply_transform(self.image, "color_jitter", seed=21)
        second = apply_transform(self.image, "color_jitter", seed=21)

        self.assert_same_pixels(first, second)

    def test_different_color_jitter_seeds_change_pixels(self) -> None:
        first = apply_transform(self.image, "color_jitter", seed=21)
        second = apply_transform(self.image, "color_jitter", seed=22)

        self.assertFalse(np.array_equal(np.asarray(first), np.asarray(second)))

    def test_jpeg_is_deterministic(self) -> None:
        first = apply_transform(self.image, "jpeg", 70)
        second = apply_transform(self.image, "jpeg", 70)

        self.assert_same_pixels(first, second)

    def test_blur_preserves_dimensions(self) -> None:
        for sigma in BENCHMARK_SEVERITIES["gaussian_blur"]:
            with self.subTest(sigma=sigma):
                result = apply_transform(self.image, "gaussian_blur", sigma)
                self.assertEqual(result.size, self.image.size)

    def test_resize_returns_original_dimensions(self) -> None:
        for factor in BENCHMARK_SEVERITIES["resize"]:
            with self.subTest(factor=factor):
                result = apply_transform(self.image, "resize", factor)
                self.assertEqual(result.size, self.image.size)

    def test_center_crop_returns_rounded_eighty_percent_dimensions(self) -> None:
        result = apply_transform(self.image, "center_crop", 0.8)

        expected = (
            int(round(self.image.width * 0.8)),
            int(round(self.image.height * 0.8)),
        )
        self.assertEqual(result.size, expected)

    def test_outputs_are_rgb(self) -> None:
        cases = (
            ("clean", None),
            ("jpeg", 90),
            ("gaussian_blur", 0.5),
            ("resize", 0.5),
            ("gaussian_noise", 0.02),
            ("color_jitter", None),
            ("center_crop", 0.8),
        )
        for transform, severity in cases:
            with self.subTest(transform=transform):
                result = apply_transform(self.image, transform, severity)
                self.assertEqual(result.mode, "RGB")

    def test_non_rgb_input_is_converted_to_rgb(self) -> None:
        rgba_image = self.image.convert("RGBA")

        result = apply_transform(rgba_image, "clean")

        self.assertEqual(result.mode, "RGB")

    def test_transform_does_not_mutate_input(self) -> None:
        original_pixels = np.asarray(self.image).copy()
        cases = (
            ("clean", None),
            ("jpeg", 30),
            ("gaussian_blur", 2.0),
            ("resize", 0.25),
            ("gaussian_noise", 0.10),
            ("color_jitter", None),
            ("center_crop", 0.8),
        )
        for transform, severity in cases:
            with self.subTest(transform=transform):
                apply_transform(self.image, transform, severity, seed=11)
                np.testing.assert_array_equal(np.asarray(self.image), original_pixels)

    def test_stochastic_transforms_do_not_modify_global_rng_state(self) -> None:
        np.random.seed(1234)
        expected_numpy = np.random.random(4)
        np.random.seed(1234)
        apply_transform(self.image, "gaussian_noise", 0.05, seed=99)
        actual_numpy = np.random.random(4)
        np.testing.assert_array_equal(actual_numpy, expected_numpy)

        random.seed(1234)
        expected_python = [random.random() for _ in range(4)]
        random.seed(1234)
        apply_transform(self.image, "color_jitter", seed=99)
        actual_python = [random.random() for _ in range(4)]
        self.assertEqual(actual_python, expected_python)


if __name__ == "__main__":
    unittest.main()
