import unittest
from types import MappingProxyType

import numpy as np
from PIL import Image

from augmentations import COMPOUND_POLICIES, apply_pipeline, get_compound_policy


EXPECTED_POLICIES = {
    "mild": (
        ("resize", 0.5),
        ("jpeg", 90),
    ),
    "medium": (
        ("center_crop", 0.8),
        ("resize", 0.5),
        ("jpeg", 70),
    ),
    "severe": (
        ("center_crop", 0.8),
        ("resize", 0.25),
        ("gaussian_blur", 1.0),
        ("jpeg", 30),
    ),
}


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


class CompoundPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = make_test_image()

    def assert_same_pixels(self, first: Image.Image, second: Image.Image) -> None:
        self.assertEqual(first.size, second.size)
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))

    def test_exactly_three_canonical_policy_names_exist(self) -> None:
        self.assertEqual(set(COMPOUND_POLICIES), {"mild", "medium", "severe"})
        self.assertEqual(len(COMPOUND_POLICIES), 3)

    def test_policies_exactly_match_agreed_ordered_pipelines(self) -> None:
        self.assertEqual(COMPOUND_POLICIES, EXPECTED_POLICIES)
        for name, expected in EXPECTED_POLICIES.items():
            with self.subTest(name=name):
                self.assertEqual(get_compound_policy(name), expected)

    def test_every_policy_can_be_applied(self) -> None:
        for name in COMPOUND_POLICIES:
            with self.subTest(name=name):
                result = apply_pipeline(self.image, get_compound_policy(name), seed=42)
                self.assertIsInstance(result, Image.Image)
                self.assertEqual(result.mode, "RGB")

    def test_same_policy_and_seed_is_deterministic(self) -> None:
        for name, policy in COMPOUND_POLICIES.items():
            with self.subTest(name=name):
                first = apply_pipeline(self.image, policy, seed=73)
                second = apply_pipeline(self.image, policy, seed=73)
                self.assert_same_pixels(first, second)

    def test_output_dimensions_follow_crop_behavior(self) -> None:
        cropped_size = (
            int(round(self.image.width * 0.8)),
            int(round(self.image.height * 0.8)),
        )
        expected_sizes = {
            "mild": self.image.size,
            "medium": cropped_size,
            "severe": cropped_size,
        }

        for name, expected_size in expected_sizes.items():
            with self.subTest(name=name):
                result = apply_pipeline(self.image, get_compound_policy(name))
                self.assertEqual(result.size, expected_size)

    def test_unknown_policy_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown compound policy"):
            get_compound_policy("extreme")

    def test_canonical_mapping_and_pipelines_are_immutable(self) -> None:
        self.assertIsInstance(COMPOUND_POLICIES, MappingProxyType)

        with self.assertRaises(TypeError):
            COMPOUND_POLICIES["mild"] = ()
        with self.assertRaises(TypeError):
            COMPOUND_POLICIES["mild"][0] = ("jpeg", 30)
        with self.assertRaises(TypeError):
            COMPOUND_POLICIES["mild"][0][1] = 30


if __name__ == "__main__":
    unittest.main()
