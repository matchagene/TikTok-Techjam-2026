import numpy as np
from PIL import Image

from evaluation.conditions import benchmark_conditions


def _image():
    arr = np.arange(24 * 24 * 3, dtype=np.uint8).reshape(24, 24, 3)
    return Image.fromarray(arr, mode="RGB")


def test_benchmark_suite_has_expected_conditions():
    conditions = benchmark_conditions()
    assert len(conditions) == 18
    assert conditions[0].condition_id == "clean"
    assert sum(not condition.is_clean for condition in conditions) == 17
    ids = {condition.condition_id for condition in conditions}
    assert {"jpeg_30", "gaussian_blur_2p0", "resize_0p25", "compound_severe"}.issubset(ids)


def test_condition_is_reproducible_per_image():
    condition = next(c for c in benchmark_conditions() if c.condition_id == "gaussian_noise_0p05")
    image = _image()
    first, first_seed = condition.apply(image, image_id="abc")
    second, second_seed = condition.apply(image, image_id="abc")
    assert first_seed == second_seed
    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_stochastic_condition_uses_different_seed_for_different_images():
    condition = next(c for c in benchmark_conditions() if c.condition_id == "color_jitter_fixed")
    assert condition.seed_for("a") != condition.seed_for("b")
