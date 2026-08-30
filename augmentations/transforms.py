"""Deterministic, single-image robustness transformations.

This module deliberately operates only on PIL images. Model-specific tensor
conversion, resizing, and normalization belong to the model preprocessing
pipeline and are outside the scope of these transformations.
"""

from __future__ import annotations

import io
import operator
import random
from types import MappingProxyType
from typing import Final, Mapping

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


BENCHMARK_SEVERITIES: Final[Mapping[str, tuple[int | float, ...]]] = MappingProxyType(
    {
        "jpeg": (90, 70, 50, 30),
        "gaussian_blur": (0.5, 1.0, 2.0),
        "resize": (0.5, 0.25),
        "gaussian_noise": (0.02, 0.05, 0.10),
        "center_crop": (0.8,),
    }
)

_FIXED_TRANSFORMS: Final[tuple[str, ...]] = (
    "clean",
    "color_jitter",
)
SUPPORTED_TRANSFORMS: Final[tuple[str, ...]] = (
    *_FIXED_TRANSFORMS,
    *BENCHMARK_SEVERITIES,
)

_COLOR_JITTER_RANGE: Final[tuple[float, float]] = (0.8, 1.2)


def apply_transform(
    image: Image.Image,
    transform: str,
    severity: int | float | None = None,
    seed: int = 42,
) -> Image.Image:
    """Apply one canonical robustness transform to a single PIL image.

    Args:
        image: Input PIL image in any mode. It is converted to RGB without
            modifying the caller's object.
        transform: One of ``SUPPORTED_TRANSFORMS``.
        severity: Required for transforms listed in ``BENCHMARK_SEVERITIES``
            and disallowed for fixed transforms.
        seed: Non-negative integer used by noise and color jitter only.

    Returns:
        A new RGB PIL image.

    Raises:
        TypeError: If ``image`` is not a PIL image.
        ValueError: If the transform, severity, or stochastic seed is invalid.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if not isinstance(transform, str) or transform not in SUPPORTED_TRANSFORMS:
        raise ValueError(
            f"Unsupported transform {transform!r}. Expected one of {SUPPORTED_TRANSFORMS}."
        )

    _validate_severity(transform, severity)

    # The explicit copy guarantees that even the clean operation never returns
    # or mutates the caller's image object.
    rgb_image = image.convert("RGB").copy()

    if transform == "clean":
        return rgb_image
    if transform == "jpeg":
        return _jpeg(rgb_image, int(severity))
    if transform == "gaussian_blur":
        return rgb_image.filter(ImageFilter.GaussianBlur(radius=float(severity)))
    if transform == "resize":
        return _downsample_then_upscale(rgb_image, float(severity))
    if transform == "gaussian_noise":
        return _gaussian_noise(rgb_image, float(severity), _validate_seed(seed))
    if transform == "color_jitter":
        return _color_jitter(rgb_image, _validate_seed(seed))
    if transform == "center_crop":
        return _center_crop(rgb_image, float(severity))

    # All supported names are handled above. This is defensive against future
    # edits that add a name without adding its implementation.
    raise ValueError(f"Transform {transform!r} has no implementation.")


def _validate_severity(transform: str, severity: int | float | None) -> None:
    if transform in BENCHMARK_SEVERITIES:
        allowed = BENCHMARK_SEVERITIES[transform]
        if isinstance(severity, bool) or severity not in allowed:
            raise ValueError(
                f"Invalid severity {severity!r} for {transform!r}. "
                f"Expected one of {allowed}."
            )
        return

    if severity is not None:
        raise ValueError(f"Transform {transform!r} does not accept a severity.")


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool):
        raise ValueError("seed must be a non-negative integer")
    try:
        seed_value = operator.index(seed)
    except TypeError as exc:
        raise ValueError("seed must be a non-negative integer") from exc
    if seed_value < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed_value


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    # Fix the benchmark to conventional 4:2:0 chroma subsampling for reproducibility.
    image.save(
        buffer,
        format="JPEG",
        quality=quality,
        subsampling=2,
        optimize=False,
        progressive=False,
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def _downsample_then_upscale(image: Image.Image, factor: float) -> Image.Image:
    original_width, original_height = image.size
    downsampled_size = (
        max(1, round(original_width * factor)),
        max(1, round(original_height * factor)),
    )
    downsampled = image.resize(downsampled_size, Image.Resampling.BICUBIC)
    return downsampled.resize(image.size, Image.Resampling.BICUBIC)


def _gaussian_noise(image: Image.Image, sigma: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    noisy_pixels = np.clip(pixels + rng.normal(0.0, sigma, pixels.shape), 0.0, 1.0)
    quantized = np.rint(noisy_pixels * 255.0).astype(np.uint8)
    return Image.fromarray(quantized)


def _color_jitter(image: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    minimum, maximum = _COLOR_JITTER_RANGE
    brightness = rng.uniform(minimum, maximum)
    contrast = rng.uniform(minimum, maximum)
    saturation = rng.uniform(minimum, maximum)

    jittered = ImageEnhance.Brightness(image).enhance(brightness)
    jittered = ImageEnhance.Contrast(jittered).enhance(contrast)
    return ImageEnhance.Color(jittered).enhance(saturation).convert("RGB")


def _center_crop(image: Image.Image, fraction: float) -> Image.Image:
    width, height = image.size
    crop_width = int(round(width * fraction))
    crop_height = int(round(height * fraction))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))
