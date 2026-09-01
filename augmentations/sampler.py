"""Stochastic corruption sampler for M2/M3/M4 training.

This module is intentionally separate from the deterministic benchmark policies.
Evaluation should use ``augmentations.transforms`` / ``augmentations.policies``;
training should use ``sample_corruption`` so the detector does not simply
memorize a small set of fixed benchmark pipelines.
"""

from __future__ import annotations

import operator
import random
from types import MappingProxyType
from typing import Final, Mapping

from PIL import Image

from .composition import PipelineStep, apply_pipeline

Difficulty = str

DIFFICULTIES: Final[tuple[Difficulty, ...]] = ("mild", "medium", "severe")

# Severity choices are a compute-efficient adaptation of the agreed Track 5
# transformation set. ``None`` denotes a fixed-severity transform such as color jitter.
TRAINING_POOLS: Final[Mapping[Difficulty, Mapping[str, tuple[int | float | None, ...]]]] = (
    MappingProxyType(
        {
            "mild": MappingProxyType(
                {
                    "jpeg": (90,),
                    "gaussian_blur": (0.5,),
                    "resize": (0.5,),
                    "gaussian_noise": (0.02,),
                    "center_crop": (0.8,),
                    "color_jitter": (None,),
                }
            ),
            "medium": MappingProxyType(
                {
                    "jpeg": (70, 50),
                    "gaussian_blur": (1.0,),
                    "resize": (0.5,),
                    "gaussian_noise": (0.05,),
                    "center_crop": (0.8,),
                    "color_jitter": (None,),
                }
            ),
            "severe": MappingProxyType(
                {
                    "jpeg": (50, 30),
                    "gaussian_blur": (1.0, 2.0),
                    "resize": (0.25,),
                    "gaussian_noise": (0.10,),
                    "center_crop": (0.8,),
                    "color_jitter": (None,),
                }
            ),
        }
    )
)

OPERATION_COUNT_RANGES: Final[Mapping[Difficulty, tuple[int, int]]] = MappingProxyType(
    {
        "mild": (1, 1),
        "medium": (1, 3),
        "severe": (2, 4),
    }
)


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool):
        raise ValueError("seed must be a non-negative integer")
    try:
        value = operator.index(seed)
    except TypeError as exc:
        raise ValueError("seed must be a non-negative integer") from exc
    if value < 0:
        raise ValueError("seed must be a non-negative integer")
    return value


def _validate_difficulty(difficulty: str) -> str:
    if not isinstance(difficulty, str) or difficulty not in DIFFICULTIES:
        raise ValueError(
            f"Invalid difficulty {difficulty!r}. Expected one of {DIFFICULTIES}."
        )
    return difficulty


def sample_pipeline(*, difficulty: str, seed: int) -> tuple[PipelineStep, ...]:
    """Sample a deterministic corruption pipeline from a difficulty level.

    The same ``difficulty`` and ``seed`` always produce the same ordered
    pipeline. Transformation families are sampled without replacement, so a
    single training view cannot contain duplicate JPEG/blur/etc. operations.
    """

    difficulty = _validate_difficulty(difficulty)
    seed = _validate_seed(seed)
    rng = random.Random(seed)
    pool = TRAINING_POOLS[difficulty]
    minimum, maximum = OPERATION_COUNT_RANGES[difficulty]
    operation_count = rng.randint(minimum, maximum)

    transform_names = list(pool.keys())
    selected_names = rng.sample(transform_names, k=operation_count)
    steps: list[PipelineStep] = []
    for transform_name in selected_names:
        severity = rng.choice(pool[transform_name])
        steps.append((transform_name, severity))
    return tuple(steps)


def sample_corruption(
    image: Image.Image,
    difficulty: str,
    seed: int,
) -> tuple[Image.Image, dict[str, object]]:
    """Apply one stochastic training corruption and return an audit trace.

    Args:
        image: Unprocessed PIL image. The caller's object is never mutated.
        difficulty: ``mild``, ``medium`` or ``severe``.
        seed: Non-negative seed controlling both operation selection and each
            stochastic transform inside the resulting pipeline.

    Returns:
        ``(corrupted_rgb_image, trace)`` where ``trace`` is JSON-serializable.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    difficulty = _validate_difficulty(difficulty)
    seed = _validate_seed(seed)
    pipeline = sample_pipeline(difficulty=difficulty, seed=seed)
    corrupted = apply_pipeline(image, pipeline, seed=seed)
    trace: dict[str, object] = {
        "difficulty": difficulty,
        "operations": [[name, severity] for name, severity in pipeline],
        "seed": seed,
    }
    return corrupted, trace
