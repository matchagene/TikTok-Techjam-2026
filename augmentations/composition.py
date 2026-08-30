"""Ordered composition for deterministic robustness transformations."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from PIL import Image

from .transforms import apply_transform


PipelineStep: TypeAlias = tuple[str, int | float | None]


def apply_pipeline(
    image: Image.Image,
    pipeline: Sequence[PipelineStep],
    seed: int = 42,
) -> Image.Image:
    """Apply an ordered sequence of robustness transformations.

    Each pipeline position receives a stable local seed. Position zero retains
    the caller's seed so a single-step pipeline is equivalent to a direct
    ``apply_transform`` call; subsequent seeds are derived with
    ``numpy.random.SeedSequence``.

    Args:
        image: Input PIL image. The caller's image is never mutated.
        pipeline: Ordered ``(transform_name, severity)`` pairs.
        seed: Non-negative integer root seed.

    Returns:
        A separate RGB PIL image containing the composed result.

    Raises:
        TypeError: If ``image`` is not a PIL image or ``pipeline`` is not an
            ordered sequence.
        ValueError: If the seed or a pipeline element is invalid. Transform and
            severity errors from ``apply_transform`` propagate unchanged.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")

    validated_seed = _validate_seed(seed)
    steps = _validate_pipeline(pipeline)
    child_seeds = _derive_child_seeds(validated_seed, len(steps))

    result = image.convert("RGB").copy()
    for (transform_name, severity), child_seed in zip(steps, child_seeds):
        result = apply_transform(
            result,
            transform_name,
            severity=severity,
            seed=child_seed,
        )
    return result


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


def _validate_pipeline(pipeline: Sequence[PipelineStep]) -> tuple[PipelineStep, ...]:
    if isinstance(pipeline, (str, bytes)) or not isinstance(pipeline, Sequence):
        raise TypeError("pipeline must be an ordered sequence of (transform_name, severity) pairs")

    validated_steps: list[PipelineStep] = []
    for index, step in enumerate(pipeline):
        if isinstance(step, (str, bytes)) or not isinstance(step, Sequence):
            raise ValueError(
                f"pipeline step {index} must be a (transform_name, severity) pair"
            )
        if len(step) != 2:
            raise ValueError(
                f"pipeline step {index} must contain exactly two items: "
                "(transform_name, severity)"
            )
        transform_name, severity = step
        validated_steps.append((transform_name, severity))
    return tuple(validated_steps)


def _derive_child_seeds(seed: int, count: int) -> tuple[int, ...]:
    if count == 0:
        return ()

    seeds = [seed]
    used_seeds = {seed}
    root_sequence = np.random.SeedSequence(seed)
    for child_sequence in root_sequence.spawn(count - 1):
        child_seed = int(child_sequence.generate_state(1, dtype=np.uint64)[0])
        # A collision is extraordinarily unlikely, but resolving it makes the
        # distinct-per-position contract unconditional.
        while child_seed in used_seeds:
            child_seed += 1
        seeds.append(child_seed)
        used_seeds.add(child_seed)
    return tuple(seeds)
