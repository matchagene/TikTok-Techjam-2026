"""Difficulty distributions for fixed M2/M3 training and progressive M4."""

from __future__ import annotations

import math
import operator
import random
from types import MappingProxyType
from typing import Final, Mapping

from .sampler import DIFFICULTIES

FIXED_TRAINING_WEIGHTS: Final[Mapping[str, float]] = MappingProxyType(
    {"mild": 0.40, "medium": 0.40, "severe": 0.20}
)
EARLY_CURRICULUM_WEIGHTS: Final[Mapping[str, float]] = MappingProxyType(
    {"mild": 0.80, "medium": 0.20, "severe": 0.00}
)
MIDDLE_CURRICULUM_WEIGHTS: Final[Mapping[str, float]] = MappingProxyType(
    {"mild": 0.35, "medium": 0.50, "severe": 0.15}
)
LATE_CURRICULUM_WEIGHTS: Final[Mapping[str, float]] = MappingProxyType(
    {"mild": 0.20, "medium": 0.45, "severe": 0.35}
)


def _validate_weights(weights: Mapping[str, float]) -> None:
    if set(weights) != set(DIFFICULTIES):
        raise ValueError(f"weights must contain exactly {DIFFICULTIES}")
    values = [float(weights[name]) for name in DIFFICULTIES]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("difficulty weights must be finite and non-negative")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"difficulty weights must sum to 1.0, got {sum(values)}")


def curriculum_weights(progress: float) -> Mapping[str, float]:
    """Return M4 corruption weights for normalized training progress in [0, 1]."""

    if isinstance(progress, bool) or not isinstance(progress, (int, float)):
        raise ValueError("progress must be a finite number in [0, 1]")
    progress = float(progress)
    if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
        raise ValueError("progress must be a finite number in [0, 1]")
    if progress < 1.0 / 3.0:
        return EARLY_CURRICULUM_WEIGHTS
    if progress < 2.0 / 3.0:
        return MIDDLE_CURRICULUM_WEIGHTS
    return LATE_CURRICULUM_WEIGHTS


def epoch_progress(epoch_index: int, total_epochs: int) -> float:
    """Map a zero-based epoch index to [0, 1], including both endpoints.

    With a single epoch the run is treated as late-stage (progress=1) because
    there is no meaningful progression to schedule.
    """

    if isinstance(epoch_index, bool) or isinstance(total_epochs, bool):
        raise ValueError("epoch_index and total_epochs must be integers")
    try:
        epoch_index = operator.index(epoch_index)
        total_epochs = operator.index(total_epochs)
    except TypeError as exc:
        raise ValueError("epoch_index and total_epochs must be integers") from exc
    if total_epochs <= 0:
        raise ValueError("total_epochs must be > 0")
    if not 0 <= epoch_index < total_epochs:
        raise ValueError("epoch_index must satisfy 0 <= epoch_index < total_epochs")
    if total_epochs == 1:
        return 1.0
    return epoch_index / (total_epochs - 1)


def sample_difficulty(weights: Mapping[str, float], *, seed: int) -> str:
    """Sample one difficulty deterministically from an explicit distribution."""

    _validate_weights(weights)
    if isinstance(seed, bool):
        raise ValueError("seed must be a non-negative integer")
    try:
        seed = operator.index(seed)
    except TypeError as exc:
        raise ValueError("seed must be a non-negative integer") from exc
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")

    rng = random.Random(seed)
    draw = rng.random()
    cumulative = 0.0
    for name in DIFFICULTIES:
        cumulative += float(weights[name])
        if draw < cumulative:
            return name
    # Floating-point rounding fallback; validated weights sum to 1.
    return DIFFICULTIES[-1]


def fixed_training_difficulty(*, seed: int) -> str:
    """M2/M3: use the same 40/40/20 distribution for the entire run."""

    return sample_difficulty(FIXED_TRAINING_WEIGHTS, seed=seed)


def curriculum_difficulty(*, progress: float, seed: int) -> str:
    """M4: sample according to the stage-specific progressive distribution."""

    return sample_difficulty(curriculum_weights(progress), seed=seed)
