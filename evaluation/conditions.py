"""Deterministic benchmark conditions for Track 5 robustness evaluation.

Training corruptions are intentionally stochastic and live in
``augmentations.sampler``.  This module is the opposite: a frozen evaluation
suite that is identical for M0--M4.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

from PIL import Image

from augmentations.composition import apply_pipeline
from augmentations.policies import COMPOUND_POLICIES, get_compound_policy
from augmentations.transforms import BENCHMARK_SEVERITIES, apply_transform

DEFAULT_BENCHMARK_SEED: Final[int] = 42


def _stable_image_seed(root_seed: int, image_id: str, condition_id: str) -> int:
    """Derive a reproducible per-image seed without Python hash randomisation."""

    payload = f"{int(root_seed)}|{image_id}|{condition_id}".encode("utf-8")
    # Pillow/numpy are happy with a 32-bit non-negative seed and it is easy to
    # serialize in CSV prediction logs.
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


@dataclass(frozen=True)
class BenchmarkCondition:
    """One immutable corruption condition in the frozen robustness benchmark."""

    condition_id: str
    corruption: str
    severity: int | float | str | None
    root_seed: int = DEFAULT_BENCHMARK_SEED
    pipeline_name: str | None = None

    @property
    def is_clean(self) -> bool:
        return self.corruption == "clean"

    def seed_for(self, image_id: str) -> int:
        return _stable_image_seed(self.root_seed, str(image_id), self.condition_id)

    def apply(self, image: Image.Image, *, image_id: str) -> tuple[Image.Image, int]:
        """Apply this condition and return ``(RGB image, actual seed)``."""

        seed = self.seed_for(image_id)
        if self.pipeline_name is not None:
            transformed = apply_pipeline(
                image,
                get_compound_policy(self.pipeline_name),
                seed=seed,
            )
        else:
            transformed = apply_transform(
                image,
                self.corruption,
                severity=None if self.is_clean or self.corruption == "color_jitter" else self.severity,
                seed=seed,
            )
        return transformed.convert("RGB"), seed


def benchmark_conditions(*, include_clean: bool = True) -> tuple[BenchmarkCondition, ...]:
    """Return the canonical, ordered benchmark suite.

    The order is deliberately stable because downstream raw-prediction files
    and figures may rely on it.  Mean/worst robustness metrics should use only
    the non-clean conditions; clean is reported separately.
    """

    conditions: list[BenchmarkCondition] = []
    if include_clean:
        conditions.append(BenchmarkCondition("clean", "clean", None))

    for transform_name in (
        "jpeg",
        "gaussian_blur",
        "resize",
        "gaussian_noise",
        "center_crop",
    ):
        for severity in BENCHMARK_SEVERITIES[transform_name]:
            severity_token = str(severity).replace(".", "p")
            conditions.append(
                BenchmarkCondition(
                    condition_id=f"{transform_name}_{severity_token}",
                    corruption=transform_name,
                    severity=severity,
                )
            )

    conditions.append(
        BenchmarkCondition(
            condition_id="color_jitter_fixed",
            corruption="color_jitter",
            severity="fixed_pm20pct",
        )
    )

    for name in COMPOUND_POLICIES:
        conditions.append(
            BenchmarkCondition(
                condition_id=f"compound_{name}",
                corruption="compound",
                severity=name,
                pipeline_name=name,
            )
        )

    ids = [condition.condition_id for condition in conditions]
    if len(ids) != len(set(ids)):  # defensive invariant
        raise RuntimeError("Benchmark condition IDs must be unique")
    return tuple(conditions)
