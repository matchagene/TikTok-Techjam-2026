"""Canonical compound redistribution benchmark policies.

This module defines policy data only. Transform execution remains the
responsibility of :func:`augmentations.composition.apply_pipeline`.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping, TypeAlias


CompoundPolicy: TypeAlias = tuple[tuple[str, int | float], ...]

COMPOUND_POLICIES: Final[Mapping[str, CompoundPolicy]] = MappingProxyType(
    {
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
)


def get_compound_policy(name: str) -> CompoundPolicy:
    """Return a canonical compound policy by name.

    Raises:
        ValueError: If ``name`` is not one of the canonical policy names.
    """
    if not isinstance(name, str) or name not in COMPOUND_POLICIES:
        available = ", ".join(COMPOUND_POLICIES)
        raise ValueError(
            f"Unknown compound policy {name!r}. Expected one of: {available}."
        )
    return COMPOUND_POLICIES[name]
