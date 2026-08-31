"""Losses used by the robust paired-training experiments."""

from .consistency import (
    PairwiseLossOutput,
    binary_symmetric_kl,
    pairwise_training_loss,
    representation_mse,
)

__all__ = [
    "PairwiseLossOutput",
    "binary_symmetric_kl",
    "pairwise_training_loss",
    "representation_mse",
]
