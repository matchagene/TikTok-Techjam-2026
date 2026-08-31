"""TeleAI-inspired clean/corrupt consistency objectives.

The NTIRE report states that TeleAI used classification + KL prediction
consistency + MSE representation consistency. It does not fully pin down KL
direction/detachment, so this project explicitly uses *symmetric binary KL*
with gradients through both branches as our documented adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PairwiseLossOutput:
    total: torch.Tensor
    classification: torch.Tensor
    prediction: torch.Tensor
    representation: torch.Tensor


def _validate_binary_logits(name: str, logits: torch.Tensor) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if logits.ndim == 1:
        logits = logits.unsqueeze(1)
    if logits.ndim != 2 or logits.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B, 1], got {tuple(logits.shape)}")
    return logits


def _bernoulli_distribution(logits: torch.Tensor, *, eps: float) -> torch.Tensor:
    logits = _validate_binary_logits("logits", logits)
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must lie in (0, 0.5)")
    p_fake = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    return torch.cat((1.0 - p_fake, p_fake), dim=1)


def binary_symmetric_kl(
    clean_logits: torch.Tensor,
    corrupt_logits: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Symmetric KL between Bernoulli clean/corrupt prediction distributions."""

    clean_logits = _validate_binary_logits("clean_logits", clean_logits)
    corrupt_logits = _validate_binary_logits("corrupt_logits", corrupt_logits)
    if clean_logits.shape != corrupt_logits.shape:
        raise ValueError(
            "clean_logits and corrupt_logits must have the same shape, got "
            f"{tuple(clean_logits.shape)} and {tuple(corrupt_logits.shape)}"
        )

    p_clean = _bernoulli_distribution(clean_logits, eps=eps)
    p_corrupt = _bernoulli_distribution(corrupt_logits, eps=eps)
    log_clean = p_clean.log()
    log_corrupt = p_corrupt.log()
    kl_clean_corrupt = (p_clean * (log_clean - log_corrupt)).sum(dim=1)
    kl_corrupt_clean = (p_corrupt * (log_corrupt - log_clean)).sum(dim=1)
    return 0.5 * (kl_clean_corrupt + kl_corrupt_clean).mean()


def representation_mse(
    clean_representation: torch.Tensor,
    corrupt_representation: torch.Tensor,
) -> torch.Tensor:
    """Direct MSE alignment of pre-dropout robust representations."""

    if not isinstance(clean_representation, torch.Tensor) or not isinstance(
        corrupt_representation, torch.Tensor
    ):
        raise TypeError("representations must be torch.Tensor objects")
    if clean_representation.shape != corrupt_representation.shape:
        raise ValueError(
            "clean and corrupt representations must match, got "
            f"{tuple(clean_representation.shape)} and {tuple(corrupt_representation.shape)}"
        )
    if clean_representation.ndim != 2:
        raise ValueError(
            f"representations must have shape [B, D], got {tuple(clean_representation.shape)}"
        )
    return F.mse_loss(clean_representation, corrupt_representation)


def pairwise_training_loss(
    *,
    clean_logits: torch.Tensor,
    corrupt_logits: torch.Tensor,
    labels: torch.Tensor,
    clean_representation: torch.Tensor,
    corrupt_representation: torch.Tensor,
    lambda_pred: float,
    lambda_repr: float,
    clean_consistency_logits: torch.Tensor | None = None,
    corrupt_consistency_logits: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> PairwiseLossOutput:
    """Compute the common M2/M3/M4 objective.

    M2 is represented exactly by ``lambda_pred=lambda_repr=0``. M3/M4 use the
    same classification term plus non-zero consistency weights.
    """

    clean_logits = _validate_binary_logits("clean_logits", clean_logits)
    corrupt_logits = _validate_binary_logits("corrupt_logits", corrupt_logits)
    if clean_logits.shape != corrupt_logits.shape:
        raise ValueError("clean and corrupt logits must have identical shapes")

    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    if labels.shape != clean_logits.shape:
        raise ValueError(
            f"labels must match logits shape {tuple(clean_logits.shape)}, got {tuple(labels.shape)}"
        )
    labels = labels.to(dtype=clean_logits.dtype)
    if lambda_pred < 0.0 or lambda_repr < 0.0:
        raise ValueError("consistency weights must be non-negative")

    clean_bce = F.binary_cross_entropy_with_logits(clean_logits, labels)
    corrupt_bce = F.binary_cross_entropy_with_logits(corrupt_logits, labels)
    classification = 0.5 * (clean_bce + corrupt_bce)

    # Always compute these for logging/diagnostics, even in M2. Multiplication
    # by zero makes the total objective exactly the classification objective.
    if (clean_consistency_logits is None) != (corrupt_consistency_logits is None):
        raise ValueError(
            "clean_consistency_logits and corrupt_consistency_logits must be supplied together"
        )
    prediction_clean = clean_logits if clean_consistency_logits is None else clean_consistency_logits
    prediction_corrupt = corrupt_logits if corrupt_consistency_logits is None else corrupt_consistency_logits
    prediction = binary_symmetric_kl(prediction_clean, prediction_corrupt, eps=eps)
    representation = representation_mse(clean_representation, corrupt_representation)
    total = classification + float(lambda_pred) * prediction + float(lambda_repr) * representation
    return PairwiseLossOutput(
        total=total,
        classification=classification,
        prediction=prediction,
        representation=representation,
    )
