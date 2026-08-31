"""Shared M2/M3/M4 optimization engine.

Using one engine is important scientifically: M2 and M3 should differ only in
consistency weights, while M3 and M4 should differ only in the corruption
schedule supplied by the paired dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses.consistency import PairwiseLossOutput, pairwise_training_loss


@dataclass(frozen=True)
class PairwiseEpochMetrics:
    classification: float
    prediction: float
    representation: float
    total: float
    n_samples: int


def paired_forward(
    model: nn.Module,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Efficiently forward clean+corrupt views as one doubled batch."""

    if clean.shape != corrupt.shape:
        raise ValueError(
            f"clean and corrupt batches must match, got {tuple(clean.shape)} and {tuple(corrupt.shape)}"
        )
    if clean.ndim != 4:
        raise ValueError(f"Expected image batches [B,C,H,W], got {tuple(clean.shape)}")
    batch_size = clean.shape[0]
    combined = torch.cat((clean, corrupt), dim=0)
    output = model(combined, return_features=True)
    if not isinstance(output, (tuple, list)) or len(output) != 2:
        raise TypeError("Robust model must return (logits, representation) when return_features=True")
    logits, representations = output
    clean_logits, corrupt_logits = logits[:batch_size], logits[batch_size:]
    clean_repr, corrupt_repr = representations[:batch_size], representations[batch_size:]
    return clean_logits, corrupt_logits, clean_repr, corrupt_repr


def compute_pairwise_batch_loss(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    device: torch.device,
    lambda_pred: float,
    lambda_repr: float,
) -> PairwiseLossOutput:
    clean = batch["clean"].to(device, non_blocking=True)
    corrupt = batch["corrupt"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True).float().view(-1, 1)
    clean_logits, corrupt_logits, clean_repr, corrupt_repr = paired_forward(
        model, clean, corrupt
    )
    return pairwise_training_loss(
        clean_logits=clean_logits,
        corrupt_logits=corrupt_logits,
        labels=labels,
        clean_representation=clean_repr,
        corrupt_representation=corrupt_repr,
        lambda_pred=lambda_pred,
        lambda_repr=lambda_repr,
    )


def train_pairwise_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    lambda_pred: float,
    lambda_repr: float,
    grad_clip_norm: float | None = None,
) -> PairwiseEpochMetrics:
    model.train()
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be > 0 when supplied")

    sums = {"classification": 0.0, "prediction": 0.0, "representation": 0.0, "total": 0.0}
    n_samples = 0
    for batch in tqdm(loader, desc="paired train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        losses = compute_pairwise_batch_loss(
            model,
            batch,
            device=device,
            lambda_pred=lambda_pred,
            lambda_repr=lambda_repr,
        )
        if not torch.isfinite(losses.total):
            raise FloatingPointError(
                "Non-finite pairwise loss encountered: "
                f"cls={losses.classification.item()} pred={losses.prediction.item()} "
                f"repr={losses.representation.item()} total={losses.total.item()}"
            )
        losses.total.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        count = int(batch["label"].shape[0])
        n_samples += count
        sums["classification"] += float(losses.classification.detach()) * count
        sums["prediction"] += float(losses.prediction.detach()) * count
        sums["representation"] += float(losses.representation.detach()) * count
        sums["total"] += float(losses.total.detach()) * count

    if n_samples == 0:
        raise ValueError("Pairwise DataLoader produced zero samples")
    return PairwiseEpochMetrics(
        classification=sums["classification"] / n_samples,
        prediction=sums["prediction"] / n_samples,
        representation=sums["representation"] / n_samples,
        total=sums["total"] / n_samples,
        n_samples=n_samples,
    )
