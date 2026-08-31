"""Threshold-free primary metrics plus thresholded diagnostic metrics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def _as_1d(name: str, values: Iterable[Any], *, dtype: Any) -> np.ndarray:
    array = np.asarray(list(values), dtype=dtype).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def compute_binary_metrics(
    labels: Iterable[int],
    p_fake: Iterable[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute canonical fake-positive binary metrics.

    ROC-AUC is the primary model-selection/reporting metric. Accuracy,
    precision, recall and F1 are diagnostics at a documented threshold.
    """

    y_true = _as_1d("labels", labels, dtype=np.int64)
    y_prob = _as_1d("p_fake", p_fake, dtype=np.float64)
    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"labels and p_fake must have the same length, got {len(y_true)} and {len(y_prob)}"
        )
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("labels must contain only 0=real and 1=fake")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("ROC-AUC requires both real and fake examples")
    if not np.isfinite(y_prob).all():
        raise ValueError("p_fake contains NaN or infinite values")
    if ((y_prob < 0.0) | (y_prob > 1.0)).any():
        raise ValueError("p_fake values must lie in [0, 1]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")

    y_pred = (y_prob >= threshold).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "n": int(y_true.size),
        "n_real": int((y_true == 0).sum()),
        "n_fake": int((y_true == 1).sum()),
    }
