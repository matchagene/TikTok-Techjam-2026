"""Reusable inference and aggregation primitives for the robustness benchmark."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evaluation.conditions import BenchmarkCondition, benchmark_conditions
from evaluation.datasets import CorruptedDatasetView
from evaluation.metrics import compute_binary_metrics
from evaluation.model_adapter import ModelAdapter, logits_to_fake_probability

RAW_PREDICTION_COLUMNS = (
    "image_id",
    "label",
    "logit",
    "p_fake",
    "dataset",
    "model_id",
    "condition_id",
    "corruption",
    "severity",
    "seed",
)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


@torch.inference_mode()
def predict_condition(
    adapter: ModelAdapter,
    loader: DataLoader,
    *,
    device: torch.device,
    dataset_name: str,
) -> pd.DataFrame:
    """Run one condition and return row-level predictions, never only metrics."""

    adapter.model.eval()
    rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="robust eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        logits = adapter.predict_logits(images)
        # Convert the already-computed logits instead of forwarding the model
        # a second time. This matters for throughput and avoids accidental
        # stochastic disagreement if a caller forgot model.eval().
        p_fake = logits_to_fake_probability(logits, model_id=adapter.model_id)

        logits_list = logits.squeeze(1).detach().cpu().tolist()
        probs_list = p_fake.squeeze(1).detach().cpu().tolist()
        labels = [int(v) for v in _as_list(batch["label"])]
        image_ids = [str(v) for v in _as_list(batch["image_id"])]
        condition_ids = [str(v) for v in _as_list(batch["condition_id"])]
        corruptions = [str(v) for v in _as_list(batch["corruption"])]
        severities = [str(v) for v in _as_list(batch["severity"])]
        seeds = [int(v) for v in _as_list(batch["seed"])]

        n = len(labels)
        lengths = {
            len(logits_list), len(probs_list), len(image_ids), len(condition_ids),
            len(corruptions), len(severities), len(seeds), n
        }
        if len(lengths) != 1:
            raise RuntimeError("Evaluation batch fields have inconsistent lengths")

        for i in range(n):
            rows.append(
                {
                    "image_id": image_ids[i],
                    "label": labels[i],
                    "logit": float(logits_list[i]),
                    "p_fake": float(probs_list[i]),
                    "dataset": str(dataset_name),
                    "model_id": adapter.model_id,
                    "condition_id": condition_ids[i],
                    "corruption": corruptions[i],
                    "severity": severities[i],
                    "seed": seeds[i],
                }
            )

    frame = pd.DataFrame(rows, columns=RAW_PREDICTION_COLUMNS)
    if frame.empty:
        raise ValueError("Evaluation produced no predictions")
    return frame


def build_condition_loader(
    base_dataset: Dataset,
    preprocess: Any,
    condition: BenchmarkCondition,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    view = CorruptedDatasetView(base_dataset, preprocess, condition)
    return DataLoader(
        view,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=False,
    )


def run_benchmark(
    adapter: ModelAdapter,
    base_dataset: Dataset,
    preprocess: Any,
    *,
    device: torch.device,
    dataset_name: str,
    batch_size: int = 32,
    num_workers: int = 4,
    conditions: Sequence[BenchmarkCondition] | None = None,
) -> pd.DataFrame:
    """Evaluate identical fixed conditions and concatenate raw predictions."""

    selected = tuple(conditions or benchmark_conditions(include_clean=True))
    if not selected:
        raise ValueError("conditions must not be empty")
    frames: list[pd.DataFrame] = []
    for condition in selected:
        loader = build_condition_loader(
            base_dataset,
            preprocess,
            condition,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
        frame = predict_condition(
            adapter,
            loader,
            device=device,
            dataset_name=dataset_name,
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def validate_prediction_frame(predictions: pd.DataFrame) -> None:
    missing = set(RAW_PREDICTION_COLUMNS).difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction frame missing columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("Prediction frame is empty")
    if not predictions["label"].isin([0, 1]).all():
        raise ValueError("Prediction labels must follow 0=real, 1=fake")
    probs = predictions["p_fake"].astype(float).to_numpy()
    if not np.isfinite(probs).all() or ((probs < 0.0) | (probs > 1.0)).any():
        raise ValueError("p_fake must be finite and lie in [0, 1]")


def condition_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute diagnostic metrics separately for each frozen condition."""

    validate_prediction_frame(predictions)
    rows: list[dict[str, Any]] = []
    grouping = ["model_id", "dataset", "condition_id", "corruption", "severity"]
    for keys, group in predictions.groupby(grouping, sort=False, dropna=False):
        metrics = compute_binary_metrics(group["label"], group["p_fake"])
        rows.append(dict(zip(grouping, keys), **metrics))
    return pd.DataFrame(rows)


def summarize_robustness(predictions: pd.DataFrame) -> dict[str, float | int | str]:
    """Compute clean, pooled, mean-condition, worst-condition and drop metrics.

    ``robust_pooled_auc`` pools every *distorted* prediction row.  Mean and
    worst-condition AUC likewise exclude clean.  This keeps clean as a separate
    guardrail and avoids making an easy clean condition inflate robustness.
    """

    validate_prediction_frame(predictions)
    models = predictions["model_id"].astype(str).unique()
    datasets = predictions["dataset"].astype(str).unique()
    if len(models) != 1 or len(datasets) != 1:
        raise ValueError("summarize_robustness expects exactly one model and one dataset")

    clean = predictions[predictions["corruption"] == "clean"]
    distorted = predictions[predictions["corruption"] != "clean"]
    if clean.empty:
        raise ValueError("Prediction frame has no clean condition")
    if distorted.empty:
        raise ValueError("Prediction frame has no distorted conditions")

    clean_auc = float(compute_binary_metrics(clean["label"], clean["p_fake"])["roc_auc"])
    pooled_auc = float(
        compute_binary_metrics(distorted["label"], distorted["p_fake"])["roc_auc"]
    )
    per_condition = condition_metrics(distorted)
    condition_aucs = per_condition["roc_auc"].astype(float)

    return {
        "model_id": str(models[0]),
        "dataset": str(datasets[0]),
        "clean_auc": clean_auc,
        "robust_pooled_auc": pooled_auc,
        "mean_condition_auc": float(condition_aucs.mean()),
        "worst_case_auc": float(condition_aucs.min()),
        "robustness_drop": float(clean_auc - pooled_auc),
        "n_clean_images": int(len(clean)),
        "n_distorted_predictions": int(len(distorted)),
        "n_distorted_conditions": int(per_condition["condition_id"].nunique()),
    }


def save_benchmark_outputs(
    predictions: pd.DataFrame,
    *,
    predictions_path: str | Path,
    by_condition_path: str | Path,
    summary_path: str | Path,
) -> dict[str, float | int | str]:
    """Persist raw evidence first, then recomputable aggregate tables."""

    validate_prediction_frame(predictions)
    predictions_path = Path(predictions_path)
    by_condition_path = Path(by_condition_path)
    summary_path = Path(summary_path)
    for path in (predictions_path, by_condition_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(predictions_path, index=False)
    by_condition = condition_metrics(predictions)
    by_condition.to_csv(by_condition_path, index=False)
    summary = summarize_robustness(predictions)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    return summary
