"""Stable output naming for checkpoint evaluation runs."""

from __future__ import annotations

from pathlib import Path


def resolve_run_tag(checkpoint: Path, run_tag: str | None = None) -> str:
    """Return a filesystem-safe run identifier.

    By default the checkpoint filename stem is used, so evaluating multiple
    epoch checkpoints cannot overwrite one another.
    """

    tag = checkpoint.stem if run_tag is None else str(run_tag).strip()

    if not tag:
        raise ValueError("run tag must not be empty")
    if tag in {".", ".."} or "/" in tag or "\\" in tag:
        raise ValueError(
            "run tag must be a single filesystem-safe path component"
        )

    return tag


def clean_output_paths(
    *,
    output_root: Path,
    model_id: str,
    run_tag: str,
    dataset_name: str,
) -> tuple[Path, Path]:
    prediction_path = (
        output_root
        / "predictions"
        / model_id
        / run_tag
        / f"{dataset_name}_clean.csv"
    )
    metric_path = (
        output_root
        / "evaluation"
        / model_id
        / run_tag
        / f"{dataset_name}_clean.csv"
    )
    return prediction_path, metric_path


def robustness_output_paths(
    *,
    output_root: Path,
    model_id: str,
    run_tag: str,
    dataset_name: str,
) -> tuple[Path, Path, Path]:
    prediction_path = (
        output_root
        / "predictions"
        / model_id
        / run_tag
        / f"{dataset_name}_robustness.csv"
    )
    by_condition_path = (
        output_root
        / "evaluation"
        / model_id
        / run_tag
        / f"{dataset_name}_by_condition.csv"
    )
    summary_path = (
        output_root
        / "evaluation"
        / model_id
        / run_tag
        / f"{dataset_name}_summary.csv"
    )
    return prediction_path, by_condition_path, summary_path
