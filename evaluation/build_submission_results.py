"""Aggregate canonical robustness runs into submission-ready comparison CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import pandas as pd

CANONICAL_RUN_TAGS: Mapping[str, str] = {
    "M0": "M0_original_baseline",
    "M1": "M1_corrected_baseline",
    "M2": "M2_augmented",
    "M3": "M3_pairwise",
    "M4": "M4_curriculum",
}
DISPLAY_NAMES: Mapping[str, str] = {
    "M0": "M0 Historical",
    "M1": "M1 Clean Baseline",
    "M2": "M2 Augmented",
    "M3": "M3 Pairwise Robust",
    "M4": "M4 Curriculum",
}


def parse_run_tag_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--run-tag entries must have MODEL=RUN_TAG format")
        model_id, run_tag = value.split("=", 1)
        model_id, run_tag = model_id.strip(), run_tag.strip()
        if model_id not in CANONICAL_RUN_TAGS or not run_tag:
            raise ValueError(f"Invalid run-tag override: {value!r}")
        overrides[model_id] = run_tag
    return overrides


def _paths(
    *,
    output_root: Path,
    model_id: str,
    run_tag: str,
    dataset_name: str,
) -> tuple[Path, Path]:
    base = output_root / "evaluation" / model_id / run_tag
    return (
        base / f"{dataset_name}_summary.csv",
        base / f"{dataset_name}_by_condition.csv",
    )


def build_submission_tables(
    *,
    model_ids: list[str],
    output_root: str | Path = "results",
    dataset_name: str = "SID_internal_test",
    run_tag_overrides: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_root = Path(output_root)
    overrides = dict(run_tag_overrides or {})
    summary_rows = []
    condition_frames = []

    for model_id in model_ids:
        if model_id not in CANONICAL_RUN_TAGS:
            raise ValueError(f"Unsupported model ID: {model_id}")
        run_tag = overrides.get(model_id, CANONICAL_RUN_TAGS[model_id])
        summary_path, condition_path = _paths(
            output_root=output_root,
            model_id=model_id,
            run_tag=run_tag,
            dataset_name=dataset_name,
        )
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"Missing robustness summary for {model_id}: {summary_path}\n"
                "Run evaluation.evaluate_robustness first."
            )
        if not condition_path.is_file():
            raise FileNotFoundError(f"Missing condition metrics for {model_id}: {condition_path}")

        summary = pd.read_csv(summary_path)
        if len(summary) != 1:
            raise ValueError(f"Expected exactly one summary row in {summary_path}")
        row = summary.iloc[0]
        summary_rows.append(
            {
                "Model": DISPLAY_NAMES[model_id],
                "Model ID": model_id,
                "Clean AUC": float(row["clean_auc"]),
                "Robust Pooled AUC": float(row["robust_pooled_auc"]),
                "Mean Condition AUC": float(row["mean_condition_auc"]),
                "Worst-Case AUC": float(row["worst_case_auc"]),
                "Robustness Drop": float(row["robustness_drop"]),
            }
        )

        conditions = pd.read_csv(condition_path).copy()
        conditions.insert(0, "Model", DISPLAY_NAMES[model_id])
        conditions.insert(1, "Model ID", model_id)
        condition_frames.append(conditions)

    return pd.DataFrame(summary_rows), pd.concat(condition_frames, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["M1", "M3"], choices=list(CANONICAL_RUN_TAGS))
    parser.add_argument("--dataset-name", default="SID_internal_test")
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--run-tag",
        action="append",
        default=[],
        metavar="MODEL=RUN_TAG",
        help="Override a canonical run tag; may be supplied multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = parse_run_tag_overrides(args.run_tag)
    comparison, conditions = build_submission_tables(
        model_ids=list(args.models),
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        run_tag_overrides=overrides,
    )
    evaluation_dir = args.output_root / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = evaluation_dir / "model_comparison.csv"
    conditions_path = evaluation_dir / "condition_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    conditions.to_csv(conditions_path, index=False)
    print(comparison.to_string(index=False))
    print(f"\nWrote {comparison_path}")
    print(f"Wrote {conditions_path}")


if __name__ == "__main__":
    main()
