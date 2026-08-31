"""Select the canonical M2/M3/M4 checkpoint using validation robustness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from evaluation.output_paths import resolve_run_tag, robustness_output_paths


VALID_MODEL_IDS = frozenset({"M2", "M3", "M4"})
VALIDATION_DATASET_NAME = "SID_dev_val"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    return value


def _load_json_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping in {path}")
    return value


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose robust AUC first, clean AUC second, earlier epoch third."""

    if not candidates:
        raise ValueError("No checkpoint candidates supplied")

    for candidate in candidates:
        for key in ("epoch", "robust_pooled_auc", "clean_auc"):
            if key not in candidate:
                raise ValueError(f"Candidate missing {key!r}: {candidate}")

        robust_auc = float(candidate["robust_pooled_auc"])
        clean_auc = float(candidate["clean_auc"])
        if not math.isfinite(robust_auc) or not math.isfinite(clean_auc):
            raise ValueError("Candidate AUC values must be finite")

    return max(
        candidates,
        key=lambda row: (
            float(row["robust_pooled_auc"]),
            float(row["clean_auc"]),
            -int(row["epoch"]),
        ),
    )


def collect_candidates(
    *,
    model_id: str,
    metadata: dict[str, Any],
    output_root: Path,
) -> list[dict[str, Any]]:
    epoch_checkpoints = metadata.get("epoch_checkpoints")
    if not isinstance(epoch_checkpoints, list) or not epoch_checkpoints:
        raise ValueError(
            "Training metadata must contain a non-empty epoch_checkpoints list"
        )

    candidates: list[dict[str, Any]] = []

    for item in epoch_checkpoints:
        if not isinstance(item, dict):
            raise ValueError("Each epoch_checkpoints entry must be a mapping")

        epoch = int(item["epoch"])
        checkpoint = Path(str(item["path"]))
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Epoch {epoch} checkpoint not found: {checkpoint}"
            )

        run_tag = resolve_run_tag(checkpoint)
        _, _, summary_path = robustness_output_paths(
            output_root=output_root,
            model_id=model_id,
            run_tag=run_tag,
            dataset_name=VALIDATION_DATASET_NAME,
        )

        if not summary_path.is_file():
            raise FileNotFoundError(
                f"Missing robust validation summary for epoch {epoch}: "
                f"{summary_path}\n"
                "Evaluate every epoch checkpoint on sid_val.csv before selection."
            )

        frame = pd.read_csv(summary_path)
        if len(frame) != 1:
            raise ValueError(
                f"Expected exactly one summary row in {summary_path}, "
                f"found {len(frame)}"
            )

        row = frame.iloc[0]
        if str(row["model_id"]) != model_id:
            raise ValueError(
                f"{summary_path} model_id={row['model_id']!r}, "
                f"expected {model_id!r}"
            )
        if str(row["dataset"]) != VALIDATION_DATASET_NAME:
            raise ValueError(
                f"{summary_path} dataset={row['dataset']!r}, "
                f"expected {VALIDATION_DATASET_NAME!r}"
            )

        candidates.append(
            {
                "epoch": epoch,
                "checkpoint": str(checkpoint),
                "run_tag": run_tag,
                "summary_path": str(summary_path),
                "clean_auc": float(row["clean_auc"]),
                "robust_pooled_auc": float(row["robust_pooled_auc"]),
                "mean_condition_auc": float(row["mean_condition_auc"]),
                "worst_case_auc": float(row["worst_case_auc"]),
                "robustness_drop": float(row["robustness_drop"]),
            }
        )

    return candidates


def _update_history(
    history_path: Path,
    candidates: list[dict[str, Any]],
) -> None:
    if not history_path.is_file():
        raise FileNotFoundError(f"Training history not found: {history_path}")

    history = pd.read_csv(history_path)
    if "epoch" not in history.columns:
        raise ValueError(f"{history_path} has no epoch column")

    robust_by_epoch = {
        int(row["epoch"]): float(row["robust_pooled_auc"])
        for row in candidates
    }

    history["robust_val_auc"] = history["epoch"].map(robust_by_epoch)

    if history["robust_val_auc"].isna().any():
        missing = history.loc[
            history["robust_val_auc"].isna(), "epoch"
        ].tolist()
        raise ValueError(
            f"Missing robust validation result for history epochs: {missing}"
        )

    history.to_csv(history_path, index=False)


def run(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = _load_mapping(config_path)

    model_id = str(config["experiment"]["model_id"])
    if model_id not in VALID_MODEL_IDS:
        raise ValueError(
            f"Robust checkpoint selection is only for "
            f"{sorted(VALID_MODEL_IDS)}, got {model_id!r}"
        )

    val_manifest = Path(str(config["data"]["val_manifest"]))
    if val_manifest.name != "sid_val.csv":
        raise ValueError(
            "Checkpoint selection must use the SID validation manifest; "
            f"got {val_manifest}"
        )

    canonical_checkpoint = Path(str(config["output"]["checkpoint"]))
    metadata_path = Path(str(config["output"]["metadata"]))
    history_path = Path(str(config["output"]["history"]))

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Training metadata not found: {metadata_path}"
        )

    metadata = _load_json_mapping(metadata_path)
    if str(metadata.get("model_id")) != model_id:
        raise ValueError(
            f"Metadata model_id={metadata.get('model_id')!r}, "
            f"expected {model_id!r}"
        )

    candidates = collect_candidates(
        model_id=model_id,
        metadata=metadata,
        output_root=output_root,
    )
    selected = select_candidate(candidates)

    selected_checkpoint = Path(str(selected["checkpoint"]))
    canonical_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_checkpoint, canonical_checkpoint)

    _update_history(history_path, candidates)

    selection_table = (
        output_root
        / "evaluation"
        / model_id
        / f"{VALIDATION_DATASET_NAME}_checkpoint_selection.csv"
    )
    selection_table.parent.mkdir(parents=True, exist_ok=True)

    ranked = sorted(
        candidates,
        key=lambda row: (
            -float(row["robust_pooled_auc"]),
            -float(row["clean_auc"]),
            int(row["epoch"]),
        ),
    )
    table = pd.DataFrame(ranked)
    table.insert(0, "rank", range(1, len(table) + 1))
    table["selected"] = table["epoch"] == int(selected["epoch"])
    table.to_csv(selection_table, index=False)

    metadata.update(
        {
            "checkpoint_selection_method": (
                "maximum SID validation robust_pooled_auc; "
                "ties broken by higher clean_auc, then earlier epoch"
            ),
            "checkpoint_selection_dataset": VALIDATION_DATASET_NAME,
            "selected_epoch": int(selected["epoch"]),
            "selected_epoch_checkpoint": str(selected_checkpoint),
            "selected_validation_summary": str(
                selected["summary_path"]
            ),
            "selected_robust_val_auc": float(
                selected["robust_pooled_auc"]
            ),
            "selected_clean_val_auc": float(selected["clean_auc"]),
            "selected_mean_condition_auc": float(
                selected["mean_condition_auc"]
            ),
            "selected_worst_case_auc": float(
                selected["worst_case_auc"]
            ),
            "selected_robustness_drop": float(
                selected["robustness_drop"]
            ),
            "selection_table": str(selection_table),
            "selection_completed_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "canonical_checkpoint_sha256": _sha256(
                canonical_checkpoint
            ),
            "checkpoint_selection_note": (
                "Canonical checkpoint selected using SID validation "
                "robustness only. sid_test.csv is not used for model "
                "selection."
            ),
        }
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"{model_id}: selected epoch {selected['epoch']} "
        f"robust_auc={selected['robust_pooled_auc']:.6f} "
        f"clean_auc={selected['clean_auc']:.6f}"
    )
    print(f"Canonical checkpoint: {canonical_checkpoint}")
    print(f"Selection table: {selection_table}")

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, args.output_root)


if __name__ == "__main__":
    main()
