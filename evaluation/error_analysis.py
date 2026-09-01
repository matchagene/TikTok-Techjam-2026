"""Extract representative false positives, false negatives and instability cases."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from data.sid_dataset import SIDManifestDataset
from evaluation.conditions import benchmark_conditions


def _safe_token(value: object) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return token[:100] or "sample"


def analyze_predictions(
    predictions: pd.DataFrame,
    *,
    threshold: float = 0.5,
    top_k: int = 5,
) -> pd.DataFrame:
    required = {"image_id", "label", "p_fake", "condition_id", "corruption", "severity", "model_id"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction CSV missing columns: {sorted(missing)}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    clean = predictions[predictions["corruption"] == "clean"].copy()
    if clean.empty:
        raise ValueError("Predictions contain no clean condition")

    fps = clean[(clean["label"] == 0) & (clean["p_fake"] >= threshold)].copy()
    fps = fps.sort_values("p_fake", ascending=False).head(top_k)
    fps["case_type"] = "false_positive"
    fps["clean_p_fake"] = fps["p_fake"]
    fps["score_delta"] = 0.0
    fps["decision_flip"] = False

    fns = clean[(clean["label"] == 1) & (clean["p_fake"] < threshold)].copy()
    fns = fns.sort_values("p_fake", ascending=True).head(top_k)
    fns["case_type"] = "false_negative"
    fns["clean_p_fake"] = fns["p_fake"]
    fns["score_delta"] = 0.0
    fns["decision_flip"] = False

    distorted = predictions[predictions["corruption"] != "clean"].copy()
    clean_lookup = clean[["image_id", "p_fake"]].rename(columns={"p_fake": "clean_p_fake"})
    shifts = distorted.merge(clean_lookup, on="image_id", how="inner", validate="many_to_one")
    shifts["score_delta"] = shifts["p_fake"] - shifts["clean_p_fake"]
    shifts["abs_score_delta"] = shifts["score_delta"].abs()
    shifts["decision_flip"] = (
        (shifts["p_fake"] >= threshold) != (shifts["clean_p_fake"] >= threshold)
    )
    shifts = shifts.sort_values(
        ["decision_flip", "abs_score_delta"],
        ascending=[False, False],
    ).drop_duplicates(subset=["image_id"]).head(top_k)
    shifts["case_type"] = "transformation_shift"

    frames = [frame for frame in (fps, fns, shifts) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    cases = pd.concat(frames, ignore_index=True, sort=False)
    cases.insert(0, "case_rank", cases.groupby("case_type").cumcount() + 1)
    return cases


def export_case_assets(
    cases: pd.DataFrame,
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Save clean images and, for instability cases, their exact transformed view."""

    if cases.empty:
        return cases
    dataset = SIDManifestDataset(manifest_path)
    index_by_id = {str(row["image_id"]): i for i, row in dataset.frame.iterrows()}
    condition_by_id = {condition.condition_id: condition for condition in benchmark_conditions(include_clean=True)}
    assets_dir = Path(output_dir) / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for _, row in cases.iterrows():
        record = row.to_dict()
        image_id = str(row["image_id"])
        if image_id not in index_by_id:
            raise KeyError(f"Prediction image_id not found in manifest: {image_id}")
        sample = dataset[index_by_id[image_id]]
        clean_image = sample["image"]
        stem = f"{_safe_token(row['case_type'])}_{int(row['case_rank']):02d}_{_safe_token(image_id)}"
        clean_path = assets_dir / f"{stem}_clean.png"
        clean_image.save(clean_path, format="PNG")
        record["clean_asset"] = clean_path.as_posix()
        record["transformed_asset"] = ""

        if row["case_type"] == "transformation_shift":
            condition_id = str(row["condition_id"])
            condition = condition_by_id.get(condition_id)
            if condition is None:
                raise KeyError(f"Unknown benchmark condition: {condition_id}")
            transformed, _seed = condition.apply(clean_image, image_id=image_id)
            transformed_path = assets_dir / f"{stem}_{_safe_token(condition_id)}.png"
            transformed.save(transformed_path, format="PNG")
            record["transformed_asset"] = transformed_path.as_posix()
        records.append(record)

    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/sid_test.csv"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    model_ids = predictions["model_id"].astype(str).unique()
    if len(model_ids) != 1:
        raise ValueError("Error analysis expects predictions from exactly one model")
    model_id = str(model_ids[0])
    output_dir = args.output_dir or Path("results/error_analysis") / model_id
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = analyze_predictions(predictions, threshold=args.threshold, top_k=args.top_k)
    cases = export_case_assets(cases, manifest_path=args.manifest, output_dir=output_dir)
    output_path = output_dir / "cases.csv"
    cases.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")
    if cases.empty:
        print("No threshold errors or transformation shifts were found.")
    else:
        columns = [
            column for column in
            ["case_type", "case_rank", "image_id", "label", "condition_id", "p_fake", "clean_p_fake", "score_delta", "decision_flip"]
            if column in cases.columns
        ]
        print(cases[columns].to_string(index=False))


if __name__ == "__main__":
    main()
