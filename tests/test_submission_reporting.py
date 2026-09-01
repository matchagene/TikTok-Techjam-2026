from pathlib import Path

import pandas as pd

from evaluation.build_submission_results import build_submission_tables
from evaluation.error_analysis import analyze_predictions


def _write_eval(root: Path, model_id: str, run_tag: str, clean: float, robust: float):
    directory = root / "evaluation" / model_id / run_tag
    directory.mkdir(parents=True)
    pd.DataFrame([
        {
            "model_id": model_id,
            "dataset": "SID_internal_test",
            "clean_auc": clean,
            "robust_pooled_auc": robust,
            "mean_condition_auc": robust - 0.01,
            "worst_case_auc": robust - 0.05,
            "robustness_drop": clean - robust,
        }
    ]).to_csv(directory / "SID_internal_test_summary.csv", index=False)
    pd.DataFrame([
        {"model_id": model_id, "dataset": "SID_internal_test", "condition_id": "clean", "corruption": "clean", "severity": "none", "roc_auc": clean},
        {"model_id": model_id, "dataset": "SID_internal_test", "condition_id": "jpeg_30", "corruption": "jpeg", "severity": "30", "roc_auc": robust},
    ]).to_csv(directory / "SID_internal_test_by_condition.csv", index=False)


def test_build_submission_tables_for_m1_m3(tmp_path: Path):
    _write_eval(tmp_path, "M1", "M1_corrected_baseline", 0.99, 0.80)
    _write_eval(tmp_path, "M3", "M3_pairwise", 0.98, 0.92)

    comparison, conditions = build_submission_tables(
        model_ids=["M1", "M3"],
        output_root=tmp_path,
    )

    assert comparison["Model ID"].tolist() == ["M1", "M3"]
    assert comparison.loc[1, "Robust Pooled AUC"] == 0.92
    assert set(conditions["Model ID"]) == {"M1", "M3"}


def test_error_analysis_finds_fp_fn_and_transformation_shift():
    rows = [
        {"image_id": "real_bad", "label": 0, "p_fake": 0.9, "condition_id": "clean", "corruption": "clean", "severity": "none", "model_id": "M3"},
        {"image_id": "fake_bad", "label": 1, "p_fake": 0.1, "condition_id": "clean", "corruption": "clean", "severity": "none", "model_id": "M3"},
        {"image_id": "stable", "label": 1, "p_fake": 0.9, "condition_id": "clean", "corruption": "clean", "severity": "none", "model_id": "M3"},
        {"image_id": "real_bad", "label": 0, "p_fake": 0.2, "condition_id": "jpeg_30", "corruption": "jpeg", "severity": "30", "model_id": "M3"},
        {"image_id": "fake_bad", "label": 1, "p_fake": 0.8, "condition_id": "jpeg_30", "corruption": "jpeg", "severity": "30", "model_id": "M3"},
        {"image_id": "stable", "label": 1, "p_fake": 0.88, "condition_id": "jpeg_30", "corruption": "jpeg", "severity": "30", "model_id": "M3"},
    ]
    cases = analyze_predictions(pd.DataFrame(rows), top_k=2)
    assert {"false_positive", "false_negative", "transformation_shift"}.issubset(set(cases["case_type"]))
    shift = cases[cases["case_type"] == "transformation_shift"].iloc[0]
    assert bool(shift["decision_flip"])
