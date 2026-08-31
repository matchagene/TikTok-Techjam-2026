import pandas as pd
import pytest

from evaluation.robustness import condition_metrics, summarize_robustness


def _frame():
    rows = []
    # Clean perfect ranking.
    for image_id, label, p in [("r", 0, 0.1), ("f", 1, 0.9)]:
        rows.append(dict(image_id=image_id, label=label, logit=0.0, p_fake=p,
                         dataset="d", model_id="M3", condition_id="clean",
                         corruption="clean", severity="none", seed=1))
    # JPEG perfect ranking.
    for image_id, label, p in [("r", 0, 0.2), ("f", 1, 0.8)]:
        rows.append(dict(image_id=image_id, label=label, logit=0.0, p_fake=p,
                         dataset="d", model_id="M3", condition_id="jpeg_30",
                         corruption="jpeg", severity="30", seed=2))
    # Blur reversed ranking => AUC 0.
    for image_id, label, p in [("r", 0, 0.9), ("f", 1, 0.1)]:
        rows.append(dict(image_id=image_id, label=label, logit=0.0, p_fake=p,
                         dataset="d", model_id="M3", condition_id="blur_2",
                         corruption="gaussian_blur", severity="2.0", seed=3))
    return pd.DataFrame(rows)


def test_condition_metrics_are_separate():
    by_condition = condition_metrics(_frame())
    assert set(by_condition["condition_id"]) == {"clean", "jpeg_30", "blur_2"}


def test_summary_excludes_clean_from_mean_and_worst():
    summary = summarize_robustness(_frame())
    assert summary["clean_auc"] == pytest.approx(1.0)
    assert summary["mean_condition_auc"] == pytest.approx(0.5)
    assert summary["worst_case_auc"] == pytest.approx(0.0)
    assert summary["n_distorted_conditions"] == 2
    # Pooled is computed from all distorted rows, not mean of condition AUCs.
    assert 0.0 <= summary["robust_pooled_auc"] <= 1.0
    assert summary["robustness_drop"] == pytest.approx(
        summary["clean_auc"] - summary["robust_pooled_auc"]
    )
