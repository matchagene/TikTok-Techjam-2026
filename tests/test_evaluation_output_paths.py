import unittest
from pathlib import Path

from evaluation.output_paths import (
    clean_output_paths,
    resolve_run_tag,
    robustness_output_paths,
)


class EvaluationOutputPathTests(unittest.TestCase):
    def test_checkpoint_stem_is_default_run_tag(self):
        self.assertEqual(
            resolve_run_tag(Path("checkpoints/M2_augmented_epoch03.pth")),
            "M2_augmented_epoch03",
        )

    def test_explicit_run_tag_overrides_checkpoint_stem(self):
        self.assertEqual(
            resolve_run_tag(
                Path("checkpoints/M2_augmented_epoch03.pth"),
                "selected_final",
            ),
            "selected_final",
        )

    def test_invalid_run_tag_is_rejected(self):
        for tag in ("", "..", "epoch/03", r"epoch\03"):
            with self.subTest(tag=tag):
                with self.assertRaises(ValueError):
                    resolve_run_tag(Path("model.pth"), tag)

    def test_clean_paths_are_checkpoint_scoped(self):
        prediction, metrics = clean_output_paths(
            output_root=Path("results"),
            model_id="M2",
            run_tag="M2_augmented_epoch03",
            dataset_name="SID_dev_val",
        )

        self.assertEqual(
            prediction,
            Path(
                "results/predictions/M2/M2_augmented_epoch03/"
                "SID_dev_val_clean.csv"
            ),
        )
        self.assertEqual(
            metrics,
            Path(
                "results/evaluation/M2/M2_augmented_epoch03/"
                "SID_dev_val_clean.csv"
            ),
        )

    def test_robust_paths_are_checkpoint_scoped(self):
        prediction, by_condition, summary = robustness_output_paths(
            output_root=Path("results"),
            model_id="M3",
            run_tag="M3_pairwise_epoch05",
            dataset_name="SID_dev_val",
        )

        base = Path("results/evaluation/M3/M3_pairwise_epoch05")
        self.assertEqual(
            prediction,
            Path(
                "results/predictions/M3/M3_pairwise_epoch05/"
                "SID_dev_val_robustness.csv"
            ),
        )
        self.assertEqual(
            by_condition,
            base / "SID_dev_val_by_condition.csv",
        )
        self.assertEqual(
            summary,
            base / "SID_dev_val_summary.csv",
        )


if __name__ == "__main__":
    unittest.main()
