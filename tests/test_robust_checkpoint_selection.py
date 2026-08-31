import unittest

from evaluation.select_robust_checkpoint import select_candidate


class RobustCheckpointSelectionTests(unittest.TestCase):
    def test_robust_auc_is_primary_selection_metric(self):
        candidates = [
            {
                "epoch": 1,
                "robust_pooled_auc": 0.91,
                "clean_auc": 0.99,
            },
            {
                "epoch": 2,
                "robust_pooled_auc": 0.93,
                "clean_auc": 0.95,
            },
        ]

        selected = select_candidate(candidates)
        self.assertEqual(selected["epoch"], 2)

    def test_clean_auc_breaks_exact_robust_tie(self):
        candidates = [
            {
                "epoch": 1,
                "robust_pooled_auc": 0.93,
                "clean_auc": 0.96,
            },
            {
                "epoch": 2,
                "robust_pooled_auc": 0.93,
                "clean_auc": 0.98,
            },
        ]

        selected = select_candidate(candidates)
        self.assertEqual(selected["epoch"], 2)

    def test_earlier_epoch_breaks_complete_tie(self):
        candidates = [
            {
                "epoch": 3,
                "robust_pooled_auc": 0.93,
                "clean_auc": 0.98,
            },
            {
                "epoch": 2,
                "robust_pooled_auc": 0.93,
                "clean_auc": 0.98,
            },
        ]

        selected = select_candidate(candidates)
        self.assertEqual(selected["epoch"], 2)

    def test_empty_candidate_list_is_rejected(self):
        with self.assertRaises(ValueError):
            select_candidate([])


if __name__ == "__main__":
    unittest.main()
