import unittest
from collections import Counter

from augmentations.curriculum import (
    EARLY_CURRICULUM_WEIGHTS,
    FIXED_TRAINING_WEIGHTS,
    LATE_CURRICULUM_WEIGHTS,
    MIDDLE_CURRICULUM_WEIGHTS,
    curriculum_difficulty,
    curriculum_weights,
    epoch_progress,
    fixed_training_difficulty,
)


class CurriculumTests(unittest.TestCase):
    def test_curriculum_early_distribution(self):
        self.assertEqual(dict(curriculum_weights(0.0)), dict(EARLY_CURRICULUM_WEIGHTS))
        self.assertEqual(dict(curriculum_weights(0.32)), dict(EARLY_CURRICULUM_WEIGHTS))

    def test_curriculum_middle_distribution(self):
        self.assertEqual(dict(curriculum_weights(1 / 3)), dict(MIDDLE_CURRICULUM_WEIGHTS))
        self.assertEqual(dict(curriculum_weights(0.5)), dict(MIDDLE_CURRICULUM_WEIGHTS))

    def test_curriculum_late_distribution(self):
        self.assertEqual(dict(curriculum_weights(2 / 3)), dict(LATE_CURRICULUM_WEIGHTS))
        self.assertEqual(dict(curriculum_weights(1.0)), dict(LATE_CURRICULUM_WEIGHTS))

    def test_fixed_m2_m3_distribution(self):
        self.assertEqual(
            dict(FIXED_TRAINING_WEIGHTS), {"mild": 0.4, "medium": 0.4, "severe": 0.2}
        )

    def test_seeded_sampling_is_deterministic(self):
        self.assertEqual(
            fixed_training_difficulty(seed=73), fixed_training_difficulty(seed=73)
        )
        self.assertEqual(
            curriculum_difficulty(progress=0.8, seed=91),
            curriculum_difficulty(progress=0.8, seed=91),
        )

    def test_early_stage_never_samples_severe(self):
        observed = {curriculum_difficulty(progress=0.1, seed=i) for i in range(1000)}
        self.assertNotIn("severe", observed)
        self.assertTrue(observed.issubset({"mild", "medium"}))

    def test_fixed_sampler_approximately_matches_weights(self):
        n = 10000
        counts = Counter(fixed_training_difficulty(seed=i) for i in range(n))
        for name, expected in FIXED_TRAINING_WEIGHTS.items():
            self.assertAlmostEqual(counts[name] / n, expected, delta=0.025)

    def test_epoch_progress_covers_endpoints(self):
        self.assertEqual(epoch_progress(0, 5), 0.0)
        self.assertEqual(epoch_progress(4, 5), 1.0)
        self.assertEqual(epoch_progress(0, 1), 1.0)

    def test_invalid_progress_errors(self):
        for value in (-0.1, 1.1, float("nan"), True, "0.5"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    curriculum_weights(value)


if __name__ == "__main__":
    unittest.main()
