import unittest
from pathlib import Path

from training.run_robust_experiment import load_config, validate_config


class RobustConfigTests(unittest.TestCase):
    def test_project_configs_encode_controlled_ablation(self):
        m2 = load_config(Path("configs/M2_augmented.yaml"))
        m3 = load_config(Path("configs/M3_pairwise.yaml"))
        m4 = load_config(Path("configs/M4_curriculum.yaml"))

        self.assertEqual(m2["training"]["pair_mode"], "fixed")
        self.assertEqual(m3["training"]["pair_mode"], "fixed")
        self.assertEqual(m4["training"]["pair_mode"], "curriculum")
        self.assertEqual(m2["loss"]["lambda_pred"], 0.0)
        self.assertEqual(m2["loss"]["lambda_repr"], 0.0)
        self.assertEqual(m3["loss"]["lambda_pred"], m4["loss"]["lambda_pred"])
        self.assertEqual(m3["loss"]["lambda_repr"], m4["loss"]["lambda_repr"])

        # All non-intervention training knobs stay equal across M2/M3/M4.
        for key in (
            "epochs",
            "batch_size",
            "eval_batch_size",
            "num_workers",
            "head_learning_rate",
            "backbone_learning_rate",
            "weight_decay",
            "grad_clip_norm",
        ):
            self.assertEqual(m2["training"][key], m3["training"][key], key)
            self.assertEqual(m3["training"][key], m4["training"][key], key)

    def test_m2_cannot_accidentally_enable_consistency(self):
        config = load_config(Path("configs/M2_augmented.yaml"))
        config["loss"]["lambda_pred"] = 0.5
        with self.assertRaisesRegex(ValueError, "M2"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
