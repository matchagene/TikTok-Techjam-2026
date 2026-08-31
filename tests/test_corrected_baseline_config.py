import tempfile
import unittest
from pathlib import Path

import yaml

from training.train_corrected_baseline import load_config


def valid_config():
    return {
        "model": {
            "name": "vit_base_patch14_dinov2.lvd142m",
            "freeze_backbone": True,
            "input_size": 518,
            "hidden_dim": 256,
            "dropout": 0.3,
        }
    }


class CorrectedBaselineConfigTests(unittest.TestCase):
    def _load(self, config):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            return load_config(path)

    def test_valid_model_config_is_accepted(self):
        loaded = self._load(valid_config())
        self.assertEqual(loaded["model"]["hidden_dim"], 256)

    def test_noncanonical_input_size_is_rejected(self):
        config = valid_config()
        config["model"]["input_size"] = 224

        with self.assertRaises(ValueError):
            self._load(config)

    def test_nonpositive_hidden_dim_is_rejected(self):
        config = valid_config()
        config["model"]["hidden_dim"] = 0

        with self.assertRaises(ValueError):
            self._load(config)

    def test_invalid_dropout_is_rejected(self):
        config = valid_config()
        config["model"]["dropout"] = 1.0

        with self.assertRaises(ValueError):
            self._load(config)


if __name__ == "__main__":
    unittest.main()
