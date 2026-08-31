import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from data.sid_dataset import SIDManifestDataset, SIDPreprocessedDataset, sid_to_binary_label


class SIDDatasetTests(unittest.TestCase):
    def test_label_mapping_is_explicit(self):
        self.assertEqual(sid_to_binary_label(0), 0)
        self.assertEqual(sid_to_binary_label(1), 1)
        self.assertIsNone(sid_to_binary_label(2))
        with self.assertRaises(ValueError):
            sid_to_binary_label(3)

    def test_manifest_dataset_returns_canonical_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "sample.png"
            Image.new("RGB", (10, 8), (12, 34, 56)).save(image_path)
            manifest = root / "manifest.csv"
            pd.DataFrame([
                {
                    "image_id": "abc",
                    "cached_path": "sample.png",
                    "source_split": "train",
                    "sid_label": 1,
                    "binary_label": 1,
                    "width": 10,
                    "height": 8,
                }
            ]).to_csv(manifest, index=False)

            dataset = SIDManifestDataset(manifest, project_root=root)
            sample = dataset[0]
            self.assertEqual(sample["label"], 1)
            self.assertEqual(sample["image_id"], "abc")
            self.assertEqual(sample["source_split"], "train")
            self.assertEqual(sample["image"].mode, "RGB")

    def test_tampered_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (2, 2)).save(root / "x.png")
            manifest = root / "manifest.csv"
            pd.DataFrame([
                {
                    "image_id": "tampered",
                    "cached_path": "x.png",
                    "source_split": "train",
                    "sid_label": 2,
                    "binary_label": 1,
                }
            ]).to_csv(manifest, index=False)
            with self.assertRaisesRegex(ValueError, "tampered"):
                SIDManifestDataset(manifest, project_root=root)

    def test_preprocessed_wrapper_only_changes_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (2, 2), (1, 2, 3)).save(root / "x.png")
            manifest = root / "manifest.csv"
            pd.DataFrame([
                {
                    "image_id": "real",
                    "cached_path": "x.png",
                    "source_split": "validation",
                    "sid_label": 0,
                    "binary_label": 0,
                }
            ]).to_csv(manifest, index=False)
            base = SIDManifestDataset(manifest, project_root=root)
            wrapped = SIDPreprocessedDataset(base, lambda _im: torch.zeros(3, 4, 4))
            sample = wrapped[0]
            self.assertEqual(tuple(sample["image"].shape), (3, 4, 4))
            self.assertEqual(sample["label"], 0)
            self.assertEqual(sample["image_id"], "real")


if __name__ == "__main__":
    unittest.main()
