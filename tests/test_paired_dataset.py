import unittest

import torch
from PIL import Image
from torch.utils.data import Dataset

from training.paired_dataset import PairedSIDDataset, paired_collate_fn, stable_seed


class TinyPILDataset(Dataset):
    def __init__(self, size=5):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {
            "image": Image.new("RGB", (12, 10), (index * 20, 50, 100)),
            "label": index % 2,
            "image_id": f"img-{index}",
            "source_split": "train",
        }


def preprocess(image):
    # Tiny deterministic tensor conversion sufficient for unit tests.
    values = torch.tensor(image.resize((1, 1)).getpixel((0, 0)), dtype=torch.float32) / 255.0
    return values.view(3, 1, 1).expand(3, 4, 4).clone()


class PairedDatasetTests(unittest.TestCase):
    def test_stable_seed_is_repeatable(self):
        self.assertEqual(stable_seed(42, "abc", 1), stable_seed(42, "abc", 1))
        self.assertNotEqual(stable_seed(42, "abc", 1), stable_seed(42, "abc", 2))

    def test_same_epoch_sample_is_deterministic(self):
        dataset = PairedSIDDataset(
            TinyPILDataset(), preprocess, mode="fixed", base_seed=42, total_epochs=5
        )
        first = dataset[1]
        second = dataset[1]
        self.assertEqual(first["trace"], second["trace"])
        torch.testing.assert_close(first["corrupt"], second["corrupt"])

    def test_epoch_changes_corruption_seed(self):
        dataset = PairedSIDDataset(
            TinyPILDataset(), preprocess, mode="fixed", base_seed=42, total_epochs=5
        )
        first = dataset[1]
        dataset.set_epoch(1)
        second = dataset[1]
        self.assertNotEqual(first["trace"]["seed"], second["trace"]["seed"])

    def test_curriculum_early_never_severe_across_examples(self):
        dataset = PairedSIDDataset(
            TinyPILDataset(size=50), preprocess, mode="curriculum", base_seed=42, total_epochs=6
        )
        dataset.set_epoch(0)
        observed = {dataset[i]["trace"]["difficulty"] for i in range(len(dataset))}
        self.assertNotIn("severe", observed)

    def test_custom_collate_preserves_variable_traces(self):
        dataset = PairedSIDDataset(
            TinyPILDataset(), preprocess, mode="fixed", base_seed=42, total_epochs=5
        )
        batch = paired_collate_fn([dataset[0], dataset[1], dataset[2]])
        self.assertEqual(tuple(batch["clean"].shape), (3, 3, 4, 4))
        self.assertEqual(tuple(batch["corrupt"].shape), (3, 3, 4, 4))
        self.assertEqual(tuple(batch["label"].shape), (3,))
        self.assertEqual(len(batch["trace"]), 3)


if __name__ == "__main__":
    unittest.main()
