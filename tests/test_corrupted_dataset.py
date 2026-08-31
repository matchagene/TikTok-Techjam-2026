import torch
from PIL import Image
from torch.utils.data import Dataset

from evaluation.conditions import BenchmarkCondition
from evaluation.datasets import CorruptedDatasetView


class Base(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"image": Image.new("RGB", (8, 8), (100, 110, 120)), "label": 1,
                "image_id": "x", "source_split": "validation"}


def test_corruption_happens_before_preprocess_and_metadata_is_batchable():
    seen = {}

    def preprocess(image):
        seen["mode"] = image.mode
        seen["size"] = image.size
        return torch.zeros(3, 4, 4)

    view = CorruptedDatasetView(Base(), preprocess, BenchmarkCondition("clean", "clean", None))
    sample = view[0]
    assert seen == {"mode": "RGB", "size": (8, 8)}
    assert sample["image"].shape == (3, 4, 4)
    assert sample["severity"] == "none"
    assert isinstance(sample["seed"], int)
