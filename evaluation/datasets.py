"""Evaluation-only dataset views that apply a frozen corruption before preprocessing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from evaluation.conditions import BenchmarkCondition


class CorruptedDatasetView(Dataset):
    """Apply exactly one :class:`BenchmarkCondition` to a canonical PIL dataset.

    ``base_dataset`` must follow the team sample contract and return ``image``
    as a PIL image plus ``label`` and ``image_id``.  The transformation happens
    before model preprocessing, preventing invalid operations such as JPEG on
    normalized tensors.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        preprocess: Callable[[Image.Image], Any],
        condition: BenchmarkCondition,
    ) -> None:
        self.base_dataset = base_dataset
        self.preprocess = preprocess
        self.condition = condition

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        sample = dict(self.base_dataset[index])
        image = sample.get("image")
        if not isinstance(image, Image.Image):
            raise TypeError("CorruptedDatasetView requires base_dataset['image'] to be PIL.Image")
        image_id = str(sample["image_id"])
        transformed, actual_seed = self.condition.apply(image, image_id=image_id)
        return {
            "image": self.preprocess(transformed),
            "label": int(sample["label"]),
            "image_id": image_id,
            "source_split": str(sample.get("source_split", "unknown")),
            "condition_id": self.condition.condition_id,
            "corruption": self.condition.corruption,
            # Avoid None because PyTorch's default collate cannot batch None.
            "severity": "none" if self.condition.severity is None else str(self.condition.severity),
            "seed": int(actual_seed),
        }
