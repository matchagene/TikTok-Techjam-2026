"""Clean/corrupt paired dataset used by M2, M3 and M4."""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Literal, Mapping, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from augmentations.curriculum import (
    curriculum_difficulty,
    epoch_progress,
    fixed_training_difficulty,
)
from augmentations.sampler import sample_corruption

PairMode = Literal["fixed", "curriculum"]


def stable_seed(*parts: object, modulo: int = 2**63 - 1) -> int:
    """Stable cross-process seed; unlike Python ``hash()``, never changes per run."""

    if modulo <= 0:
        raise ValueError("modulo must be > 0")
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % modulo


class PairedSIDDataset(Dataset):
    """Wrap canonical PIL samples into deterministic clean/corrupt tensor pairs."""

    def __init__(
        self,
        base_dataset: Dataset,
        preprocess: Callable[[Image.Image], torch.Tensor],
        *,
        mode: PairMode,
        base_seed: int,
        total_epochs: int,
    ) -> None:
        if mode not in ("fixed", "curriculum"):
            raise ValueError("mode must be 'fixed' or 'curriculum'")
        if base_seed < 0:
            raise ValueError("base_seed must be non-negative")
        if total_epochs <= 0:
            raise ValueError("total_epochs must be > 0")
        self.base_dataset = base_dataset
        self.preprocess = preprocess
        self.mode = mode
        self.base_seed = int(base_seed)
        self.total_epochs = int(total_epochs)
        self.epoch_index = 0

    def __len__(self) -> int:
        return len(self.base_dataset)

    def set_epoch(self, epoch_index: int) -> None:
        # Reuse the curriculum helper for strict range/type validation.
        epoch_progress(epoch_index, self.total_epochs)
        self.epoch_index = int(epoch_index)

    def _difficulty_and_seed(self, image_id: str) -> tuple[str, int]:
        difficulty_seed = stable_seed(
            self.base_seed, image_id, self.epoch_index, "difficulty"
        )
        corruption_seed = stable_seed(
            self.base_seed, image_id, self.epoch_index, "corruption"
        )
        if self.mode == "fixed":
            difficulty = fixed_training_difficulty(seed=difficulty_seed)
        else:
            progress = epoch_progress(self.epoch_index, self.total_epochs)
            difficulty = curriculum_difficulty(progress=progress, seed=difficulty_seed)
        return difficulty, corruption_seed

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base_dataset[index]
        if not isinstance(sample, Mapping):
            raise TypeError("Base paired dataset must return a mapping")
        for required in ("image", "label", "image_id"):
            if required not in sample:
                raise KeyError(f"Base sample missing required key {required!r}")
        clean_pil = sample["image"]
        if not isinstance(clean_pil, Image.Image):
            raise TypeError("Base sample 'image' must be a PIL image before corruption")
        image_id = str(sample["image_id"])
        label = int(sample["label"])
        if label not in (0, 1):
            raise ValueError(f"Binary label must be 0 or 1, got {label}")

        difficulty, corruption_seed = self._difficulty_and_seed(image_id)
        corrupt_pil, trace = sample_corruption(
            clean_pil, difficulty=difficulty, seed=corruption_seed
        )
        return {
            "clean": self.preprocess(clean_pil),
            "corrupt": self.preprocess(corrupt_pil),
            "label": label,
            "image_id": image_id,
            "source_split": str(sample.get("source_split", "unknown")),
            "trace": trace,
        }


def paired_collate_fn(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate variable-length corruption traces without default-collate errors."""

    if not samples:
        raise ValueError("Cannot collate an empty batch")
    return {
        "clean": torch.stack([sample["clean"] for sample in samples], dim=0),
        "corrupt": torch.stack([sample["corrupt"] for sample in samples], dim=0),
        "label": torch.tensor([int(sample["label"]) for sample in samples], dtype=torch.float32),
        "image_id": [str(sample["image_id"]) for sample in samples],
        "source_split": [str(sample["source_split"]) for sample in samples],
        "trace": [sample["trace"] for sample in samples],
    }
