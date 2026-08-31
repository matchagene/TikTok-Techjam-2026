"""Manifest-backed SID-Set dataset with explicit binary label semantics.

Primary Track 5 task:
    0 = real
    1 = fully synthetic / AI-generated
    SID label 2 (tampered) is excluded.

The manifest, not directory names, is the source of truth for labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

SID_REAL_LABEL = 0
SID_SYNTHETIC_LABEL = 1
SID_TAMPERED_LABEL = 2
VALID_BINARY_LABELS = frozenset({0, 1})
REQUIRED_MANIFEST_COLUMNS = frozenset(
    {"image_id", "cached_path", "source_split", "sid_label", "binary_label"}
)


def sid_to_binary_label(sid_label: int) -> int | None:
    """Map an official SID label to the primary binary Track 5 target.

    Returns ``None`` for tampered images so callers must explicitly exclude
    them rather than accidentally merging them into the synthetic class.
    """

    if sid_label == SID_REAL_LABEL:
        return 0
    if sid_label == SID_SYNTHETIC_LABEL:
        return 1
    if sid_label == SID_TAMPERED_LABEL:
        return None
    raise ValueError(f"Unexpected SID label {sid_label!r}; expected 0, 1, or 2.")


class SIDManifestDataset(Dataset):
    """Load canonical PIL samples from a deterministic SID manifest.

    Samples follow the team contract::

        {
            "image": PIL.Image.Image,
            "label": 0 or 1,
            "image_id": str,
            "source_split": str,
        }

    ``cached_path`` is additionally returned for traceability.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        project_root: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        frame = pd.read_csv(self.manifest_path)
        missing = REQUIRED_MANIFEST_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(
                f"Manifest {self.manifest_path} is missing required columns: "
                f"{sorted(missing)}"
            )

        if frame.empty:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")
        if frame["image_id"].isna().any():
            raise ValueError("Manifest contains null image_id values.")
        if frame["image_id"].astype(str).duplicated().any():
            raise ValueError("Manifest contains duplicate image_id values.")

        sid_labels = set(frame["sid_label"].astype(int).unique().tolist())
        if SID_TAMPERED_LABEL in sid_labels:
            raise ValueError("Primary SID manifest must not contain tampered label 2.")
        if not sid_labels.issubset({SID_REAL_LABEL, SID_SYNTHETIC_LABEL}):
            raise ValueError(f"Manifest has unsupported SID labels: {sorted(sid_labels)}")

        binary_labels = set(frame["binary_label"].astype(int).unique().tolist())
        if not binary_labels.issubset(VALID_BINARY_LABELS):
            raise ValueError(f"Manifest has invalid binary labels: {sorted(binary_labels)}")

        expected_binary = frame["sid_label"].astype(int)
        actual_binary = frame["binary_label"].astype(int)
        if not expected_binary.equals(actual_binary):
            raise ValueError(
                "For SID labels 0/1, binary_label must exactly match sid_label."
            )

        self.frame = frame.reset_index(drop=True)
        self.project_root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else self._infer_project_root()
        )

    def _infer_project_root(self) -> Path:
        # Standard layout is <repo>/data/manifests/file.csv.
        parents = self.manifest_path.parents
        if len(parents) >= 3 and parents[0].name == "manifests" and parents[1].name == "data":
            return parents[2]
        return Path.cwd().resolve()

    def __len__(self) -> int:
        return len(self.frame)

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        return path if path.is_absolute() else self.project_root / path

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        path = self._resolve_path(str(row["cached_path"]))
        if not path.is_file():
            raise FileNotFoundError(f"Cached SID image not found: {path}")

        with Image.open(path) as opened:
            image = opened.convert("RGB").copy()

        return {
            "image": image,
            "label": int(row["binary_label"]),
            "image_id": str(row["image_id"]),
            "source_split": str(row["source_split"]),
            "cached_path": str(row["cached_path"]),
        }


class SIDPreprocessedDataset(Dataset):
    """Tensor-view wrapper used by ordinary clean-image training/evaluation."""

    def __init__(
        self,
        base_dataset: SIDManifestDataset,
        preprocess: Callable[[Image.Image], Any],
    ) -> None:
        self.base_dataset = base_dataset
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        sample = dict(self.base_dataset[index])
        sample["image"] = self.preprocess(sample["image"])
        return sample
