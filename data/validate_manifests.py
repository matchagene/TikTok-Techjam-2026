"""Validate SID manifests before any training/evaluation run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image

from data.sid_dataset import REQUIRED_MANIFEST_COLUMNS, SID_TAMPERED_LABEL


def _load_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = REQUIRED_MANIFEST_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return frame


def _resolve_cached_path(raw: str, project_root: Path) -> Path:
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else project_root / path


def validate_manifest_frame(
    frame: pd.DataFrame,
    *,
    split_name: str,
    project_root: Path,
    verify_images: bool = True,
) -> dict[str, int]:
    if frame.empty:
        raise ValueError(f"{split_name} manifest is empty")
    if frame["image_id"].isna().any():
        raise ValueError(f"{split_name}: null image_id")
    if frame["image_id"].astype(str).duplicated().any():
        raise ValueError(f"{split_name}: duplicate image_id")

    sid_labels = frame["sid_label"].astype(int)
    binary_labels = frame["binary_label"].astype(int)
    if (sid_labels == SID_TAMPERED_LABEL).any():
        raise ValueError(f"{split_name}: tampered SID label 2 present")
    if not set(binary_labels.unique()).issubset({0, 1}):
        raise ValueError(f"{split_name}: binary labels outside {{0, 1}}")
    if set(binary_labels.unique()) != {0, 1}:
        raise ValueError(f"{split_name}: both binary classes must be present")
    if not sid_labels.equals(binary_labels):
        raise ValueError(f"{split_name}: sid_label and binary_label disagree")

    counts = binary_labels.value_counts().to_dict()
    if counts.get(0, 0) != counts.get(1, 0):
        raise ValueError(f"{split_name}: classes are not balanced: {counts}")

    if verify_images:
        for raw_path in frame["cached_path"].astype(str):
            path = _resolve_cached_path(raw_path, project_root)
            if not path.is_file():
                raise FileNotFoundError(f"{split_name}: missing cached image {path}")
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                raise ValueError(f"{split_name}: unreadable image {path}: {exc}") from exc

    return {"real": int(counts.get(0, 0)), "fake": int(counts.get(1, 0))}


def validate_no_overlap(frames: dict[str, pd.DataFrame]) -> None:
    names = list(frames)
    ids = {name: set(frame["image_id"].astype(str)) for name, frame in frames.items()}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = ids[left].intersection(ids[right])
            if overlap:
                examples = sorted(overlap)[:5]
                raise ValueError(
                    f"Split overlap {left}<->{right}: {len(overlap)} IDs, e.g. {examples}"
                )


def validate_manifests(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    *,
    project_root: Path,
    counts_output: Path,
    verify_images: bool = True,
) -> pd.DataFrame:
    frames = {
        "train": _load_manifest(train_path),
        "val": _load_manifest(val_path),
        "test": _load_manifest(test_path),
    }
    validate_no_overlap(frames)

    records: list[dict[str, int | str]] = []
    for split, frame in frames.items():
        counts = validate_manifest_frame(
            frame,
            split_name=split,
            project_root=project_root,
            verify_images=verify_images,
        )
        print(f"{split.capitalize()}: real={counts['real']} fake={counts['fake']}")
        records.append({"split": split, **counts, "total": counts["real"] + counts["fake"]})

    counts_frame = pd.DataFrame(records)
    counts_output.parent.mkdir(parents=True, exist_ok=True)
    counts_frame.to_csv(counts_output, index=False)
    return counts_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/manifests/sid_train.csv"))
    parser.add_argument("--val", type=Path, default=Path("data/manifests/sid_val.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/manifests/sid_test.csv"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--counts-output",
        type=Path,
        default=Path("results/data/class_counts.csv"),
    )
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Validate labels/splits without opening every cached image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_manifests(
        args.train,
        args.val,
        args.test,
        project_root=args.project_root.resolve(),
        counts_output=args.counts_output,
        verify_images=not args.skip_image_check,
    )


if __name__ == "__main__":
    main()
