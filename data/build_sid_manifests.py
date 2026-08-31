"""Build deterministic, balanced SID-Set manifests and lossless PNG cache.

This script intentionally excludes SID label 2 (tampered). It samples class
quotas from shuffled *official* train/validation splits instead of taking the
first N streaming records, then splits the official validation pool into a
model-development validation set and a held-out internal test set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pandas as pd
from PIL import Image

from data.sid_dataset import SID_TAMPERED_LABEL, sid_to_binary_label


@dataclass(frozen=True)
class ManifestBuildConfig:
    dataset_name: str = "saberzl/SID_Set"
    train_per_class: int = 1250
    validation_per_class: int = 250
    test_per_class: int = 250
    seed: int = 42
    shuffle_buffer_size: int = 10_000
    cache_root: Path = Path("data/cache/sid")
    manifests_dir: Path = Path("data/manifests")

    def validate(self) -> None:
        for name in ("train_per_class", "validation_per_class", "test_per_class"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if self.shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be > 0")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


def _load_hf_streams(dataset_name: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only in real runs
        raise RuntimeError(
            "The 'datasets' package is required. Install project requirements first."
        ) from exc
    return load_dataset(dataset_name, streaming=True)


def _choose_validation_split_name(dataset: Mapping[str, Any]) -> str:
    for candidate in ("validation", "val"):
        if candidate in dataset:
            return candidate
    raise KeyError(
        f"Dataset has no validation split; available splits: {sorted(dataset.keys())}"
    )


def _validated_sample(sample: Mapping[str, Any]) -> tuple[str, Image.Image, int, int] | None:
    required = {"img_id", "image", "label"}
    missing = required.difference(sample)
    if missing:
        raise KeyError(f"SID sample missing fields: {sorted(missing)}")

    sid_label = int(sample["label"])
    binary_label = sid_to_binary_label(sid_label)
    if sid_label == SID_TAMPERED_LABEL:
        return None
    if binary_label is None:  # defensive if mapping changes later
        return None

    image_id = str(sample["img_id"])
    if not image_id:
        raise ValueError("SID sample has empty img_id")
    image = sample["image"]
    if not isinstance(image, Image.Image):
        raise TypeError(f"SID sample {image_id} image is not a PIL image")
    return image_id, image, sid_label, binary_label


def collect_balanced(
    stream: Iterable[Mapping[str, Any]],
    *,
    per_class: int,
) -> list[Mapping[str, Any]]:
    """Collect an equal quota of real and fully-synthetic samples."""

    if per_class <= 0:
        raise ValueError("per_class must be > 0")

    counts = {0: 0, 1: 0}
    selected: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()

    for sample in stream:
        validated = _validated_sample(sample)
        if validated is None:
            continue
        image_id, _image, _sid_label, binary_label = validated
        if image_id in seen_ids:
            continue
        if counts[binary_label] >= per_class:
            continue

        selected.append(sample)
        seen_ids.add(image_id)
        counts[binary_label] += 1
        if counts[0] == per_class and counts[1] == per_class:
            break

    if counts != {0: per_class, 1: per_class}:
        raise RuntimeError(
            "Could not satisfy balanced quota. "
            f"Requested {per_class}/class, collected {counts}."
        )
    return selected


def split_balanced_validation_pool(
    selected: Sequence[Mapping[str, Any]],
    *,
    validation_per_class: int,
    test_per_class: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split an already shuffled balanced pool without reusing an image."""

    by_class: dict[int, list[Mapping[str, Any]]] = {0: [], 1: []}
    for sample in selected:
        validated = _validated_sample(sample)
        if validated is None:
            continue
        by_class[validated[3]].append(sample)

    needed = validation_per_class + test_per_class
    for label, items in by_class.items():
        if len(items) < needed:
            raise RuntimeError(
                f"Validation pool has only {len(items)} class-{label} samples; need {needed}."
            )

    val = (
        by_class[0][:validation_per_class]
        + by_class[1][:validation_per_class]
    )
    test = (
        by_class[0][validation_per_class:needed]
        + by_class[1][validation_per_class:needed]
    )
    return val, test


def _safe_cache_filename(image_id: str) -> str:
    # Keep readable IDs when safe; append a short digest to make path collisions
    # impossible even if two IDs sanitize to the same string.
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in image_id)
    safe = safe[:120] or "image"
    digest = hashlib.sha1(image_id.encode("utf-8")).hexdigest()[:10]
    return f"{safe}__{digest}.png"


def _portable_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def cache_samples_and_build_rows(
    samples: Sequence[Mapping[str, Any]],
    *,
    manifest_split: str,
    source_split: str,
    cache_root: Path,
    project_root: Path,
) -> list[dict[str, Any]]:
    output_dir = cache_root / manifest_split
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for sample in samples:
        validated = _validated_sample(sample)
        if validated is None:
            continue
        image_id, image, sid_label, binary_label = validated

        output_path = output_dir / _safe_cache_filename(image_id)
        # PNG introduces no extra lossy compression and keeps Person 2 in
        # control of when JPEG corruption is applied.
        image.convert("RGB").save(output_path, format="PNG")

        width = int(sample.get("width", image.width))
        height = int(sample.get("height", image.height))
        rows.append(
            {
                "image_id": image_id,
                "cached_path": _portable_path(output_path, project_root),
                "source_split": source_split,
                "sid_label": sid_label,
                "binary_label": binary_label,
                "width": width,
                "height": height,
            }
        )
    return rows


def write_manifest(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"Refusing to write empty manifest: {path}")
    frame.to_csv(path, index=False)


def build_manifests(config: ManifestBuildConfig, *, project_root: Path | None = None) -> None:
    config.validate()
    root = (project_root or Path.cwd()).resolve()
    cache_root = (root / config.cache_root).resolve() if not config.cache_root.is_absolute() else config.cache_root
    manifests_dir = (
        (root / config.manifests_dir).resolve()
        if not config.manifests_dir.is_absolute()
        else config.manifests_dir
    )

    dataset = _load_hf_streams(config.dataset_name)
    if "train" not in dataset:
        raise KeyError(f"Dataset has no train split; available: {sorted(dataset.keys())}")
    val_name = _choose_validation_split_name(dataset)

    train_stream = dataset["train"].shuffle(
        seed=config.seed, buffer_size=config.shuffle_buffer_size
    )
    val_stream = dataset[val_name].shuffle(
        seed=config.seed + 1, buffer_size=config.shuffle_buffer_size
    )

    train_samples = collect_balanced(train_stream, per_class=config.train_per_class)
    val_test_pool = collect_balanced(
        val_stream,
        per_class=config.validation_per_class + config.test_per_class,
    )
    val_samples, test_samples = split_balanced_validation_pool(
        val_test_pool,
        validation_per_class=config.validation_per_class,
        test_per_class=config.test_per_class,
    )

    train_rows = cache_samples_and_build_rows(
        train_samples,
        manifest_split="train",
        source_split="train",
        cache_root=cache_root,
        project_root=root,
    )
    val_rows = cache_samples_and_build_rows(
        val_samples,
        manifest_split="val",
        source_split=val_name,
        cache_root=cache_root,
        project_root=root,
    )
    test_rows = cache_samples_and_build_rows(
        test_samples,
        manifest_split="test",
        source_split=val_name,
        cache_root=cache_root,
        project_root=root,
    )

    write_manifest(train_rows, manifests_dir / "sid_train.csv")
    write_manifest(val_rows, manifests_dir / "sid_val.csv")
    write_manifest(test_rows, manifests_dir / "sid_test.csv")

    build_meta = {
        "dataset_name": config.dataset_name,
        "seed": config.seed,
        "shuffle_buffer_size": config.shuffle_buffer_size,
        "train_per_class": config.train_per_class,
        "validation_per_class": config.validation_per_class,
        "test_per_class": config.test_per_class,
        "sid_label_mapping": {"0": "real", "1": "full_synthetic", "2": "excluded_tampered"},
    }
    (manifests_dir / "sid_manifest_build.meta.json").write_text(
        json.dumps(build_meta, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="saberzl/SID_Set")
    parser.add_argument("--train-per-class", type=int, default=1250)
    parser.add_argument("--validation-per-class", type=int, default=250)
    parser.add_argument("--test-per-class", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000)
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/sid"))
    parser.add_argument("--manifests-dir", type=Path, default=Path("data/manifests"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ManifestBuildConfig(
        dataset_name=args.dataset_name,
        train_per_class=args.train_per_class,
        validation_per_class=args.validation_per_class,
        test_per_class=args.test_per_class,
        seed=args.seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
        cache_root=args.cache_root,
        manifests_dir=args.manifests_dir,
    )
    build_manifests(config)


if __name__ == "__main__":
    main()
