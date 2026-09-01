"""Directory-to-JSON inference entry point required by the hackathon.

Given a directory of images and a trained M1--M4 checkpoint, write a JSON array
containing ``image_path`` and ``pred`` for each image, where ``pred`` is the
canonical probability that the image is AI-generated/fake.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image

from data.preprocessing import get_dinov2_preprocess
from evaluation.model_adapter import ModelAdapter
from evaluation.model_loading import load_adapter

SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})


def select_device(requested: str = "auto") -> torch.device:
    """Resolve an inference device without requiring an accelerator."""

    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


def discover_images(input_dir: str | Path, *, recursive: bool = True) -> list[Path]:
    """Return supported image paths in deterministic lexical order."""

    root = Path(input_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Input directory not found: {root}")
    iterator: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
    paths = sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported images found under {root}")
    return paths


def _predict_batch(
    adapter: ModelAdapter,
    image_paths: list[Path],
    *,
    device: torch.device,
) -> list[float]:
    preprocess = get_dinov2_preprocess()
    tensors = []
    for path in image_paths:
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
                tensors.append(preprocess(image))
        except Exception as exc:
            raise RuntimeError(f"Failed to read image {path}") from exc

    batch = torch.stack(tensors, dim=0).to(device)
    with torch.inference_mode():
        probabilities = adapter.predict_fake_probability(batch)
    return [float(value) for value in probabilities.squeeze(1).detach().cpu().tolist()]


def predict_paths(
    adapter: ModelAdapter,
    image_paths: list[Path],
    *,
    device: torch.device,
    batch_size: int = 16,
) -> list[dict[str, object]]:
    """Predict canonical P(fake) for already-discovered image paths."""

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    rows: list[dict[str, object]] = []
    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start : start + batch_size]
        probabilities = _predict_batch(adapter, chunk, device=device)
        for path, probability in zip(chunk, probabilities):
            rows.append({"image_path": path.as_posix(), "pred": probability})
    return rows


def run_directory_inference(
    *,
    input_dir: str | Path,
    checkpoint: str | Path,
    model_id: str,
    output: str | Path,
    batch_size: int = 16,
    recursive: bool = True,
    device_name: str = "auto",
    model_name: str = "vit_base_patch14_dinov2.lvd142m",
) -> list[dict[str, object]]:
    """Load a checkpoint, score a directory, and write the required JSON."""

    device = select_device(device_name)
    image_paths = discover_images(input_dir, recursive=recursive)
    adapter = load_adapter(
        model_id,
        checkpoint,
        device=device,
        model_name=model_name,
    )
    rows = predict_paths(adapter, image_paths, device=device, batch_size=batch_size)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--input-dir", dest="input_dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-id", required=True, choices=["M0", "M1", "M2", "M3", "M4"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--model-name", default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Only score files directly inside the input directory.",
    )
    parser.set_defaults(recursive=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_directory_inference(
        input_dir=args.input_dir,
        checkpoint=args.checkpoint,
        model_id=args.model_id,
        output=args.output,
        batch_size=args.batch_size,
        recursive=args.recursive,
        device_name=args.device,
        model_name=args.model_name,
    )
    print(f"Scored {len(rows)} images -> {args.output}")
    print("pred semantics: probability that the image is AI-generated/fake")


if __name__ == "__main__":
    main()
