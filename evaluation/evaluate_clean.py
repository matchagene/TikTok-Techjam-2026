"""Evaluate a checkpoint on clean images while preserving raw predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from data.preprocessing import get_dinov2_preprocess
from data.sid_dataset import SIDManifestDataset
from evaluation.conditions import benchmark_conditions
from evaluation.metrics import compute_binary_metrics
from evaluation.model_loading import load_adapter
from evaluation.output_paths import clean_output_paths, resolve_run_tag
from evaluation.robustness import run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, choices=["M0", "M1", "M2", "M3", "M4"])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/sid_test.csv"))
    parser.add_argument("--dataset-name", default="SID_internal_test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional output run identifier; defaults to checkpoint filename stem.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = load_adapter(args.model_id, args.checkpoint, device=device)
    base = SIDManifestDataset(args.manifest)
    clean_condition = (benchmark_conditions(include_clean=True)[0],)
    predictions = run_benchmark(
        adapter,
        base,
        get_dinov2_preprocess(),
        device=device,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        conditions=clean_condition,
    )
    metrics = compute_binary_metrics(predictions["label"], predictions["p_fake"])

    run_tag = resolve_run_tag(args.checkpoint, args.run_tag)
    pred_path, metric_path = clean_output_paths(
        output_root=args.output_root,
        model_id=args.model_id,
        run_tag=run_tag,
        dataset_name=args.dataset_name,
    )
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(pred_path, index=False)
    pd.DataFrame([{"model_id": args.model_id, "dataset": args.dataset_name, **metrics}]).to_csv(metric_path, index=False)
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
