"""Evaluate one M0--M4 checkpoint on the frozen Track 5 robustness suite."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.preprocessing import get_dinov2_preprocess
from data.sid_dataset import SIDManifestDataset
from evaluation.model_loading import load_adapter
from evaluation.robustness import run_benchmark, save_benchmark_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, choices=["M0", "M1", "M2", "M3", "M4"])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/sid_test.csv"))
    parser.add_argument("--dataset-name", default="SID_internal_test")
    parser.add_argument("--model-name", default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be >0 and num-workers must be >=0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{args.model_id}: evaluating on {device}")
    base = SIDManifestDataset(args.manifest)
    adapter = load_adapter(
        args.model_id,
        args.checkpoint,
        device=device,
        model_name=args.model_name,
    )
    predictions = run_benchmark(
        adapter,
        base,
        get_dinov2_preprocess(),
        device=device,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    predictions_path = args.output_root / "predictions" / args.model_id / f"{args.dataset_name}_robustness.csv"
    by_condition_path = args.output_root / "evaluation" / f"{args.model_id}_{args.dataset_name}_by_condition.csv"
    summary_path = args.output_root / "evaluation" / f"{args.model_id}_{args.dataset_name}_summary.csv"
    summary = save_benchmark_outputs(
        predictions,
        predictions_path=predictions_path,
        by_condition_path=by_condition_path,
        summary_path=summary_path,
    )
    print("\nRobustness summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"Raw predictions: {predictions_path}")


if __name__ == "__main__":
    main()
