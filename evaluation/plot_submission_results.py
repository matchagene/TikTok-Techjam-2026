"""Create compact submission figures from real aggregated evaluation results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_clean_vs_robust(comparison: pd.DataFrame, output_path: Path) -> None:
    required = {"Model", "Clean AUC", "Robust Pooled AUC"}
    missing = required.difference(comparison.columns)
    if missing:
        raise ValueError(f"Comparison CSV missing columns: {sorted(missing)}")

    ax = comparison.set_index("Model")[["Clean AUC", "Robust Pooled AUC"]].plot.bar(
        figsize=(9, 5),
        rot=0,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Clean vs Transformed Performance")
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(ax.figure)


def plot_condition_comparison(conditions: pd.DataFrame, output_path: Path) -> None:
    required = {"Model", "condition_id", "roc_auc"}
    missing = required.difference(conditions.columns)
    if missing:
        raise ValueError(f"Condition CSV missing columns: {sorted(missing)}")

    pivot = conditions.pivot(index="condition_id", columns="Model", values="roc_auc")
    ax = pivot.plot(figsize=(13, 5), marker="o")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_xlabel("Benchmark condition")
    ax.set_title("ROC-AUC by Transformation Condition")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=55)
    ax.figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(ax.figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("results/evaluation/model_comparison.csv"),
    )
    parser.add_argument(
        "--conditions",
        type=Path,
        default=Path("results/evaluation/condition_comparison.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = pd.read_csv(args.comparison)
    conditions = pd.read_csv(args.conditions)
    clean_path = args.output_dir / "clean_vs_robust.png"
    condition_path = args.output_dir / "condition_auc_comparison.png"
    plot_clean_vs_robust(comparison, clean_path)
    plot_condition_comparison(conditions, condition_path)
    print(f"Wrote {clean_path}")
    print(f"Wrote {condition_path}")


if __name__ == "__main__":
    main()
