"""Launch M3: fixed-distribution paired consistency training."""

from pathlib import Path

from training.run_robust_experiment import run


if __name__ == "__main__":
    run(Path("configs/M3_pairwise.yaml"))
