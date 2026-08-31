"""Launch M4: M3 objective with progressive corruption difficulty."""

from pathlib import Path

from training.run_robust_experiment import run


if __name__ == "__main__":
    run(Path("configs/M4_curriculum.yaml"))
