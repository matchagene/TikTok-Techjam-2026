"""Launch M2: paired clean/corrupt augmentation training, classification only."""

from pathlib import Path

from training.run_robust_experiment import run


if __name__ == "__main__":
    run(Path("configs/M2_augmented.yaml"))
