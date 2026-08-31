"""Shared runner for the controlled M2/M3/M4 ablation ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmentations.curriculum import (
    EARLY_CURRICULUM_WEIGHTS,
    FIXED_TRAINING_WEIGHTS,
    LATE_CURRICULUM_WEIGHTS,
    MIDDLE_CURRICULUM_WEIGHTS,
)
from augmentations.sampler import (
    DIFFICULTIES,
    OPERATION_COUNT_RANGES,
    TRAINING_POOLS,
)
from data.preprocessing import get_dinov2_preprocess
from data.sid_dataset import SIDManifestDataset, SIDPreprocessedDataset
from evaluation.metrics import compute_binary_metrics
from models.robust_detector import RobustDINODetector
from training.paired_dataset import PairedSIDDataset, paired_collate_fn
from training.pairwise_engine import train_pairwise_epoch

VALID_MODEL_IDS = frozenset({"M2", "M3", "M4"})


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Referenced configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return value


def _resolve_config_reference(value: object, *, experiment_config_path: Path) -> Path:
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate

    # Canonical project configs use repo-root-relative paths such as
    # configs/augmentation_train.yaml. Prefer cwd when invoked from repo root,
    # then fall back to the repository containing the experiment config.
    if candidate.is_file():
        return candidate

    repository_root = experiment_config_path.resolve().parent.parent
    return repository_root / candidate


def _assert_float_mapping_equal(
    actual: Mapping[str, Any],
    expected: Mapping[str, float],
    *,
    context: str,
) -> None:
    if set(actual) != set(expected):
        raise ValueError(
            f"{context} keys drifted from runtime constants: "
            f"actual={sorted(actual)}, expected={sorted(expected)}"
        )

    for key, expected_value in expected.items():
        actual_value = float(actual[key])
        if not math.isclose(
            actual_value,
            float(expected_value),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{context}.{key}={actual_value} does not match "
                f"runtime value {float(expected_value)}"
            )


def validate_augmentation_config_mirror(path: Path) -> None:
    """Ensure augmentation_train.yaml exactly mirrors runtime sampler constants."""

    config = _load_yaml_mapping(path)
    root = config.get("training_corruption")
    if not isinstance(root, Mapping):
        raise ValueError(
            f"{path} must contain a training_corruption mapping"
        )

    fixed = root.get("fixed_distribution")
    if not isinstance(fixed, Mapping):
        raise ValueError(
            f"{path}: training_corruption.fixed_distribution must be a mapping"
        )
    _assert_float_mapping_equal(
        fixed,
        FIXED_TRAINING_WEIGHTS,
        context=f"{path}: fixed_distribution",
    )

    expected_difficulty_keys = set(DIFFICULTIES)
    actual_difficulty_keys = {
        key for key in root if key != "fixed_distribution"
    }
    if actual_difficulty_keys != expected_difficulty_keys:
        raise ValueError(
            f"{path}: difficulty sections drifted from runtime constants: "
            f"actual={sorted(actual_difficulty_keys)}, "
            f"expected={sorted(expected_difficulty_keys)}"
        )

    for difficulty in DIFFICULTIES:
        section = root.get(difficulty)
        if not isinstance(section, Mapping):
            raise ValueError(f"{path}: {difficulty} must be a mapping")

        expected_operation_count = list(OPERATION_COUNT_RANGES[difficulty])
        actual_operation_count = section.get("operation_count")
        if actual_operation_count != expected_operation_count:
            raise ValueError(
                f"{path}: {difficulty}.operation_count="
                f"{actual_operation_count!r} does not match runtime "
                f"{expected_operation_count!r}"
            )

        expected_pool = TRAINING_POOLS[difficulty]
        actual_pool_keys = {
            key for key in section if key != "operation_count"
        }
        if actual_pool_keys != set(expected_pool):
            raise ValueError(
                f"{path}: {difficulty} transform keys drifted from runtime "
                f"constants: actual={sorted(actual_pool_keys)}, "
                f"expected={sorted(expected_pool)}"
            )

        for transform_name, severities in expected_pool.items():
            actual_severities = section.get(transform_name)
            expected_severities = list(severities)
            if actual_severities != expected_severities:
                raise ValueError(
                    f"{path}: {difficulty}.{transform_name}="
                    f"{actual_severities!r} does not match runtime "
                    f"{expected_severities!r}"
                )


def validate_curriculum_config_mirror(path: Path) -> None:
    """Ensure curriculum.yaml mirrors the runtime M4 stage boundaries/weights."""

    config = _load_yaml_mapping(path)
    curriculum = config.get("curriculum")
    if not isinstance(curriculum, Mapping):
        raise ValueError(f"{path} must contain a curriculum mapping")

    expected = {
        "early": ((0.0, 1.0 / 3.0), EARLY_CURRICULUM_WEIGHTS),
        "middle": ((1.0 / 3.0, 2.0 / 3.0), MIDDLE_CURRICULUM_WEIGHTS),
        "late": ((2.0 / 3.0, 1.0), LATE_CURRICULUM_WEIGHTS),
    }

    if set(curriculum) != set(expected):
        raise ValueError(
            f"{path}: curriculum stages drifted from runtime constants"
        )

    for stage, (expected_progress, expected_weights) in expected.items():
        section = curriculum.get(stage)
        if not isinstance(section, Mapping):
            raise ValueError(f"{path}: curriculum.{stage} must be a mapping")

        actual_progress = section.get("progress")
        if (
            not isinstance(actual_progress, list)
            or len(actual_progress) != 2
            or not all(
                math.isclose(
                    float(actual),
                    float(expected_value),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for actual, expected_value in zip(
                    actual_progress, expected_progress
                )
            )
        ):
            raise ValueError(
                f"{path}: curriculum.{stage}.progress="
                f"{actual_progress!r} does not match runtime boundary "
                f"{list(expected_progress)!r}"
            )

        weights = section.get("weights")
        if not isinstance(weights, Mapping):
            raise ValueError(
                f"{path}: curriculum.{stage}.weights must be a mapping"
            )
        _assert_float_mapping_equal(
            weights,
            expected_weights,
            context=f"{path}: curriculum.{stage}.weights",
        )


def validate_runtime_config_mirrors(
    config: Mapping[str, Any],
    *,
    experiment_config_path: Path,
) -> None:
    """Fail before training if YAML experiment specs disagree with runtime code."""

    training = config["training"]

    augmentation_reference = training.get("augmentation_config")
    if not augmentation_reference:
        raise ValueError(
            "M2/M3/M4 must declare training.augmentation_config so the "
            "documented corruption specification can be validated"
        )

    augmentation_path = _resolve_config_reference(
        augmentation_reference,
        experiment_config_path=experiment_config_path,
    )
    validate_augmentation_config_mirror(augmentation_path)

    pair_mode = str(training["pair_mode"])
    curriculum_reference = training.get("curriculum_config")

    if pair_mode == "curriculum":
        if not curriculum_reference:
            raise ValueError(
                "Curriculum training must declare training.curriculum_config"
            )
        curriculum_path = _resolve_config_reference(
            curriculum_reference,
            experiment_config_path=experiment_config_path,
        )
        validate_curriculum_config_mirror(curriculum_path)
    elif curriculum_reference is not None:
        raise ValueError(
            "Fixed-mode M2/M3 must use training.curriculum_config: null"
        )


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    validate_config(config)
    validate_runtime_config_mirrors(
        config,
        experiment_config_path=path,
    )
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    for section in ("experiment", "data", "model", "training", "loss", "output"):
        if section not in config or not isinstance(config[section], Mapping):
            raise ValueError(f"Missing mapping section {section!r}")

    model_id = str(config["experiment"].get("model_id"))
    if model_id not in VALID_MODEL_IDS:
        raise ValueError(f"model_id must be one of {sorted(VALID_MODEL_IDS)}, got {model_id!r}")

    pair_mode = str(config["training"].get("pair_mode"))
    if pair_mode not in {"fixed", "curriculum"}:
        raise ValueError("training.pair_mode must be 'fixed' or 'curriculum'")
    if model_id in {"M2", "M3"} and pair_mode != "fixed":
        raise ValueError(f"{model_id} must use fixed corruption distribution")
    if model_id == "M4" and pair_mode != "curriculum":
        raise ValueError("M4 must use curriculum corruption distribution")

    lambda_pred = float(config["loss"].get("lambda_pred", -1))
    lambda_repr = float(config["loss"].get("lambda_repr", -1))
    if lambda_pred < 0 or lambda_repr < 0:
        raise ValueError("loss lambdas must be non-negative")
    if model_id == "M2" and (lambda_pred != 0.0 or lambda_repr != 0.0):
        raise ValueError("M2 must have lambda_pred=lambda_repr=0")
    if model_id in {"M3", "M4"} and (lambda_pred == 0.0 or lambda_repr == 0.0):
        raise ValueError(f"{model_id} should enable both consistency losses")

    epochs = int(config["training"].get("epochs", 0))
    batch_size = int(config["training"].get("batch_size", 0))
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("training.epochs and training.batch_size must be > 0")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_train_loader(config: Mapping[str, Any]) -> tuple[DataLoader, PairedSIDDataset]:
    training_cfg = config["training"]
    experiment_cfg = config["experiment"]
    base = SIDManifestDataset(Path(config["data"]["train_manifest"]))
    paired = PairedSIDDataset(
        base,
        get_dinov2_preprocess(),
        mode=str(training_cfg["pair_mode"]),
        base_seed=int(experiment_cfg["seed"]),
        total_epochs=int(training_cfg["epochs"]),
    )
    generator = torch.Generator().manual_seed(int(experiment_cfg["seed"]))
    loader = DataLoader(
        paired,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(training_cfg["num_workers"]),
        pin_memory=bool(training_cfg.get("pin_memory", True)) and torch.cuda.is_available(),
        collate_fn=paired_collate_fn,
        worker_init_fn=_seed_worker if int(training_cfg["num_workers"]) > 0 else None,
        generator=generator,
        persistent_workers=False,
    )
    return loader, paired


def build_clean_val_loader(config: Mapping[str, Any]) -> DataLoader:
    training_cfg = config["training"]
    experiment_cfg = config["experiment"]
    base = SIDManifestDataset(Path(config["data"]["val_manifest"]))
    dataset = SIDPreprocessedDataset(base, get_dinov2_preprocess())
    generator = torch.Generator().manual_seed(int(experiment_cfg["seed"]))
    return DataLoader(
        dataset,
        batch_size=int(training_cfg.get("eval_batch_size", training_cfg["batch_size"])),
        shuffle=False,
        num_workers=int(training_cfg["num_workers"]),
        pin_memory=bool(training_cfg.get("pin_memory", True)) and torch.cuda.is_available(),
        worker_init_fn=_seed_worker if int(training_cfg["num_workers"]) > 0 else None,
        generator=generator,
    )


def build_model(config: Mapping[str, Any]) -> RobustDINODetector:
    model_cfg = config["model"]
    model = RobustDINODetector(
        model_name=str(model_cfg["name"]),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        dropout=float(model_cfg.get("dropout", 0.3)),
    )
    init_checkpoint = model_cfg.get("initialize_from_m1")
    if init_checkpoint:
        path = Path(str(init_checkpoint))
        if not path.is_file():
            raise FileNotFoundError(
                f"M1 initialization checkpoint not found: {path}. Train/copy M1 first or set initialize_from_m1: null."
            )
        model.initialize_head_from_m1(path)

    unfreeze_last = int(model_cfg.get("unfreeze_last_blocks", 0))
    if unfreeze_last > 0:
        model.unfreeze_last_blocks(unfreeze_last)
    return model


def build_optimizer(model: RobustDINODetector, config: Mapping[str, Any]) -> AdamW:
    training_cfg = config["training"]
    head_lr = float(training_cfg["head_learning_rate"])
    backbone_lr = float(training_cfg.get("backbone_learning_rate", head_lr))
    weight_decay = float(training_cfg["weight_decay"])

    head_params = [
        parameter
        for module in (model.projector, model.classifier)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    backbone_params = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    groups: list[dict[str, Any]] = []
    if head_params:
        groups.append({"params": head_params, "lr": head_lr, "name": "head"})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr, "name": "backbone"})
    if not groups:
        raise RuntimeError("No trainable parameters found")
    return AdamW(groups, weight_decay=weight_decay)


@torch.inference_mode()
def evaluate_clean(
    model: RobustDINODetector,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    for batch in tqdm(loader, desc="clean val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits).squeeze(1).cpu().tolist()
        batch_labels = batch["label"]
        if isinstance(batch_labels, torch.Tensor):
            batch_labels = batch_labels.cpu().tolist()
        labels.extend(int(label) for label in batch_labels)
        probabilities.extend(float(prob) for prob in probs)
    return compute_binary_metrics(labels, probabilities)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_epoch_path(canonical: Path, epoch: int) -> Path:
    return canonical.with_name(f"{canonical.stem}_epoch{epoch:02d}{canonical.suffix}")


def run(config_path: Path) -> None:
    config = load_config(config_path)
    model_id = str(config["experiment"]["model_id"])
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{model_id}: using device {device}")
    train_loader, paired_dataset = build_train_loader(config)
    val_loader = build_clean_val_loader(config)
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)

    loss_cfg = config["loss"]
    output_cfg = config["output"]
    canonical_checkpoint = Path(output_cfg["checkpoint"])
    metadata_path = Path(output_cfg["metadata"])
    history_path = Path(output_cfg["history"])
    canonical_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_clean_auc = float("-inf")
    best_epoch = -1
    epoch_checkpoints: list[dict[str, Any]] = []

    for epoch_index in range(int(config["training"]["epochs"])):
        epoch = epoch_index + 1
        paired_dataset.set_epoch(epoch_index)
        train_metrics = train_pairwise_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            lambda_pred=float(loss_cfg["lambda_pred"]),
            lambda_repr=float(loss_cfg["lambda_repr"]),
            grad_clip_norm=(
                None
                if config["training"].get("grad_clip_norm") is None
                else float(config["training"]["grad_clip_norm"])
            ),
        )
        clean_metrics = evaluate_clean(model, val_loader, device=device)

        epoch_path = _checkpoint_epoch_path(canonical_checkpoint, epoch)
        torch.save(model.state_dict(), epoch_path)
        epoch_checkpoints.append(
            {"epoch": epoch, "path": str(epoch_path), "sha256": _sha256(epoch_path)}
        )

        row = {
            "epoch": epoch,
            "train_cls_loss": train_metrics.classification,
            "train_pred_loss": train_metrics.prediction,
            "train_repr_loss": train_metrics.representation,
            "train_total_loss": train_metrics.total,
            "clean_val_auc": clean_metrics["roc_auc"],
            "clean_val_accuracy": clean_metrics["accuracy"],
            "clean_val_precision": clean_metrics["precision"],
            "clean_val_recall": clean_metrics["recall"],
            "clean_val_f1": clean_metrics["f1"],
            "robust_val_auc": None,
            "head_learning_rate": next(
                group["lr"] for group in optimizer.param_groups if group.get("name") == "head"
            ),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(
            f"{model_id} epoch {epoch}: total={train_metrics.total:.4f} "
            f"cls={train_metrics.classification:.4f} pred={train_metrics.prediction:.4f} "
            f"repr={train_metrics.representation:.4f} clean_auc={clean_metrics['roc_auc']:.4f}"
        )

        # Until robust model-selection is integrated, keep all epoch checkpoints
        # and make the canonical path point to best *clean* AUC as a fallback.
        if float(clean_metrics["roc_auc"]) > best_clean_auc:
            best_clean_auc = float(clean_metrics["roc_auc"])
            best_epoch = epoch
            shutil.copy2(epoch_path, canonical_checkpoint)

    metadata = {
        "model_id": model_id,
        "git_commit": _git_commit(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "training_manifest": config["data"]["train_manifest"],
        "validation_manifest": config["data"]["val_manifest"],
        "seed": seed,
        "epochs": int(config["training"]["epochs"]),
        "batch_size": int(config["training"]["batch_size"]),
        "optimizer": "AdamW",
        "head_learning_rate": float(config["training"]["head_learning_rate"]),
        "backbone_learning_rate": float(config["training"].get("backbone_learning_rate", config["training"]["head_learning_rate"])),
        "weight_decay": float(config["training"]["weight_decay"]),
        "lambda_pred": float(loss_cfg["lambda_pred"]),
        "lambda_repr": float(loss_cfg["lambda_repr"]),
        "pair_mode": config["training"]["pair_mode"],
        "augmentation_config": config["training"].get("augmentation_config"),
        "curriculum_config": config["training"].get("curriculum_config"),
        "label_convention": {"0": "real", "1": "ai_generated_fake"},
        "initialize_from_m1": config["model"].get("initialize_from_m1"),
        "best_clean_auc_epoch_fallback": best_epoch,
        "best_clean_auc_fallback": best_clean_auc,
        "canonical_checkpoint_sha256": _sha256(canonical_checkpoint),
        "epoch_checkpoints": epoch_checkpoints,
        "checkpoint_selection_note": (
            "All epoch checkpoints are retained. Canonical checkpoint currently uses best clean validation AUC "
            "only as a fallback; Person 4 should select/report robust-validation-best once available."
        ),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
