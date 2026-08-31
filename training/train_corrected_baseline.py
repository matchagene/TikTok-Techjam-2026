"""Train M1: scientifically controlled clean DINOv2 baseline.

M1 intentionally keeps the M0 model/training architecture while correcting the
protocol: explicit 0=real/1=fake labels, tampered exclusion, manifest-backed
splits, lossless cache, and validation checkpoint selection by ROC-AUC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.preprocessing import get_dinov2_preprocess
from data.sid_dataset import SIDManifestDataset, SIDPreprocessedDataset
from models.baseline import DINOBaseline


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic behavior where PyTorch supports it; warn_only avoids a hard
    # failure if a backend operation has no deterministic implementation.
    torch.use_deterministic_algorithms(True, warn_only=True)


def _worker_seed(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(
    manifest: Path,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    base = SIDManifestDataset(manifest)
    dataset = SIDPreprocessedDataset(base, get_dinov2_preprocess())
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        worker_init_fn=_worker_seed if num_workers > 0 else None,
        generator=generator,
    )


def _to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True).float().view(-1, 1)
    return images, labels


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    # Frozen backbones should stay in eval mode; this also protects against
    # accidental training-time state changes if the backbone ever gains such layers.
    if hasattr(model, "backbone") and not any(p.requires_grad for p in model.backbone.parameters()):
        model.backbone.eval()

    total_loss = 0.0
    total = 0
    correct = 0
    for batch in tqdm(loader, desc="M1 train", leave=False):
        images, labels = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        count = labels.shape[0]
        total_loss += float(loss.detach()) * count
        probabilities = torch.sigmoid(logits.detach())
        correct += int(((probabilities >= 0.5) == (labels >= 0.5)).sum().item())
        total += count

    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    labels_all: list[float] = []
    probs_all: list[float] = []
    total_loss = 0.0
    total = 0

    for batch in tqdm(loader, desc="M1 validate", leave=False):
        images, labels = _to_device(batch, device)
        logits = model(images)
        loss = criterion(logits, labels)
        probabilities = torch.sigmoid(logits)

        count = labels.shape[0]
        total_loss += float(loss) * count
        total += count
        labels_all.extend(labels.squeeze(1).cpu().tolist())
        probs_all.extend(probabilities.squeeze(1).cpu().tolist())

    if len(set(labels_all)) != 2:
        raise ValueError("ROC-AUC requires both real and fake examples in validation data.")

    y_true = np.asarray(labels_all, dtype=np.int64)
    y_prob = np.asarray(probs_all, dtype=np.float64)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    return {
        "loss": total_loss / total,
        "auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_real": int((y_true == 0).sum()),
        "n_fake": int((y_true == 1).sum()),
    }


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


def _write_metadata(
    *,
    config: dict[str, Any],
    config_path: Path,
    checkpoint_path: Path,
    metadata_path: Path,
    best_epoch: int,
    best_val: dict[str, float],
) -> None:
    data_cfg = config["data"]
    training_cfg = config["training"]
    metadata = {
        "model_id": config["experiment"]["model_id"],
        "git_commit": _git_commit(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "training_manifest": data_cfg["train_manifest"],
        "validation_manifest": data_cfg["val_manifest"],
        "test_manifest": data_cfg["test_manifest"],
        "seed": config["experiment"]["seed"],
        "epochs_configured": training_cfg["epochs"],
        "best_epoch": best_epoch,
        "batch_size": training_cfg["batch_size"],
        "optimizer": "AdamW",
        "learning_rate": training_cfg["learning_rate"],
        "weight_decay": training_cfg["weight_decay"],
        "lambda_pred": 0.0,
        "lambda_repr": 0.0,
        "augmentation_config": None,
        "label_convention": {"0": "real", "1": "ai_generated_fake"},
        "checkpoint_sha256": _sha256(checkpoint_path),
        "best_validation_metrics": best_val,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def train(config_path: Path) -> None:
    config = load_config(config_path)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    training_cfg = config["training"]
    train_loader = build_loader(
        Path(config["data"]["train_manifest"]),
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(training_cfg["num_workers"]),
        pin_memory=bool(training_cfg["pin_memory"]),
        seed=seed,
    )
    val_loader = build_loader(
        Path(config["data"]["val_manifest"]),
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(training_cfg["num_workers"]),
        pin_memory=bool(training_cfg["pin_memory"]),
        seed=seed,
    )

    model_cfg = config["model"]
    model = DINOBaseline(
        model_name=model_cfg["name"],
        freeze_backbone=bool(model_cfg["freeze_backbone"]),
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found for M1.")
    optimizer = AdamW(
        trainable,
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )

    checkpoint_path = Path(config["output"]["checkpoint"])
    metadata_path = Path(config["output"]["metadata"])
    metrics_path = Path(config["output"]["clean_metrics"])
    history_path = Path(config["output"]["history"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | int]] = []
    best_auc = float("-inf")
    best_epoch = -1
    best_val: dict[str, float] | None = None

    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "clean_val_auc": val_metrics["auc"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)

        print(
            f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} "
            f"clean_val_auc={val_metrics['auc']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_epoch = epoch
            best_val = val_metrics
            torch.save(model.state_dict(), checkpoint_path)
            print(f"=> saved best M1 by clean validation ROC-AUC: {best_auc:.4f}")

    if best_val is None:
        raise RuntimeError("Training completed without a valid checkpoint.")

    pd.DataFrame([{
        "clean_auc": best_val["auc"],
        "accuracy": best_val["accuracy"],
        "precision": best_val["precision"],
        "recall": best_val["recall"],
        "f1": best_val["f1"],
        "n_real": best_val["n_real"],
        "n_fake": best_val["n_fake"],
        "best_epoch": best_epoch,
    }]).to_csv(metrics_path, index=False)
    _write_metadata(
        config=config,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        best_epoch=best_epoch,
        best_val=best_val,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/corrected_baseline.yaml")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
