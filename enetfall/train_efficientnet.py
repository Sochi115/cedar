"""Train EfficientNet-B0 on ENetFall's native CSI event clips."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import efficientnet_b0

from .compare_models import select_threshold
from .train_svm import EXPECTED_ROOM_COUNTS, ROOM_SPLITS, iter_enetfall_events

SEED = 42
LABEL_TO_INDEX = {"non-fall": 0, "fall": 1}


def build_efficientnet() -> nn.Module:
    """Build EfficientNet-B0 with three CSI receiver channels and two outputs."""

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model


def normalize_event(amplitude: np.ndarray) -> np.ndarray:
    """Robustly normalize one event over time; return receiver x time x subcarrier."""

    values = np.asarray(amplitude, dtype=np.float32)
    if values.shape != (625, 3, 30) or not np.isfinite(values).all():
        raise ValueError("amplitude must be finite with shape (625, 3, 30)")
    center = np.median(values, axis=0, keepdims=True)
    scale = 1.4826 * np.median(np.abs(values - center), axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    normalized = np.clip((values - center) / scale, -8.0, 8.0)
    return normalized.transpose(1, 0, 2).astype(np.float32, copy=False)


def load_splits(dataset_root: str | Path) -> dict[str, dict[str, Any]]:
    """Load validated native-resolution events into fixed room-held-out splits."""

    splits: dict[str, dict[str, Any]] = {
        name: {"images": [], "labels": [], "records": []}
        for name in ("train", "validation", "test")
    }
    room_counts: dict[str, Counter[str]] = {room: Counter() for room in ROOM_SPLITS}
    for event in iter_enetfall_events(dataset_root):
        split_name = ROOM_SPLITS[event["room"]]
        splits[split_name]["images"].append(normalize_event(event["amplitude"]))
        splits[split_name]["labels"].append(LABEL_TO_INDEX[event["label"]])
        splits[split_name]["records"].append(
            {"event_id": event["event_id"], "room": event["room"], "label": event["label"]}
        )
        room_counts[event["room"]][event["label"]] += 1

    actual_counts = {room: dict(counts) for room, counts in room_counts.items()}
    if actual_counts != EXPECTED_ROOM_COUNTS:
        raise ValueError(f"Unexpected ENetFall room/class counts: {actual_counts}")
    for split in splits.values():
        split["images"] = torch.from_numpy(np.stack(split["images"]))
        split["labels"] = torch.tensor(split["labels"], dtype=torch.long)
    return splits


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, float]:
    model.eval()
    all_scores: list[np.ndarray] = []
    loss_sum = 0.0
    count = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += float(loss_fn(logits, labels)) * labels.size(0)
            count += labels.size(0)
            all_scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(all_scores), loss_sum / count


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    truth = labels == LABEL_TO_INDEX["fall"]
    predicted = scores >= threshold
    matrix = confusion_matrix(truth, predicted, labels=[False, True])
    return {
        "support": int(labels.size),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "fall_precision": float(precision_score(truth, predicted, zero_division=0)),
        "fall_recall": float(recall_score(truth, predicted, zero_division=0)),
        "fall_f1": float(f1_score(truth, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "confusion_matrix_nonfall_fall": matrix.tolist(),
    }


def _render_report(result: dict[str, Any]) -> str:
    history = result["training"]["validation_loss_history"]
    loss_rows = "\n".join(
        f"| {entry['epoch']} | {entry['validation_loss']:.6f} |"
        for entry in history
    )

    def confusion_table(name: str) -> str:
        matrix = result[name]["confusion_matrix_nonfall_fall"]
        return (
            "| Actual / Predicted | Non-fall | Fall |\n"
            "|---|---:|---:|\n"
            f"| Non-fall | {matrix[0][0]} | {matrix[0][1]} |\n"
            f"| Fall | {matrix[1][0]} | {matrix[1][1]} |"
        )

    validation = result["validation"]
    test = result["test"]
    return f"""# ENetFall EfficientNet-B0 training report

- Device: `{result['training']['device']}`
- Epochs completed: {result['training']['epochs_completed']}
- Best epoch: {result['training']['best_epoch']}
- Best validation loss: **{result['training']['best_validation_loss']:.6f}**
- Batch size / learning rate / weight decay: {result['training']['batch_size']} / {result['training']['learning_rate']} / {result['training']['weight_decay']}
- Validation-selected threshold: {result['threshold']:.6f}
- Checkpoint: `{result['model']['artifact']}` (best validation loss; only checkpoint retained)

## Validation loss by epoch

| Epoch | Validation loss |
|---:|---:|
{loss_rows}

## Confusion matrices

Rows are actual labels; columns are predicted labels. Class order: non-fall, fall.

### Validation ({validation['support']} events; balanced accuracy {validation['balanced_accuracy']:.3f})

{confusion_table('validation')}

### Test ({test['support']} events; balanced accuracy {test['balanced_accuracy']:.3f})

{confusion_table('test')}

## Split

| Split | Rooms | Fall | Non-fall | Total |
|---|---|---:|---:|---:|
| Train | living room A, meeting room B | 149 | 233 | 382 |
| Validation | home lab left, home lab right | 76 | 94 | 170 |
| Test | lecture room D | 96 | 109 | 205 |

Participant independence is not guaranteed: ENetFall does not provide reliable participant IDs.
"""


def train_efficientnet(
    dataset_root: str | Path = "data/raw/ENetFall",
    output_dir: str | Path = "data/processed/enetfall/efficientnet-run",
    *,
    epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-3,
    patience: int = 5,
) -> dict[str, Any]:
    """Train on train rooms, select threshold on validation rooms, evaluate test room."""

    if (
        epochs < 1
        or batch_size < 1
        or learning_rate <= 0
        or weight_decay < 0
        or patience < 1
    ):
        raise ValueError(
            "epochs, batch_size, and patience must be positive; "
            "learning_rate must be > 0 and weight_decay must be >= 0"
        )
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    splits = load_splits(dataset_root)
    train, validation, test = (splits[name] for name in ("train", "validation", "test"))
    train_loader = DataLoader(
        TensorDataset(train["images"], train["labels"]), batch_size=batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        TensorDataset(validation["images"], validation["labels"]), batch_size=batch_size
    )
    test_loader = DataLoader(
        TensorDataset(test["images"], test["labels"]), batch_size=batch_size
    )

    device = _device()
    model = build_efficientnet().to(device)
    counts = torch.bincount(train["labels"], minlength=2).float()
    class_weights = counts.sum() / (2.0 * counts)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.pt"
    best_loss, best_epoch, stale_epochs = float("inf"), 0, 0
    validation_loss_history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()

        _, validation_loss = _predict(model, validation_loader, device)
        validation_loss_history.append({"epoch": epoch, "validation_loss": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch, stale_epochs = validation_loss, epoch, 0
            torch.save(model.state_dict(), model_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    validation_scores, _ = _predict(model, validation_loader, device)
    test_scores, _ = _predict(model, test_loader, device)
    threshold = select_threshold(
        np.where(validation["labels"].numpy() == 1, "fall", "non-fall"), validation_scores
    )

    result = {
        "run_id": "enetfall-efficientnet-b0-room-held-out-v1",
        "dataset": {
            "name": "ENetFall",
            "event_count": int(sum(len(splits[name]["labels"]) for name in splits)),
            "source_shape_per_event": [625, 3, 30],
        },
        "model": {
            "family": "cnn",
            "name": "EfficientNet-B0",
            "initialization": "random; no ImageNet weights",
            "input": "3 receiver channels x 625 time samples x 30 subcarriers",
            "artifact": model_path.name,
        },
        "split": {
            "strategy": "fixed environment-held-out",
            "room_assignments": ROOM_SPLITS,
            "threshold_selection": "maximize validation balanced accuracy",
            "caveat": "ENetFall lacks reliable participant IDs; participant independence is not guaranteed.",
        },
        "training": {
            "device": str(device),
            "seed": SEED,
            "epochs_completed": epoch,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "validation_loss_history": validation_loss_history,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "class_weighted_loss": True,
        },
        "threshold": threshold,
        "validation": _metrics(validation["labels"].numpy(), validation_scores, threshold),
        "test": _metrics(test["labels"].numpy(), test_scores, threshold),
    }
    result["report"] = "report.md"
    (output / "run.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_render_report(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", nargs="?", default="data/raw/ENetFall")
    parser.add_argument("output_dir", nargs="?", default="data/processed/enetfall/efficientnet-run")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    result = train_efficientnet(
        args.dataset_root,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )
    print(json.dumps(result["test"], indent=2))
    print(f"Report: {Path(args.output_dir).expanduser() / result['report']}")


if __name__ == "__main__":
    main()
