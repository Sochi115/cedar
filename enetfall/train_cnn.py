"""Train a compact CNN on ENetFall CSI clips and write a standalone HTML report."""

from __future__ import annotations

import argparse
import html
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
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .compare_models import select_threshold
from .train_svm import EXPECTED_ROOM_COUNTS, iter_enetfall_events

SEED = 42
LABEL_TO_INDEX = {"non-fall": 0, "fall": 1}


class CompactCsiCnn(nn.Module):
    """Small 2D CNN; input shape is batch x receiver x time x subcarrier."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def normalize_event(amplitude: np.ndarray) -> np.ndarray:
    """Per-event z-score each receiver/subcarrier stream; preserve shape and transients."""

    values = np.asarray(amplitude, dtype=np.float32)
    if (
        values.ndim != 3
        or values.shape[0] < 4
        or values.shape[1:] != (3, 30)
        or not np.isfinite(values).all()
    ):
        raise ValueError("amplitude must be finite with shape (at least 4 samples, 3, 30)")
    center = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    normalized = np.clip((values - center) / scale, -8.0, 8.0)
    return normalized.transpose(1, 0, 2).astype(np.float32, copy=False)


def load_splits(dataset_root: str | Path, seed: int = SEED) -> dict[str, dict[str, Any]]:
    """Create stratified random 80/20 test split plus validation from development data."""

    events = list(iter_enetfall_events(dataset_root))
    room_counts: dict[str, Counter[str]] = {}
    for event in events:
        room_counts.setdefault(event["room"], Counter())[event["label"]] += 1
    actual_counts = {room: dict(counts) for room, counts in room_counts.items()}
    if actual_counts != EXPECTED_ROOM_COUNTS:
        raise ValueError(f"Unexpected ENetFall room/class counts: {actual_counts}")

    labels = np.asarray([event["label"] for event in events])
    indices = np.arange(len(events))
    development_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=labels
    )
    train_indices, validation_indices = train_test_split(
        development_indices,
        test_size=0.2,
        random_state=seed,
        stratify=labels[development_indices],
    )
    normalized = np.stack([normalize_event(event["amplitude"]) for event in events])
    records = [
        {"event_id": event["event_id"], "room": event["room"], "label": event["label"]}
        for event in events
    ]
    del events

    splits: dict[str, dict[str, Any]] = {}
    for name, split_indices in (
        ("train", train_indices),
        ("validation", validation_indices),
        ("test", test_indices),
    ):
        splits[name] = {
            "images": torch.from_numpy(normalized[split_indices].copy()),
            "labels": torch.tensor(
                [LABEL_TO_INDEX[records[index]["label"]] for index in split_indices],
                dtype=torch.long,
            ),
            "records": [records[index] for index in split_indices],
        }
    return splits


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _scores_and_loss(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, float]:
    model.eval()
    scores: list[np.ndarray] = []
    loss_fn = nn.CrossEntropyLoss()
    loss_total = 0.0
    count = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_total += float(loss_fn(logits, labels)) * labels.size(0)
            count += labels.size(0)
            scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(scores), loss_total / count


def _training_loss(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_total = 0.0
    count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        loss_total += float(loss.detach()) * labels.size(0)
        count += labels.size(0)
    return loss_total / count


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    truth = labels == LABEL_TO_INDEX["fall"]
    predictions = scores >= threshold
    matrix = confusion_matrix(truth, predictions, labels=[False, True])
    score_distribution = {}
    for label, positive in (("non-fall", False), ("fall", True)):
        values = scores[truth == positive]
        score_distribution[label] = {
            "count": int(values.size),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return {
        "support": int(labels.size),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "fall_precision": float(precision_score(truth, predictions, zero_division=0)),
        "fall_recall": float(recall_score(truth, predictions, zero_division=0)),
        "fall_f1": float(f1_score(truth, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "confusion_matrix_nonfall_fall": matrix.tolist(),
        "score_distribution_by_actual_label": score_distribution,
    }


def _fit_batch_size(
    splits: dict[str, dict[str, Any]],
    batch_size: int,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    generator = torch.Generator().manual_seed(SEED)
    train, validation = splits["train"], splits["validation"]
    train_loader = DataLoader(
        TensorDataset(train["images"], train["labels"]),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        TensorDataset(validation["images"], validation["labels"]), batch_size=batch_size
    )

    model = CompactCsiCnn().to(device)
    class_counts = torch.bincount(train["labels"], minlength=2).float()
    class_weights = class_counts.sum() / (2.0 * class_counts)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_loss, best_epoch, stale_epochs = float("inf"), 0, 0
    best_state: dict[str, torch.Tensor] | None = None
    loss_history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        train_loss = _training_loss(model, train_loader, loss_fn, optimizer, device)
        _, validation_loss = _scores_and_loss(model, validation_loader, device)
        loss_history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        if validation_loss < best_loss:
            best_loss, best_epoch, stale_epochs = validation_loss, epoch, 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    validation_scores, _ = _scores_and_loss(model, validation_loader, device)
    threshold = select_threshold(
        np.where(validation["labels"].numpy() == 1, "fall", "non-fall"), validation_scores
    )
    validation_result = _metrics(validation["labels"].numpy(), validation_scores, threshold)
    return {
        "batch_size": batch_size,
        "best_validation_loss": best_loss,
        "best_epoch": best_epoch,
        "epochs_completed": epoch,
        "loss_history": loss_history,
        "threshold": threshold,
        "validation": validation_result,
        "state_dict": best_state,
    }


def _confusion_table(matrix: list[list[int]]) -> str:
    return (
        "<table><thead><tr><th>Actual / predicted</th><th>Non-fall</th><th>Fall</th></tr></thead>"
        "<tbody>"
        f"<tr><th>Non-fall</th><td>{matrix[0][0]}</td><td>{matrix[0][1]}</td></tr>"
        f"<tr><th>Fall</th><td>{matrix[1][0]}</td><td>{matrix[1][1]}</td></tr>"
        "</tbody></table>"
    )


def render_report(result: dict[str, Any]) -> str:
    training = result["training"]
    loss_rows = "".join(
        f"<tr><td>{entry['epoch']}</td><td>{entry['train_loss']:.6f}</td>"
        f"<td>{entry['validation_loss']:.6f}</td></tr>"
        for entry in training["loss_history"]
    )
    batch_rows = "".join(
        f"<tr><td>{candidate['batch_size']}</td><td>{candidate['best_epoch']}</td>"
        f"<td>{candidate['best_validation_loss']:.6f}</td>"
        f"<td>{candidate['validation']['balanced_accuracy']:.3f}</td>"
        f"<td>{candidate['validation']['roc_auc']:.3f}</td>"
        f"<td>{'selected' if candidate['batch_size'] == training['batch_size'] else ''}</td></tr>"
        for candidate in result["batch_comparison"]
    )

    def score_table(metrics: dict[str, Any]) -> str:
        rows = "".join(
            f"<tr><th>{html.escape(label)}</th><td>{values['count']}</td>"
            f"<td>{values['mean']:.4f}</td><td>{values['median']:.4f}</td>"
            f"<td>{values['min']:.4f}</td><td>{values['max']:.4f}</td></tr>"
            for label, values in metrics["score_distribution_by_actual_label"].items()
        )
        return (
            "<table><thead><tr><th>Actual class</th><th>Count</th><th>Mean</th>"
            "<th>Median</th><th>Min</th><th>Max</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )

    split_rows = "".join(
        f"<tr><th>{html.escape(name)}</th><td>{info['size']}</td>"
        f"<td>{info['fall']}</td><td>{info['non_fall']}</td>"
        f"<td>{html.escape(', '.join(sorted(info['rooms'])))}</td></tr>"
        for name, info in result["split"]["counts"].items()
    )
    validation, test = result["validation"], result["test"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ENetFall random-split CNN report</title><style>
body{{font:16px/1.5 system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#18212b}}
h1,h2{{color:#17365d}}table{{border-collapse:collapse;margin:1rem 0 2rem;width:100%}}
th,td{{border:1px solid #ccd3dc;padding:.5rem;text-align:right}}th:first-child{{text-align:left}}
section{{margin:2rem 0}}.summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:.7rem}}
.card{{background:#f0f4f8;padding:.8rem;border-radius:6px}}.warning{{background:#fff4d6;padding:1rem}}
</style></head><body>
<h1>ENetFall random-split compact CNN report</h1>
<p>Small CSI-specific CNN; per-event z-score normalization. Best-validation-loss checkpoint only.</p>
<div class="summary">
<div class="card"><b>Device</b><br>{html.escape(training['device'])}</div>
<div class="card"><b>Selected batch size</b><br>{training['batch_size']}</div>
<div class="card"><b>Best epoch / validation loss</b><br>{training['best_epoch']} / {training['best_validation_loss']:.6f}</div>
<div class="card"><b>Validation threshold</b><br>{result['threshold']:.6f}</div>
<div class="card"><b>LR / weight decay</b><br>{training['learning_rate']} / {training['weight_decay']}</div>
</div>
<section><h2>Batch-size comparison</h2><p>Batch size selected by lowest validation loss; test set not used for selection.</p>
<table><thead><tr><th>Batch</th><th>Best epoch</th><th>Best validation loss</th><th>Validation balanced accuracy</th><th>Validation ROC-AUC</th><th>Selection</th></tr></thead><tbody>{batch_rows}</tbody></table></section>
<section><h2>Selected model loss by epoch</h2>
<table><thead><tr><th>Epoch</th><th>Training loss</th><th>Validation loss</th></tr></thead><tbody>{loss_rows}</tbody></table></section>
<section><h2>Validation</h2>
<p>Support: {validation['support']}; balanced accuracy: {validation['balanced_accuracy']:.3f}; ROC-AUC: {validation['roc_auc']:.3f}</p>
<h3>Confusion matrix</h3>{_confusion_table(validation['confusion_matrix_nonfall_fall'])}
<h3>Fall-score distribution by actual class</h3>{score_table(validation)}</section>
<section><h2>Held-out random test</h2>
<p>Support: {test['support']}; balanced accuracy: {test['balanced_accuracy']:.3f}; ROC-AUC: {test['roc_auc']:.3f}</p>
<h3>Confusion matrix</h3>{_confusion_table(test['confusion_matrix_nonfall_fall'])}
<h3>Fall-score distribution by actual class</h3>{score_table(test)}</section>
<section><h2>Data split</h2>
<p>Stratified event-level random split: 20% held-out test; validation carved from 20% of remaining 80% development data.</p>
<table><thead><tr><th>Subset</th><th>Events</th><th>Fall</th><th>Non-fall</th><th>Rooms represented</th></tr></thead><tbody>{split_rows}</tbody></table>
<p class="warning">Random split shares rooms across train, validation, and test; room-transfer performance is not measured. ENetFall lacks reliable participant IDs. Dataset has been used in prior experiments; treat this result as exploratory.</p></section>
</body></html>
"""


def train_cnn(
    dataset_root: str | Path = "data/raw/ENetFall",
    output_dir: str | Path = "data/processed/enetfall/random-cnn-batch-comparison",
    *,
    epochs: int = 30,
    batch_sizes: tuple[int, ...] = (8, 16, 32, 60),
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-3,
    patience: int = 5,
) -> dict[str, Any]:
    if (
        epochs < 1
        or not batch_sizes
        or min(batch_sizes) < 1
        or len(set(batch_sizes)) != len(batch_sizes)
        or learning_rate <= 0
        or weight_decay < 0
        or patience < 1
    ):
        raise ValueError("invalid training parameters")
    random.seed(SEED)
    np.random.seed(SEED)
    splits = load_splits(dataset_root)
    device = _device()

    candidates = [
        _fit_batch_size(
            splits,
            batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            device=device,
        )
        for batch_size in batch_sizes
    ]
    selected = min(candidates, key=lambda candidate: candidate["best_validation_loss"])
    model = CompactCsiCnn().to(device)
    model.load_state_dict(selected["state_dict"])

    validation, test = splits["validation"], splits["test"]
    validation_loader = DataLoader(
        TensorDataset(validation["images"], validation["labels"]),
        batch_size=selected["batch_size"],
    )
    test_loader = DataLoader(
        TensorDataset(test["images"], test["labels"]), batch_size=selected["batch_size"]
    )
    validation_scores, _ = _scores_and_loss(model, validation_loader, device)
    test_scores, _ = _scores_and_loss(model, test_loader, device)
    threshold = selected["threshold"]
    validation_result = _metrics(validation["labels"].numpy(), validation_scores, threshold)
    test_result = _metrics(test["labels"].numpy(), test_scores, threshold)

    split_counts = {
        name: {
            "size": len(info["records"]),
            "fall": sum(record["label"] == "fall" for record in info["records"]),
            "non_fall": sum(record["label"] == "non-fall" for record in info["records"]),
            "rooms": dict(Counter(record["room"] for record in info["records"])),
        }
        for name, info in splits.items()
    }
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.pt"
    torch.save(selected["state_dict"], model_path)
    batch_comparison = [
        {key: value for key, value in candidate.items() if key != "state_dict"}
        for candidate in candidates
    ]
    result = {
        "run_id": "enetfall-compact-cnn-random-80-20-batch-comparison-v1",
        "model": {
            "name": "CompactCsiCnn",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "input": "3 receiver channels x 625 time samples x 30 subcarriers",
            "artifact": model_path.name,
        },
        "split": {
            "strategy": "stratified event-level random 80/20 test holdout; 20% of development subset used for validation",
            "seed": SEED,
            "counts": split_counts,
            "threshold_selection": "maximize validation balanced accuracy",
            "batch_selection": "minimum validation loss",
        },
        "batch_comparison": batch_comparison,
        "training": {
            "device": str(device),
            "seed": SEED,
            "epochs_completed": selected["epochs_completed"],
            "best_epoch": selected["best_epoch"],
            "best_validation_loss": selected["best_validation_loss"],
            "loss_history": selected["loss_history"],
            "batch_size": selected["batch_size"],
            "batch_sizes_tested": list(batch_sizes),
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "early_stopping_patience": patience,
            "class_weighted_loss": True,
        },
        "threshold": threshold,
        "validation": validation_result,
        "test": test_result,
    }
    (output / "run.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "report.html").write_text(render_report(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", nargs="?", default="data/raw/ENetFall")
    parser.add_argument(
        "output_dir", nargs="?", default="data/processed/enetfall/random-cnn-batch-comparison"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32, 60])
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    result = train_cnn(
        args.dataset_root,
        args.output_dir,
        epochs=args.epochs,
        batch_sizes=tuple(args.batch_sizes),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )
    print(json.dumps(result["batch_comparison"], indent=2))
    print(json.dumps(result["test"], indent=2))
    print(f"HTML report: {Path(args.output_dir).expanduser() / 'report.html'}")


if __name__ == "__main__":
    main()
