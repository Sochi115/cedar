"""Train a small CNN using CSI-Bench task metadata and official splits."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import random
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

WINDOW_LENGTH = 500
FEATURE_SIZE = 232
DATA_KEY = "CSI_amps"
MODEL_NAME = "csi-bench-cnn"
MODEL_ARCHITECTURE = "deep_residual_cnn_18_layer_v2"


class CsiBenchDataset(Dataset):
    """Read one HDF5 sample at a time, matching CSI-Bench preprocessing."""

    def __init__(
        self,
        task_dir: Path,
        rows: list[dict[str, str]],
        label_to_idx: dict[str, int],
        data_key: str = DATA_KEY,
    ) -> None:
        self.task_dir = task_dir
        self.rows = rows
        self.label_to_idx = label_to_idx
        self.data_key = data_key
        missing = [
            row["file_path"]
            for row in rows
            if not (task_dir / row["file_path"]).resolve().is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} CSI files missing under {task_dir}; first: {missing[0]}"
            )
        unknown = sorted({row["label"] for row in rows} - label_to_idx.keys())
        if unknown:
            raise ValueError(f"Labels absent from label_mapping.json: {unknown}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        path = (self.task_dir / row["file_path"]).resolve()
        with h5py.File(path, "r") as source:
            if self.data_key not in source:
                raise ValueError(f"{path} has no HDF5 dataset {self.data_key!r}")
            csi = np.asarray(source[self.data_key], dtype=np.float32)

        if csi.ndim != 3 or csi.shape[-1] != 1:
            raise ValueError(f"Expected CSI shape (features, time, 1), got {csi.shape} in {path}")
        if not np.isfinite(csi).all():
            raise ValueError(f"CSI contains NaN or infinity: {path}")

        # CSI-Bench stores (232 features, 500 time steps, 1), while its CNN
        # loader consumes (1 channel, 500 time steps, 232 features).
        values = torch.from_numpy(csi[:, :, 0].T.copy())
        values = (values - values.mean()) / values.std().clamp_min(1e-8)
        window = values.new_zeros((WINDOW_LENGTH, FEATURE_SIZE))
        time = min(values.shape[0], WINDOW_LENGTH)
        features = min(values.shape[1], FEATURE_SIZE)
        window[:time, :features] = values[:time, :features]
        return window.unsqueeze(0), self.label_to_idx[row["label"]]


class ResidualBlock(nn.Module):
    """Two-convolution residual block with projection when shape changes."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        nn.init.zeros_(self.norm2.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.skip(values)
        values = self.activation(self.norm1(self.conv1(values)))
        values = self.dropout(values)
        values = self.norm2(self.conv2(values))
        return self.activation(values + residual)


class DeepCnn(nn.Module):
    """ResNet-18-style CNN for CSI time-by-subcarrier maps."""

    def __init__(self, class_count: int, residual_dropout: float = 0.0) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ResidualBlock(32, 32, dropout=residual_dropout),
            ResidualBlock(32, 32, dropout=residual_dropout),
            ResidualBlock(32, 64, stride=2, dropout=residual_dropout),
            ResidualBlock(64, 64, dropout=residual_dropout),
            ResidualBlock(64, 128, stride=2, dropout=residual_dropout),
            ResidualBlock(128, 128, dropout=residual_dropout),
            ResidualBlock(128, 256, stride=2, dropout=residual_dropout),
            ResidualBlock(256, 256, dropout=residual_dropout),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, class_count),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(values))


def find_task_dir(dataset_root: Path, task: str) -> Path:
    for candidate in (dataset_root / task, dataset_root / "Multitask" / task, dataset_root):
        if (candidate / "metadata" / "sample_metadata.csv").is_file() and (
            candidate / "splits"
        ).is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find metadata and splits for {task!r} under {dataset_root}")


def load_task_datasets(
    dataset_root: Path,
    task: str,
    data_key: str = DATA_KEY,
    user_split: dict[str, list[str]] | None = None,
) -> tuple[Path, list[str], dict[str, CsiBenchDataset]]:
    task_dir = find_task_dir(dataset_root, task)
    with (task_dir / "metadata" / "label_mapping.json").open(encoding="utf-8") as source:
        label_to_idx = json.load(source)["label_to_idx"]
    class_names = [name for name, _ in sorted(label_to_idx.items(), key=lambda item: item[1])]
    if sorted(label_to_idx.values()) != list(range(len(class_names))):
        raise ValueError("label_mapping.json indices must be contiguous, starting at zero")

    with (task_dir / "metadata" / "sample_metadata.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        reader = csv.DictReader(source)
        required = {"id", "file_path", "label"} | ({"user"} if user_split is not None else set())
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Metadata must contain columns {sorted(required)}")
        rows_by_id = {row["id"]: row for row in reader}

    split_names = ("train_id", "val_id", "test_id")
    if user_split is not None:
        if set(user_split) != set(split_names) or any(
            not isinstance(user_split[name], list) for name in split_names
        ):
            raise ValueError(f"User split must contain user lists for {split_names}")
        assigned_users = [user for name in split_names for user in user_split[name]]
        observed_users = {row["user"] for row in rows_by_id.values()}
        if len(assigned_users) != len(set(assigned_users)):
            raise ValueError("A user appears in more than one split")
        if set(assigned_users) != observed_users:
            raise ValueError("User split must assign every dataset user exactly once")

    datasets: dict[str, CsiBenchDataset] = {}
    for split in split_names:
        if user_split is None:
            split_path = task_dir / "splits" / f"{split}.json"
            with split_path.open(encoding="utf-8") as source:
                sample_ids = json.load(source)
            missing = [sample_id for sample_id in sample_ids if sample_id not in rows_by_id]
            if missing:
                raise ValueError(f"{split} references {len(missing)} unknown sample IDs; first: {missing[0]}")
            rows = [rows_by_id[sample_id] for sample_id in sample_ids]
        else:
            users = set(user_split[split])
            rows = [row for row in rows_by_id.values() if row["user"] in users]
        datasets[split] = CsiBenchDataset(task_dir, rows, label_to_idx, data_key)

    split_ids = {name: {row["id"] for row in dataset.rows} for name, dataset in datasets.items()}
    if (
        split_ids["train_id"] & split_ids["val_id"]
        or split_ids["train_id"] & split_ids["test_id"]
        or split_ids["val_id"] & split_ids["test_id"]
    ):
        raise ValueError("Train/validation/test splits overlap")
    return task_dir, class_names, datasets


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.set_grad_enabled(training):
        for values, targets in loader:
            values, targets = values.to(device), targets.to(device)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            logits = model(values)
            loss = criterion(logits, targets)
            if optimizer:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * targets.size(0)
            labels.append(targets.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    if not labels:
        raise ValueError("Cannot train or evaluate on an empty split")
    return total_loss / sum(len(batch) for batch in labels), np.concatenate(labels), np.concatenate(predictions)


def _metrics(labels: np.ndarray, predictions: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    indexes = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=indexes, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, labels=indexes, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=indexes).tolist(),
        "class_names": class_names,
    }


def _image_uri(figure: Figure) -> str:
    FigureCanvasAgg(figure)
    image = io.BytesIO()
    figure.savefig(image, format="png", dpi=150, bbox_inches="tight")
    encoded = base64.b64encode(image.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_report(result: dict[str, Any]) -> str:
    """Render an offline HTML report with test confusion matrix and loss curves."""
    metrics = result["test_metrics"]
    names = metrics["class_names"]
    matrix = np.asarray(metrics["confusion_matrix"])
    figure = Figure(figsize=(6, 5))
    axis = figure.subplots()
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(names)),
        yticks=range(len(names)),
        xticklabels=names,
        yticklabels=names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Held-out test confusion matrix",
    )
    axis.tick_params(axis="x", labelrotation=30)
    for row in range(len(names)):
        for column in range(len(names)):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    confusion_uri = _image_uri(figure)

    split = result.get("split")
    split_html = ""
    if split:
        split_details = "<br>".join(
            f"{html.escape(name)}: {split['sample_counts'][name]} samples; "
            f"users: {html.escape(', '.join(split['users'][name]))}"
            for name in ("train_id", "val_id", "test_id")
        )
        split_html = (
            f"<section><h2>Data split: {html.escape(split['strategy'])}</h2>"
            f"<p>{split_details}</p></section>"
        )

    history = result["training_history"]
    figure = Figure(figsize=(8, 4))
    axis = figure.subplots()
    axis.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="Train")
    axis.plot([row["epoch"] for row in history], [row["val_loss"] for row in history], label="Validation")
    axis.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training and validation loss")
    axis.grid(alpha=0.25)
    axis.legend()
    loss_uri = _image_uri(figure)

    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>{html.escape(result['model'])} report</title>
<style>body{{font:16px system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#222}}img{{max-width:100%;height:auto}}section{{margin:2rem 0}}</style>
<h1>{html.escape(result['task'])}: {html.escape(result['model'])}</h1>
<p>Best epoch: {result['best_epoch']} · Test accuracy: {metrics['accuracy']:.4f} · Macro F1: {metrics['macro_f1']:.4f}</p>
{split_html}
<section><h2>Training loss</h2><img alt="Training and validation loss by epoch" src="{loss_uri}"></section>
<section><h2>Confusion matrix</h2><p>Rows are true classes; columns are predicted classes.</p><img alt="Held-out test confusion matrix" src="{confusion_uri}"></section>
</html>"""


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; install CUDA-enabled PyTorch and check GPU drivers")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    device = torch.device(device_name)
    device_label = torch.cuda.get_device_name(device) if device.type == "cuda" else device_name
    print(f"Device: {device_label}")

    task_dir, class_names, datasets = load_task_datasets(args.dataset_root, args.task, args.data_key)
    if any(len(datasets[name]) == 0 for name in datasets):
        raise ValueError("train_id, val_id, and test_id must all contain samples")
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train_id",
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            generator=generator if name == "train_id" else None,
        )
        for name, dataset in datasets.items()
    }

    model = DeepCnn(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{MODEL_NAME}.pt"
    best_state: dict[str, torch.Tensor] | None = None
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = _epoch(model, loaders["train_id"], criterion, device, optimizer)
        val_loss, val_labels, val_predictions = _epoch(model, loaders["val_id"], criterion, device)
        val_metrics = _metrics(val_labels, val_predictions, class_names)
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_metrics["macro_f1"],
                "learning_rate": learning_rate,
            }
        )
        print(
            f"epoch {epoch:03d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "state_dict": best_state,
                    "architecture": MODEL_ARCHITECTURE,
                    "class_names": class_names,
                    "input_shape": [1, WINDOW_LENGTH, FEATURE_SIZE],
                    "data_key": args.data_key,
                },
                model_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    test_loss, test_labels, test_predictions = _epoch(model, loaders["test_id"], criterion, device)
    result = {
        "task": args.task,
        "dataset": "CSI-Bench",
        "source": str(task_dir),
        "model": MODEL_NAME,
        "model_path": model_path.name,
        "best_epoch": best_epoch,
        "selection_metric": "validation macro F1",
        "config": {
            "architecture": MODEL_ARCHITECTURE,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "device": device_name,
            "data_key": args.data_key,
            "normalization": "per-sample global mean/std; clip or zero-pad to 500x232",
            "input_shape": [1, WINDOW_LENGTH, FEATURE_SIZE],
            "class_to_index": {name: index for index, name in enumerate(class_names)},
        },
        "sample_counts": {name: len(dataset) for name, dataset in datasets.items()},
        "training_history": history,
        "test_loss": test_loss,
        "test_metrics": _metrics(test_labels, test_predictions, class_names),
    }
    (output_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.html").write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result["test_metrics"], indent=2))
    print(f"Saved model, metrics, and report to {output_dir}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", nargs="?", type=Path, default=Path("data/raw/csi-bench"))
    parser.add_argument("--task", default="FallDetection", help="CSI-Bench task directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/csi-bench/cnn"))
    parser.add_argument("--data-key", default=DATA_KEY, help="HDF5 dataset key (CSI-Bench default: CSI_amps)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.workers < 0:
        parser.error("epochs, batch-size, and patience must be positive; workers cannot be negative")
    train(args)


if __name__ == "__main__":
    main()
