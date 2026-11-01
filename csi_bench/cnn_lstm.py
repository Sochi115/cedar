"""Train a CSI-Bench CNN encoder followed by a temporal LSTM."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .cnn import (
    DATA_KEY,
    FEATURE_SIZE,
    WINDOW_LENGTH,
    DeepCnn,
    _epoch,
    _metrics,
    load_task_datasets,
    render_report,
)

MODEL_NAME = "csi-bench-cnn-lstm"
MODEL_ARCHITECTURE = "deep_residual_cnn_lstm_v3_residual_dropout"
CNN_CHANNELS = 256
CNN_RESIDUAL_DROPOUT = 0.1
SEQUENCE_DROPOUT = 0.25
HEAD_DROPOUT = 0.4


class DeepCnnLstm(nn.Module):
    """Use CSI CNN features as a chronological sequence for one LSTM layer."""

    def __init__(self, class_count: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.cnn = DeepCnn(class_count, residual_dropout=CNN_RESIDUAL_DROPOUT)
        self.cnn.features[-1] = nn.Identity()
        self.cnn.classifier = nn.Identity()
        self.sequence_dropout = nn.Dropout(SEQUENCE_DROPOUT)
        self.lstm = nn.LSTM(
            input_size=CNN_CHANNELS,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head_dropout = nn.Dropout(HEAD_DROPOUT)
        self.classifier = nn.Linear(hidden_size, class_count)

    def encode_sequence(self, values: torch.Tensor) -> torch.Tensor:
        """Return batch × time × channels features; retain temporal order."""
        feature_map = self.cnn.features(values)
        sequence = feature_map.mean(dim=-1).transpose(1, 2)
        return self.sequence_dropout(sequence)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequence = self.encode_sequence(values)
        _, (hidden, _) = self.lstm(sequence)
        return self.classifier(self.head_dropout(hidden[-1]))


def _device(name: str) -> torch.device:
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    user_split = None
    if args.user_split is not None:
        with args.user_split.open(encoding="utf-8") as source:
            user_split = json.load(source)
    task_dir, class_names, datasets = load_task_datasets(
        args.dataset_root, args.task, args.data_key, user_split=user_split
    )
    split_info = {
        "strategy": "user-held-out" if user_split is not None else "official-sample-id",
        "users": {
            name: sorted({row["user"] for row in dataset.rows})
            for name, dataset in datasets.items()
        },
        "sample_counts": {name: len(dataset) for name, dataset in datasets.items()},
    }
    if any(len(dataset) == 0 for dataset in datasets.values()):
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
    model = DeepCnnLstm(len(class_names), args.hidden_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if user_split is not None:
        (output_dir / "user_split.json").write_text(
            json.dumps(user_split, indent=2) + "\n", encoding="utf-8"
        )
    model_path = output_dir / f"{MODEL_NAME}.pt"
    best_state: dict[str, torch.Tensor] | None = None
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = _epoch(
            model, loaders["train_id"], criterion, device, optimizer
        )
        val_loss, val_labels, val_predictions = _epoch(
            model, loaders["val_id"], criterion, device
        )
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
            f"epoch {epoch:03d}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            torch.save(
                {
                    "state_dict": best_state,
                    "architecture": MODEL_ARCHITECTURE,
                    "class_names": class_names,
                    "input_shape": [1, WINDOW_LENGTH, FEATURE_SIZE],
                    "lstm_hidden_size": args.hidden_size,
                    "dropout": {
                        "cnn_residual_branch": CNN_RESIDUAL_DROPOUT,
                        "sequence": SEQUENCE_DROPOUT,
                        "classifier_head": HEAD_DROPOUT,
                    },
                    "temporal_steps": 32,
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
    test_loss, test_labels, test_predictions = _epoch(
        model, loaders["test_id"], criterion, device
    )
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
            "device": str(device),
            "data_key": args.data_key,
            "input_shape": [1, WINDOW_LENGTH, FEATURE_SIZE],
            "temporal_steps": 32,
            "lstm_hidden_size": args.hidden_size,
            "dropout": {
                "cnn_residual_branch": CNN_RESIDUAL_DROPOUT,
                "sequence": SEQUENCE_DROPOUT,
                "classifier_head": HEAD_DROPOUT,
            },
            "scheduler": {
                "name": "ReduceLROnPlateau",
                "monitor": "val_loss",
                "factor": 0.5,
                "patience": 3,
                "min_learning_rate": 1e-5,
            },
            "class_to_index": {name: index for index, name in enumerate(class_names)},
            "split_strategy": split_info["strategy"],
            "user_split_manifest": "user_split.json" if user_split is not None else None,
        },
        "split": split_info,
        "sample_counts": {name: len(dataset) for name, dataset in datasets.items()},
        "training_history": history,
        "test_loss": test_loss,
        "test_metrics": _metrics(test_labels, test_predictions, class_names),
    }
    (output_dir / "run.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.html").write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result["test_metrics"], indent=2))
    print(f"Saved model, metrics, and report to {output_dir}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root", nargs="?", type=Path, default=Path("data/raw/csi-bench")
    )
    parser.add_argument("--task", default="FallDetection")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/csi-bench/cnn-lstm")
    )
    parser.add_argument("--data-key", default=DATA_KEY)
    parser.add_argument(
        "--user-split",
        type=Path,
        help="JSON user assignment with train_id, val_id, and test_id arrays",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience, args.hidden_size) < 1:
        parser.error("epochs, batch-size, patience, and hidden-size must be positive")
    if args.workers < 0:
        parser.error("workers cannot be negative")
    train(args)


if __name__ == "__main__":
    main()
