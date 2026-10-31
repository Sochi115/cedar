"""Bounded training replica of data/raw/ENetFall/train.py."""

from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.transforms import Normalize

from .train_svm import EXPECTED_ROOM_COUNTS, iter_enetfall_events

SEED = 42
BATCH_SIZE = 60
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
DESIRED_ACCURACY = 0.94


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(*, pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.SiLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2),
    )
    return model


def _load_random_split(
    dataset_root: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    events = list(iter_enetfall_events(dataset_root))
    actual_counts: dict[str, dict[str, int]] = {}
    for room in sorted({event["room"] for event in events}):
        actual_counts[room] = {
            label: sum(event["room"] == room and event["label"] == label for event in events)
            for label in ("fall", "non-fall")
        }
    if actual_counts != EXPECTED_ROOM_COUNTS:
        raise ValueError(f"Unexpected ENetFall room/class counts: {actual_counts}")

    values = torch.from_numpy(
        np.stack([event["amplitude"].transpose(1, 0, 2) for event in events]).astype(np.float32)
    )
    labels = torch.tensor([1 if event["label"] == "fall" else 0 for event in events])
    # Preserve downloaded script's ImageNet channel normalization and train-only global scaling.
    values = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(values)

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(values), generator=generator)
    test_size = int(np.ceil(len(values) * 0.2))
    train_indices, test_indices = indices[:-test_size], indices[-test_size:]
    train_values = values[train_indices]
    maximum = train_values.max()
    if not torch.isfinite(maximum) or maximum.abs() < 1e-8:
        raise ValueError("Training split produced invalid normalization maximum")
    values = values / maximum
    mean = values[train_indices].mean()
    values = values - mean
    preprocessing = {
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
        "training_max_after_imagenet_normalization": float(maximum),
        "training_mean_after_global_scaling": float(mean),
    }

    return (
        values[train_indices],
        labels[train_indices],
        values[test_indices],
        labels[test_indices],
        preprocessing,
    )


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    losses = []
    labels_all = []
    scores_all = []
    loss_fn = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            losses.append(float(loss_fn(logits, labels)) * labels.size(0))
            labels_all.append(labels.cpu().numpy())
            scores_all.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    y_true = np.concatenate(labels_all)
    scores = np.concatenate(scores_all)
    y_pred = scores >= 0.5
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "loss": float(sum(losses) / y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "confusion_matrix_nonfall_fall": matrix.tolist(),
        "support": int(y_true.size),
        "labels": y_true.tolist(),
        "scores": scores.tolist(),
    }


def _report(result: dict[str, Any]) -> str:
    history = "".join(
        f"<tr><td>{row['epoch']}</td><td>{row['learning_rate']:.2e}</td>"
        f"<td>{row['train_loss']:.5f}</td><td>{row['holdout_loss']:.5f}</td>"
        f"<td>{row['holdout_accuracy']:.3f}</td></tr>"
        for row in result["history"]
    )
    matrix = result["holdout"]["confusion_matrix_nonfall_fall"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Downloaded ENetFall training report</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#18212b}}
h1,h2{{color:#17365d}}table{{border-collapse:collapse;margin:1rem 0 2rem;width:100%}}
th,td{{border:1px solid #ccd3dc;padding:.5rem;text-align:right}}th:first-child{{text-align:left}}
.card{{background:#f0f4f8;padding:1rem;border-radius:6px}}.warning{{background:#fff4d6;padding:1rem}}</style></head><body>
<h1>Downloaded ENetFall EfficientNet-B0 training</h1>
<p>Bounded replica of <code>data/raw/ENetFall/train.py</code>; pretrained ImageNet EfficientNet-B0.</p>
<div class="card"><b>Epochs:</b> {result['training']['epochs_completed']}<br>
<b>Device:</b> {html.escape(result['training']['device'])}<br>
<b>Train / holdout:</b> {result['training']['train_count']} / {result['training']['holdout_count']}<br>
<b>Holdout accuracy:</b> {result['holdout']['accuracy']:.3f}<br>
<b>Balanced accuracy:</b> {result['holdout']['balanced_accuracy']:.3f}<br>
<b>ROC-AUC:</b> {result['holdout']['roc_auc']:.3f}</div>
<h2>Epoch history</h2><table><thead><tr><th>Epoch</th><th>LR</th><th>Train loss</th><th>Holdout loss</th><th>Holdout accuracy</th></tr></thead><tbody>{history}</tbody></table>
<h2>Final holdout confusion matrix</h2><p>Rows actual; columns predicted. Class order: non-fall, fall. Support: {result['holdout']['support']}.</p>
<table><thead><tr><th>Actual / predicted</th><th>Non-fall</th><th>Fall</th></tr></thead><tbody>
<tr><th>Non-fall</th><td>{matrix[0][0]}</td><td>{matrix[0][1]}</td></tr>
<tr><th>Fall</th><td>{matrix[1][0]}</td><td>{matrix[1][1]}</td></tr></tbody></table>
<p class="warning">Downloaded script randomly mixes all rooms in its 80/20 holdout and uses that holdout for epoch monitoring. This is not a room-independent test. Replica fixes its unbounded retraining loop, steps StepLR once per epoch (instead of every minibatch), and evaluates all holdout samples (no dropped final batch).</p>
</body></html>"""


def train_downloaded(
    dataset_root: str | Path = "data/raw/ENetFall",
    output_dir: str | Path = "data/processed/enetfall/downloaded-train-run",
    *,
    epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    pretrained: bool = True,
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid training parameters")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_values, train_labels, test_values, test_labels, preprocessing = _load_random_split(
        dataset_root
    )
    train_loader = DataLoader(
        TensorDataset(train_values, train_labels), batch_size=batch_size, shuffle=True, drop_last=True
    )
    holdout_loader = DataLoader(
        TensorDataset(test_values, test_labels), batch_size=batch_size, shuffle=False
    )
    device = get_default_device()
    model = build_model(pretrained=pretrained).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.detach()) * labels.size(0)
            train_count += labels.size(0)

        scheduler.step()
        holdout = _evaluate(model, holdout_loader, device)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss": train_loss_total / train_count,
                "holdout_loss": holdout["loss"],
                "holdout_accuracy": holdout["accuracy"],
            }
        )
        print(
            f"Epoch {epoch}/{epochs}: train_loss={history[-1]['train_loss']:.4f}, "
            f"holdout_loss={holdout['loss']:.4f}, holdout_acc={holdout['accuracy']:.4f}"
        )
        if epoch > 40 and holdout["accuracy"] >= DESIRED_ACCURACY:
            break

    # One final checkpoint; original threshold loop could otherwise restart training forever.
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.pt"
    torch.save(model.state_dict(), model_path)
    final_holdout = _evaluate(model, holdout_loader, device)
    result = {
        "run_id": "enetfall-downloaded-train-replica-v1",
        "dataset": {"name": "ENetFall", "event_count": 757, "source_shape": [625, 3, 30]},
        "model": {
            "name": "EfficientNet-B0",
            "initialization": "ImageNet pretrained" if pretrained else "random",
            "classifier": [1280, 512, 256, 2],
            "checkpoint": model_path.name,
        },
        "training": {
            "device": str(device),
            "seed": SEED,
            "epochs_completed": len(history),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer": "Adam",
            "scheduler": "StepLR(step_size=10, gamma=0.1), stepped once per epoch",
            "train_count": int(train_labels.numel()),
            "holdout_count": int(test_labels.numel()),
            "early_stop_target_accuracy_after_epoch_40": DESIRED_ACCURACY,
        },
        "preprocessing": preprocessing,
        "split": {
            "strategy": "seeded random event-level 80/20 split, not stratified",
            "seed": SEED,
            "warning": "All rooms occur in both train and holdout; holdout is monitored every epoch.",
        },
        "history": history,
        "holdout": final_holdout,
    }
    (output / "run.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "report.html").write_text(_report(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", nargs="?", default="data/raw/ENetFall")
    parser.add_argument("output_dir", nargs="?", default="data/processed/enetfall/downloaded-train-run")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    result = train_downloaded(
        args.dataset_root,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        pretrained=not args.no_pretrained,
    )
    holdout = result["holdout"]
    print(
        json.dumps(
            {
                key: holdout[key]
                for key in ("support", "loss", "accuracy", "balanced_accuracy", "roc_auc", "confusion_matrix_nonfall_fall")
            },
            indent=2,
        )
    )
    print(f"HTML report: {Path(args.output_dir).expanduser() / 'report.html'}")


if __name__ == "__main__":
    main()
