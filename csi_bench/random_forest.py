"""Train and report a Random Forest on CSI-Bench task splits."""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)

from .cnn import FEATURE_SIZE, WINDOW_LENGTH, load_task_datasets

WINDOW_COUNT = 4
SUBCARRIER_BANDS = 16
MODEL_NAME = "csi-bench-random-forest"


def extract_features(values: np.ndarray) -> np.ndarray:
    """Extract compact robust-normalized temporal features from a time-by-CSI map."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (WINDOW_LENGTH, FEATURE_SIZE):
        raise ValueError(
            f"Expected CSI shape {(WINDOW_LENGTH, FEATURE_SIZE)}, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("CSI contains NaN or infinity")

    center = np.median(values, axis=0)
    scale = np.median(np.abs(values - center), axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (values - center) / scale
    bands = [
        band.mean(axis=1)
        for band in np.array_split(normalized, SUBCARRIER_BANDS, axis=1)
    ]
    banded = np.stack(bands, axis=1)

    features = []
    for window in np.array_split(banded, WINDOW_COUNT, axis=0):
        delta_std = (
            np.diff(window, axis=0).std(axis=0)
            if len(window) > 1
            else np.zeros(SUBCARRIER_BANDS)
        )
        features.extend((window.mean(axis=0), window.std(axis=0), delta_std))
    return np.concatenate(features)


def build_random_forest(n_jobs: int = 2) -> RandomForestClassifier:
    """Build a reproducible forest while leaving CPU headroom by default."""
    return RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=n_jobs,
    )


def _metrics(
    labels: np.ndarray, predictions: np.ndarray, class_names: list[str]
) -> dict[str, Any]:
    indexes = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels, predictions, labels=indexes, average="macro", zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                labels, predictions, labels=indexes, average="weighted", zero_division=0
            )
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=indexes
        ).tolist(),
        "class_names": class_names,
    }


def _fit_with_loss_curve(
    estimator: RandomForestClassifier,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    class_count: int,
) -> list[dict[str, float]]:
    tree_count = estimator.n_estimators
    checkpoints = np.unique(np.linspace(1, tree_count, min(20, tree_count), dtype=int))
    estimator.set_params(n_estimators=int(checkpoints[0]), warm_start=True)
    history = []
    for count in checkpoints:
        estimator.set_params(n_estimators=int(count))
        estimator.fit(train_features, train_labels)
        classes = list(range(class_count))
        history.append(
            {
                "trees": int(count),
                "train_loss": float(
                    log_loss(
                        train_labels,
                        estimator.predict_proba(train_features),
                        labels=classes,
                    )
                ),
                "val_loss": float(
                    log_loss(
                        validation_labels,
                        estimator.predict_proba(validation_features),
                        labels=classes,
                    )
                ),
            }
        )
    estimator.set_params(warm_start=False)
    return history


def _image_uri(figure: Figure) -> str:
    FigureCanvasAgg(figure)
    image = io.BytesIO()
    figure.savefig(image, format="png", dpi=150, bbox_inches="tight")
    return "data:image/png;base64," + base64.b64encode(image.getvalue()).decode("ascii")


def render_report(result: dict[str, Any]) -> str:
    """Render a self-contained held-out confusion matrix and forest loss curve."""
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

    history = result["training_history"]
    figure = Figure(figsize=(8, 4))
    axis = figure.subplots()
    axis.plot(
        [row["trees"] for row in history],
        [row["train_loss"] for row in history],
        label="Training",
    )
    axis.plot(
        [row["trees"] for row in history],
        [row["val_loss"] for row in history],
        label="Validation",
    )
    axis.set(
        xlabel="Number of trees",
        ylabel="Log loss",
        title="Training and validation loss",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    loss_uri = _image_uri(figure)

    split = result.get("split")
    split_html = ""
    if split:
        split_details = "<br>".join(
            f"{html.escape(name)}: {split['sample_counts'][name]} samples; "
            f"users: {html.escape(', '.join(split.get('users', {}).get(name, [])))}"
            for name in ("train_id", "val_id", "test_id")
        )
        split_html = (
            f"<section><h2>Data split: {html.escape(split['strategy'])}</h2>"
            f"<p>{split_details}</p></section>"
        )

    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>{html.escape(result["model"])} report</title>
<style>body{{font:16px system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#222}}img{{max-width:100%;height:auto}}section{{margin:2rem 0}}</style>
<h1>{html.escape(result["task"])}: {html.escape(result["model"])}</h1>
<p>Test accuracy: {metrics["accuracy"]:.4f} · Macro F1: {metrics["macro_f1"]:.4f}</p>
{split_html}
<section><h2>Training and validation loss</h2><img alt="Training and validation log loss by number of trees" src="{loss_uri}"></section>
<section><h2>Confusion matrix</h2><p>Rows are true labels; columns are predicted labels.</p><img alt="Held-out test confusion matrix" src="{confusion_uri}"></section>
</html>"""


def train(args: argparse.Namespace) -> dict[str, Any]:
    user_split_path = getattr(args, "user_split", None)
    user_split = None
    if user_split_path is not None:
        with user_split_path.open(encoding="utf-8") as source:
            user_split = json.load(source)
    load_options = {"user_split": user_split} if user_split is not None else {}
    task_dir, class_names, datasets = load_task_datasets(
        args.dataset_root, args.task, args.data_key, **load_options
    )
    split_info = {
        "strategy": "user-held-out" if user_split is not None else "official-sample-id",
        "users": {
            name: sorted({row["user"] for row in getattr(dataset, "rows", [])})
            for name, dataset in datasets.items()
        },
        "sample_counts": {name: len(dataset) for name, dataset in datasets.items()},
    }
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError("train_id, val_id, and test_id must all contain samples")

    features: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for split, dataset in datasets.items():
        split_features = []
        split_labels = []
        for index in range(len(dataset)):
            values, label = dataset[index]
            split_features.append(extract_features(values.squeeze(0).numpy()))
            split_labels.append(label)
        features[split] = np.vstack(split_features)
        labels[split] = np.asarray(split_labels, dtype=np.int64)

    if set(labels["train_id"].tolist()) != set(range(len(class_names))):
        raise ValueError(
            "Training split must contain at least one sample for every mapped class"
        )

    estimator = build_random_forest(args.jobs)
    training_history = _fit_with_loss_curve(
        estimator,
        features["train_id"],
        labels["train_id"],
        features["val_id"],
        labels["val_id"],
        len(class_names),
    )
    validation_metrics = _metrics(
        labels["val_id"], estimator.predict(features["val_id"]), class_names
    )
    test_metrics = _metrics(
        labels["test_id"], estimator.predict(features["test_id"]), class_names
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if user_split is not None:
        (output_dir / "user_split.json").write_text(
            json.dumps(user_split, indent=2) + "\n", encoding="utf-8"
        )
    model_path = output_dir / f"{MODEL_NAME}.joblib"
    joblib.dump(
        {
            "estimator": estimator,
            "class_names": class_names,
            "feature_config": {
                "normalization": "per-sample, per-subcarrier median/MAD",
                "temporal_windows": WINDOW_COUNT,
                "contiguous_subcarrier_bands": SUBCARRIER_BANDS,
                "statistics_per_window": ["mean", "std", "delta_std"],
            },
        },
        model_path,
    )
    result = {
        "task": args.task,
        "dataset": "CSI-Bench",
        "source": str(task_dir),
        "model": MODEL_NAME,
        "model_path": model_path.name,
        "selection": "fixed Random Forest configuration; validation metrics are reported without tuning",
        "config": {
            "n_estimators": estimator.n_estimators,
            "max_features": estimator.max_features,
            "min_samples_leaf": estimator.min_samples_leaf,
            "class_weight": estimator.class_weight,
            "random_state": estimator.random_state,
            "n_jobs": estimator.n_jobs,
            "data_key": args.data_key,
            "split_strategy": split_info["strategy"],
            "user_split_manifest": "user_split.json" if user_split is not None else None,
            "feature_count": int(features["train_id"].shape[1]),
        },
        "sample_counts": {name: len(dataset) for name, dataset in datasets.items()},
        "split": split_info,
        "training_history": training_history,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    (output_dir / "run.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report_path = output_dir / "report.html"
    report_path.write_text(render_report(result), encoding="utf-8")
    print(json.dumps(test_metrics, indent=2))
    print(f"Saved model, metrics, and report to {output_dir}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root", nargs="?", type=Path, default=Path("data/raw/csi-bench")
    )
    parser.add_argument(
        "--task", default="FallDetection", help="CSI-Bench task directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/csi-bench/random-forest"),
    )
    parser.add_argument("--data-key", default="CSI_amps", help="HDF5 dataset key")
    parser.add_argument(
        "--user-split",
        type=Path,
        help="JSON user assignment with train_id, val_id, and test_id arrays",
    )
    parser.add_argument("--jobs", type=int, default=2, help="CPU workers (default: 2)")
    args = parser.parse_args()
    if args.jobs == 0 or args.jobs < -1:
        parser.error("jobs must be -1 or a positive integer")
    train(args)


if __name__ == "__main__":
    main()
