"""Train an ENetFall random forest on compact robust temporal features."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .compare_models import load_experiment_features, select_threshold
from .train_svm import LABELS, ROOM_SPLITS

RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.5
FEATURE_VARIANT = "window_robust"


def build_random_forest() -> RandomForestClassifier:
    """Create fixed, reproducible forest configuration for this dataset size."""

    return RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


@dataclass
class EnetFallRandomForest:
    """Persistable forest, event feature transform, and validation threshold."""

    estimator: RandomForestClassifier
    threshold: float

    def positive_scores(self, features: np.ndarray) -> np.ndarray:
        classes = list(self.estimator.classes_)
        return self.estimator.predict_proba(features)[:, classes.index("fall")]

    def predict_amplitude(self, amplitude: np.ndarray) -> str:
        from .compare_models import extract_windowed_features

        features = extract_windowed_features(amplitude, robust_normalize=True)[None, :]
        return "fall" if self.positive_scores(features)[0] >= self.threshold else "non-fall"


def _metrics(labels: np.ndarray, predictions: np.ndarray, positive_scores: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=LABELS, zero_division=0
    )
    macro = precision_recall_fscore_support(
        labels, predictions, labels=LABELS, average="macro", zero_division=0
    )[:3]
    weighted = precision_recall_fscore_support(
        labels, predictions, labels=LABELS, average="weighted", zero_division=0
    )[:3]
    matrix = confusion_matrix(labels, predictions, labels=LABELS)
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, pos_label="fall", zero_division=0)),
        "recall": float(recall_score(labels, predictions, pos_label="fall", zero_division=0)),
        "f1": float(f1_score(labels, predictions, pos_label="fall", zero_division=0)),
        "support": int(labels.size),
        "positive_label": "fall",
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "macro_avg": dict(zip(("precision", "recall", "f1"), map(float, macro))),
        "weighted_avg": dict(zip(("precision", "recall", "f1"), map(float, weighted))),
        "sensitivity": float(true_positive / (true_positive + false_negative)),
        "specificity": float(true_negative / (true_negative + false_positive)),
        "false_positive_rate": float(false_positive / (true_negative + false_positive)),
        "roc_auc": float(roc_auc_score(labels == "fall", positive_scores)),
        "pr_auc": float(average_precision_score(labels == "fall", positive_scores)),
    }


def _evaluate_split(
    estimator: RandomForestClassifier,
    records: list[dict[str, Any]],
    features: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    labels = np.asarray([record["label"] for record in records], dtype=object)
    probabilities = estimator.predict_proba(features)
    fall_index = list(estimator.classes_).index("fall")
    positive_scores = probabilities[:, fall_index]
    default_predictions = np.where(
        positive_scores >= DEFAULT_THRESHOLD, "fall", "non-fall"
    )
    predictions = np.where(positive_scores >= threshold, "fall", "non-fall")
    metrics = _metrics(labels, predictions, positive_scores)
    default_metrics = _metrics(labels, default_predictions, positive_scores)
    matrix = confusion_matrix(labels, predictions, labels=LABELS)
    return {
        "labels": list(LABELS),
        "observation_ids": [record["event_id"] for record in records],
        "group_ids": [record["room"] for record in records],
        "y_true": labels.tolist(),
        "y_pred": predictions.tolist(),
        "scores": probabilities.tolist(),
        "positive_scores": positive_scores.tolist(),
        "score_type": "predict_proba",
        "threshold": threshold,
        "default_threshold_metrics": default_metrics,
        "metrics": metrics,
        "confusion_matrix": matrix.tolist(),
        "threshold_metrics": [
            {
                "threshold": DEFAULT_THRESHOLD,
                "precision": default_metrics["precision"],
                "sensitivity": default_metrics["sensitivity"],
                "specificity": default_metrics["specificity"],
                "f1": default_metrics["f1"],
            },
            {
                "threshold": threshold,
                "precision": metrics["precision"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "f1": metrics["f1"],
            },
        ],
    }


def train_random_forest(
    dataset_root: str | Path = "data/raw/ENetFall",
    output_dir: str | Path = "data/processed/enetfall/random-forest-run",
) -> dict[str, Any]:
    """Fit on training rooms, select threshold on validation, evaluate test."""

    splits = load_experiment_features(dataset_root)
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]
    x_train = train["features"][FEATURE_VARIANT]
    x_validation = validation["features"][FEATURE_VARIANT]
    x_test = test["features"][FEATURE_VARIANT]
    y_train = np.asarray([row["label"] for row in train["records"]], dtype=object)
    y_validation = np.asarray([row["label"] for row in validation["records"]], dtype=object)

    estimator = build_random_forest()
    estimator.fit(x_train, y_train)
    validation_scores = estimator.predict_proba(x_validation)[:, list(estimator.classes_).index("fall")]
    threshold = select_threshold(y_validation, validation_scores)

    model = EnetFallRandomForest(estimator=estimator, threshold=threshold)
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.joblib"
    joblib.dump(model, model_path)

    validation_result = _evaluate_split(estimator, validation["records"], x_validation, threshold)
    test_result = _evaluate_split(estimator, test["records"], x_test, threshold)
    room_counts = {
        split: dict(sorted(Counter(record["room"] for record in values["records"]).items()))
        for split, values in splits.items()
    }
    class_counts = {
        split: dict(sorted(Counter(record["label"] for record in values["records"]).items()))
        for split, values in splits.items()
    }
    split_metadata = {
        "strategy": "fixed environment-held-out",
        "room_assignments": ROOM_SPLITS,
        "room_counts": room_counts,
        "class_counts": class_counts,
        "threshold_selection": "maximize validation balanced accuracy",
        "threshold": threshold,
        "note": "Participant IDs are unavailable; this split does not guarantee participant independence.",
    }
    model_metadata = {
        "family": "random_forest",
        "name": "RandomForestClassifier",
        "artifact": model_path.name,
        "parameters": {
            "n_estimators": estimator.n_estimators,
            "max_features": estimator.max_features,
            "min_samples_leaf": estimator.min_samples_leaf,
            "class_weight": estimator.class_weight,
            "random_state": RANDOM_STATE,
        },
    }
    dataset_metadata = {
        "name": "ENetFall",
        "source_format": "MAT, dataset_CSI_t and dataset_labels",
        "event_count": sum(len(values["records"]) for values in splits.values()),
        "source_shape_per_event": [625, 3, 30],
    }
    feature_metadata = {
        "name": "enetfall_window_robust_temporal_v1",
        "signal": "native ENetFall CSI amplitude",
        "description": "4 chronological windows; 6 contiguous subcarrier bands; mean, std, and delta std per receiver/band/window",
        "robust_normalization": "per-event, per-receiver, per-subcarrier median and 1.4826*MAD",
        "features_per_event": int(x_train.shape[1]),
        "fit_split": "no learned feature transform; each event normalized from its own samples",
    }
    preprocessing_metadata = {
        "steps": ["per-event robust normalization", "windowed temporal statistics", "RandomForestClassifier"],
        "fit_split": "train rooms only",
        "scaler": "none (tree model)",
    }
    test_evaluation = test_result
    artifact = {
        "schema_version": 1,
        "run_id": "enetfall-random-forest-window-robust-v1",
        "dataset": dataset_metadata,
        "task": {
            "name": "binary_fall_detection",
            "signal": "WiFi CSI amplitude",
            "labels": list(LABELS),
            "positive_label": "fall",
        },
        "model": model_metadata,
        "features": feature_metadata,
        "preprocessing": preprocessing_metadata,
        "split": split_metadata,
        "evaluation_split": "test",
        "evaluations": {"validation": validation_result, "test": test_result},
        "metrics": test_evaluation["metrics"],
        "confusion_matrix": test_evaluation["confusion_matrix"],
    }
    with (output / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Saved model: {model_path}")
    print(f"Saved evaluation: {output / 'run.json'}")
    print(f"Validation-selected fall threshold: {threshold:.6f}")
    print(f"Held-out test metrics: {test_result['metrics']}")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", nargs="?", default="data/raw/ENetFall")
    parser.add_argument(
        "output_dir", nargs="?", default="data/processed/enetfall/random-forest-run"
    )
    args = parser.parse_args()
    train_random_forest(args.dataset_root, args.output_dir)


if __name__ == "__main__":
    main()
