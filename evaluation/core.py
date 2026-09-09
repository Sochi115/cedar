"""Generic classification validation and result artifact handling."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
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


class EvaluationError(ValueError):
    """Raised when model validation input or result data is invalid."""


def _builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_builtin(item) for item in value]
    return value


def _classes(model: Any) -> list[Any] | None:
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None
    return [_builtin(value) for value in np.asarray(classes).tolist()]


def _positive_scores(
    raw_scores: np.ndarray,
    model_classes: list[Any] | None,
    positive_label: Any,
) -> np.ndarray | None:
    if raw_scores.ndim == 1:
        scores = raw_scores.astype(float, copy=False)
        if model_classes and len(model_classes) == 2 and model_classes[0] == positive_label:
            return -scores
        return scores
    if raw_scores.ndim != 2 or not model_classes or positive_label not in model_classes:
        return None
    return raw_scores[:, model_classes.index(positive_label)].astype(float, copy=False)


def _model_scores(
    model: Any, features: Any, positive_label: Any
) -> tuple[Any, str | None, np.ndarray | None]:
    model_classes = _classes(model)
    decision_function = getattr(model, "decision_function", None)
    if callable(decision_function):
        raw_scores = np.asarray(decision_function(features))
        return (
            _builtin(raw_scores),
            "decision_function",
            _positive_scores(raw_scores, model_classes, positive_label),
        )

    predict_proba = getattr(model, "predict_proba", None)
    if callable(predict_proba):
        raw_scores = np.asarray(predict_proba(features))
        return (
            _builtin(raw_scores),
            "predict_proba",
            _positive_scores(raw_scores, model_classes, positive_label),
        )

    return None, None, None


def _metric_averages(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[Any]
) -> tuple[dict[str, float], dict[str, float]]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    return (
        {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1": float(macro_f1),
        },
        {
            "precision": float(weighted_precision),
            "recall": float(weighted_recall),
            "f1": float(weighted_f1),
        },
    )


def _metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[Any],
    positive_label: Any | None,
    positive_scores: np.ndarray | None,
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        str(label): {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    macro, weighted = _metric_averages(y_true, y_pred, labels)
    binary_metrics: dict[str, float | None] = {
        "sensitivity": None,
        "specificity": None,
        "false_positive_rate": None,
        "roc_auc": None,
        "pr_auc": None,
    }

    if len(labels) == 2:
        positive_label = positive_label if positive_label is not None else labels[-1]
        if positive_label not in labels:
            raise EvaluationError(f"Positive label is not in labels: {positive_label!r}")
        negative_label = labels[0] if labels[1] == positive_label else labels[1]
        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=[negative_label, positive_label],
        )
        true_negative, false_positive, _, _ = matrix.ravel()
        denominator = true_negative + false_positive
        binary_metrics["sensitivity"] = float(
            recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        )
        binary_metrics["specificity"] = (
            float(true_negative / denominator) if denominator else 0.0
        )
        binary_metrics["false_positive_rate"] = (
            float(false_positive / denominator) if denominator else 0.0
        )
        if positive_scores is not None and np.unique(y_true).size == 2:
            binary_metrics["roc_auc"] = float(roc_auc_score(y_true == positive_label, positive_scores))
            binary_metrics["pr_auc"] = float(
                average_precision_score(y_true == positive_label, positive_scores)
            )
        average = "binary"
        average_kwargs = {"pos_label": positive_label}
    else:
        average = "weighted"
        average_kwargs = {}

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average=average, zero_division=0, **average_kwargs)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average=average, zero_division=0, **average_kwargs)
        ),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0, **average_kwargs)),
        "support": int(y_true.size),
        "positive_label": _builtin(positive_label),
        "per_class": per_class,
        "macro_avg": macro,
        "weighted_avg": weighted,
        **binary_metrics,
    }


def evaluate_classifier(
    model: Any,
    features: Any,
    labels: Sequence[Any],
    *,
    label_order: Sequence[Any],
    positive_label: Any | None = None,
    run_id: str = "evaluation",
    split_name: str = "validation",
    dataset_metadata: Mapping[str, Any] | None = None,
    task_metadata: Mapping[str, Any] | None = None,
    model_metadata: Mapping[str, Any] | None = None,
    feature_metadata: Mapping[str, Any] | None = None,
    preprocessing_metadata: Mapping[str, Any] | None = None,
    split_metadata: Mapping[str, Any] | None = None,
    observation_ids: Sequence[Any] | None = None,
    group_ids: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Evaluate any classifier without fitting it.

    ``model`` must expose ``predict``. It may expose ``decision_function`` or
    ``predict_proba`` to provide optional score-based metrics.
    """

    if not callable(getattr(model, "predict", None)):
        raise EvaluationError("Model must provide a predict method")
    ordered_labels = [_builtin(label) for label in label_order]
    if not ordered_labels or len(set(ordered_labels)) != len(ordered_labels):
        raise EvaluationError("label_order must contain unique labels")
    y_true = np.asarray([_builtin(label) for label in labels], dtype=object)
    if y_true.size == 0:
        raise EvaluationError("Validation labels cannot be empty")
    if any(label not in ordered_labels for label in y_true.tolist()):
        raise EvaluationError("Validation labels contain values outside label_order")

    y_pred = np.asarray([_builtin(label) for label in model.predict(features)], dtype=object)
    if y_pred.shape != y_true.shape:
        raise EvaluationError("Model predictions and validation labels have different lengths")
    if any(label not in ordered_labels for label in y_pred.tolist()):
        raise EvaluationError("Model predictions contain values outside label_order")

    if len(ordered_labels) == 2 and positive_label is None:
        positive_label = ordered_labels[-1]
    scores, score_type, positive_scores = _model_scores(model, features, positive_label)
    evaluation_metrics = _metrics(y_true, y_pred, ordered_labels, positive_label, positive_scores)
    matrix = confusion_matrix(y_true, y_pred, labels=ordered_labels)
    evaluation: dict[str, Any] = {
        "labels": ordered_labels,
        "observation_ids": _builtin(observation_ids) if observation_ids is not None else None,
        "group_ids": _builtin(group_ids) if group_ids is not None else None,
        "y_true": _builtin(y_true),
        "y_pred": _builtin(y_pred),
        "scores": scores,
        "positive_scores": _builtin(positive_scores) if positive_scores is not None else None,
        "score_type": score_type,
        "metrics": evaluation_metrics,
        "confusion_matrix": matrix.tolist(),
    }

    task = {"name": "classification"}
    task.update(_builtin(task_metadata or {}))
    task["labels"] = ordered_labels
    if positive_label is not None:
        task["positive_label"] = positive_label
    artifact = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset": _builtin(dataset_metadata or {}),
        "task": task,
        "model": _builtin(model_metadata or {}),
        "features": _builtin(feature_metadata or {}),
        "preprocessing": _builtin(preprocessing_metadata or {}),
        "split": _builtin(split_metadata or {}),
        "evaluation_split": split_name,
        "evaluations": {split_name: evaluation},
        "metrics": evaluation_metrics,
        "confusion_matrix": matrix.tolist(),
    }
    return artifact


def save_evaluation_result(result: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write one generic evaluation result artifact as JSON."""

    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_builtin(result), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return destination


def load_evaluation_result(path: str | Path) -> dict[str, Any]:
    """Load and validate the generic evaluation result envelope."""

    source = Path(path).expanduser()
    with source.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    required = ("dataset", "task", "model", "features", "split", "evaluations")
    if result.get("schema_version") != 1:
        raise EvaluationError("Unsupported evaluation result schema version")
    missing = [key for key in required if key not in result]
    if missing:
        raise EvaluationError(f"Evaluation result is missing: {', '.join(missing)}")
    return result
