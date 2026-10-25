"""Compare compact ENetFall temporal features and binary classifiers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .train_svm import (
    EXPECTED_ROOM_COUNTS,
    ROOM_SPLITS,
    extract_event_features,
    iter_enetfall_events,
)

WINDOW_COUNT = 4
SUBCARRIER_BANDS = 6
C_VALUES = (0.1, 1.0, 10.0)
MODEL_FAMILIES = ("linear_svm", "logistic_regression", "rbf_svm")


def extract_windowed_features(
    amplitude: np.ndarray,
    *,
    robust_normalize: bool = False,
    window_count: int = WINDOW_COUNT,
) -> np.ndarray:
    """Extract compact time-ordered statistics from one three-receiver event.

    Thirty subcarriers are averaged into six contiguous bands. Each of four
    chronological windows contributes mean, standard deviation, and adjacent
    difference standard deviation for each receiver/band: 216 values total.
    Robust normalization uses only this event's per-subcarrier median and MAD.
    """

    values = np.asarray(amplitude, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 30) or values.shape[0] < window_count:
        raise ValueError(
            "amplitude must have shape (at least window_count samples, 3 receivers, 30 subcarriers)"
        )
    if window_count < 1 or 30 % SUBCARRIER_BANDS:
        raise ValueError("window_count and fixed subcarrier band count must be positive")
    if not np.isfinite(values).all():
        raise ValueError("amplitude contains non-finite values")

    if robust_normalize:
        center = np.median(values, axis=0)
        scale = 1.4826 * np.median(np.abs(values - center), axis=0)
        scale[scale < 1e-8] = 1.0
        values = (values - center) / scale

    banded = values.reshape(values.shape[0], 3, SUBCARRIER_BANDS, -1).mean(axis=-1)
    features = []
    for window in np.array_split(banded, window_count, axis=0):
        delta_std = np.diff(window, axis=0).std(axis=0) if len(window) > 1 else np.zeros((3, 6))
        features.extend(
            (
                window.mean(axis=0).ravel(),
                window.std(axis=0).ravel(),
                delta_std.ravel(),
            )
        )
    return np.concatenate(features)


def select_threshold(labels: np.ndarray, positive_scores: np.ndarray) -> float:
    """Select threshold maximizing balanced accuracy on supplied validation data."""

    y_true = np.asarray(labels) == "fall"
    scores = np.asarray(positive_scores, dtype=float)
    if y_true.ndim != 1 or scores.ndim != 1 or y_true.shape != scores.shape or not scores.size:
        raise ValueError("labels and scores must be non-empty one-dimensional arrays of equal size")
    if np.unique(y_true).size != 2 or not np.isfinite(scores).all():
        raise ValueError("threshold selection requires both labels and finite scores")

    unique_scores = np.unique(scores)
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2
    candidates = np.unique(np.concatenate((midpoints, unique_scores, np.array([0.0]))))
    best_threshold = 0.0
    best_balanced_accuracy = -1.0
    for threshold in candidates:
        balanced_accuracy = balanced_accuracy_score(y_true, scores >= threshold)
        if balanced_accuracy > best_balanced_accuracy or (
            balanced_accuracy == best_balanced_accuracy
            and abs(threshold) < abs(best_threshold)
        ):
            best_balanced_accuracy = balanced_accuracy
            best_threshold = float(threshold)
    return best_threshold


def load_experiment_features(dataset_root: str | Path) -> dict[str, dict[str, Any]]:
    splits: dict[str, dict[str, Any]] = {
        name: {"records": [], "features": {"global_raw": [], "window_raw": [], "window_robust": []}}
        for name in ("train", "validation", "test")
    }
    counts: dict[str, Counter[str]] = {room: Counter() for room in ROOM_SPLITS}
    for event in iter_enetfall_events(dataset_root):
        split = ROOM_SPLITS.get(event["room"])
        if split is None:
            raise ValueError(f"No split configured for ENetFall room {event['room']!r}")
        counts[event["room"]][event["label"]] += 1
        splits[split]["records"].append(
            {key: event[key] for key in ("event_id", "room", "label")}
        )
        features = splits[split]["features"]
        features["global_raw"].append(extract_event_features(event["amplitude"]))
        features["window_raw"].append(extract_windowed_features(event["amplitude"]))
        features["window_robust"].append(
            extract_windowed_features(event["amplitude"], robust_normalize=True)
        )

    actual_counts = {room: dict(room_counts) for room, room_counts in counts.items()}
    if actual_counts != EXPECTED_ROOM_COUNTS:
        raise ValueError(f"Unexpected ENetFall room/class counts: {actual_counts}")
    for split in splits.values():
        for feature_name, rows in split["features"].items():
            split["features"][feature_name] = np.vstack(rows)
        if not split["records"]:
            raise ValueError("Dataset must provide train, validation, and test events")
    return splits


def _build_estimator(family: str, c: float) -> Pipeline:
    if family == "linear_svm":
        classifier = SVC(C=c, kernel="linear", class_weight="balanced")
    elif family == "logistic_regression":
        classifier = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
    elif family == "rbf_svm":
        classifier = SVC(C=c, kernel="rbf", gamma="scale", class_weight="balanced")
    else:
        raise ValueError(f"Unknown model family: {family}")
    return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])


def _positive_scores(estimator: Pipeline, features: np.ndarray) -> np.ndarray:
    raw = np.asarray(estimator.decision_function(features), dtype=float)
    classes = estimator.named_steps["classifier"].classes_
    return -raw if classes[0] == "fall" else raw


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(labels) == "fall"
    predictions = scores >= threshold
    matrix = confusion_matrix(y_true, predictions, labels=[False, True])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "fall_precision": float(precision_score(y_true, predictions, zero_division=0)),
        "fall_recall": float(recall_score(y_true, predictions, zero_division=0)),
        "fall_f1": float(f1_score(y_true, predictions, zero_division=0)),
        "sensitivity": float(true_positive / (true_positive + false_negative)),
        "specificity": float(true_negative / (true_negative + false_positive)),
        "false_positive_rate": float(false_positive / (true_negative + false_positive)),
        "confusion_matrix_nonfall_fall": matrix.tolist(),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "support": int(y_true.size),
    }


def _candidate_specs() -> list[dict[str, Any]]:
    candidates = [
        {
            "id": "global_raw_rbf_c1",
            "feature_variant": "global_raw",
            "model_family": "rbf_svm",
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
        }
    ]
    for feature_variant in ("window_raw", "window_robust"):
        for family in MODEL_FAMILIES:
            for c in C_VALUES:
                kernel = "linear" if family == "linear_svm" else ("rbf" if family == "rbf_svm" else None)
                candidates.append(
                    {
                        "id": f"{feature_variant}_{family}_c{c:g}",
                        "feature_variant": feature_variant,
                        "model_family": family,
                        "C": c,
                        "kernel": kernel,
                        "gamma": "scale" if family == "rbf_svm" else None,
                    }
                )
    return candidates


def run_comparison(
    dataset_root: str | Path = "data/raw/ENetFall",
    output_dir: str | Path = "data/processed/enetfall/model-comparison",
) -> dict[str, Any]:
    """Fit fixed candidate set on train rooms; tune thresholds on validation only."""

    splits = load_experiment_features(dataset_root)
    train_labels = np.asarray([row["label"] for row in splits["train"]["records"]])
    validation_labels = np.asarray([row["label"] for row in splits["validation"]["records"]])
    test_labels = np.asarray([row["label"] for row in splits["test"]["records"]])
    results = []

    for spec in _candidate_specs():
        feature_name = spec["feature_variant"]
        estimator = _build_estimator(spec["model_family"], spec["C"])
        estimator.fit(splits["train"]["features"][feature_name], train_labels)
        validation_scores = _positive_scores(
            estimator, splits["validation"]["features"][feature_name]
        )
        test_scores = _positive_scores(estimator, splits["test"]["features"][feature_name])
        threshold = select_threshold(validation_labels, validation_scores)
        results.append(
            {
                **spec,
                "validation": {
                    "default_threshold": _metrics(validation_labels, validation_scores, 0.0),
                    "validation_selected_threshold": _metrics(
                        validation_labels, validation_scores, threshold
                    ),
                },
                "test": {
                    "default_threshold": _metrics(test_labels, test_scores, 0.0),
                    "validation_selected_threshold": _metrics(test_labels, test_scores, threshold),
                },
                "validation_selected_threshold": threshold,
            }
        )

    split_metadata = {
        "strategy": "fixed environment-held-out",
        "room_assignments": ROOM_SPLITS,
        "room_counts": {
            split: dict(sorted(Counter(row["room"] for row in values["records"]).items()))
            for split, values in splits.items()
        },
        "class_counts": {
            split: dict(sorted(Counter(row["label"] for row in values["records"]).items()))
            for split, values in splits.items()
        },
        "selection": "model and threshold candidates are compared using validation only; test is descriptive",
        "note": "No reliable participant IDs; environment-held-out does not guarantee participant independence.",
    }
    artifact = {
        "experiment_id": "enetfall-windowed-feature-model-comparison-v1",
        "dataset": {"name": "ENetFall", "event_count": 757, "source_shape_per_event": [625, 3, 30]},
        "split": split_metadata,
        "features": {
            "global_raw": "existing 450 global statistics",
            "window_raw": "4 temporal windows x 3 receivers x 6 contiguous subcarrier bands x (mean, std, delta_std) = 216",
            "window_robust": "same 216 features after per-event per-subcarrier median/MAD normalization",
            "subcarrier_bands": SUBCARRIER_BANDS,
            "window_count": WINDOW_COUNT,
        },
        "protocol": {
            "train_rooms": ["living_room_A", "meeting_room_B"],
            "validation_rooms": ["home_lab_left", "home_lab_right"],
            "test_rooms": ["lecture_room_D"],
            "training_events": len(train_labels),
            "validation_events": len(validation_labels),
            "test_events": len(test_labels),
            "scaling_fit": "train only, separately for each candidate",
            "threshold_selection": "maximize validation balanced accuracy; apply unchanged to test",
            "C_values": list(C_VALUES),
        },
        "candidates": results,
    }
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "experiment.json").open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote comparison: {output / 'experiment.json'}")
    print("Candidate                              val bAcc*  test bAcc*  test AUC")
    print("* Threshold selected on validation only")
    for result in results:
        val = result["validation"]["validation_selected_threshold"]
        test = result["test"]["validation_selected_threshold"]
        print(
            f"{result['id']:<38} "
            f"{val['balanced_accuracy']:>8.3f}  "
            f"{test['balanced_accuracy']:>10.3f}  "
            f"{test['roc_auc']:>8.3f}"
        )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", nargs="?", default="data/raw/ENetFall")
    parser.add_argument(
        "output_dir", nargs="?", default="data/processed/enetfall/model-comparison"
    )
    args = parser.parse_args()
    run_comparison(args.dataset_root, args.output_dir)


if __name__ == "__main__":
    main()
