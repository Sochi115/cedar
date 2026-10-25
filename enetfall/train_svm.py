"""Train an environment-held-out RBF SVM on native-resolution ENetFall events."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.io import loadmat
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

LABELS = ("non-fall", "fall")
MAT_VARIABLE = "dataset_CSI_t"
LABEL_VARIABLE = "dataset_labels"
MAT_FILES = (
    ("dataset_living_room.mat", "living_room_A"),
    ("dataset_meeting_room.mat", "meeting_room_B"),
    ("dataset_home_lab(L).mat", "home_lab_left"),
    ("dataset_home_lab(R).mat", "home_lab_right"),
    ("dataset_lecture_room.mat", "lecture_room_D"),
)
ROOM_SPLITS = {
    "living_room_A": "train",
    "meeting_room_B": "train",
    "home_lab_left": "validation",
    "home_lab_right": "validation",
    "lecture_room_D": "test",
}
EXPECTED_ROOM_COUNTS = {
    "living_room_A": {"fall": 91, "non-fall": 157},
    "meeting_room_B": {"fall": 58, "non-fall": 76},
    "home_lab_left": {"fall": 40, "non-fall": 45},
    "home_lab_right": {"fall": 36, "non-fall": 49},
    "lecture_room_D": {"fall": 96, "non-fall": 109},
}
FEATURE_STATISTICS = ("mean", "std", "min", "max", "delta_std")


def iter_enetfall_events(dataset_root: str | Path) -> Iterator[dict[str, Any]]:
    """Yield native-resolution labeled events from ENetFall MAT files."""

    root = Path(dataset_root).expanduser().resolve()
    for filename, room in MAT_FILES:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"ENetFall MAT file not found: {path}")
        data = loadmat(path, variable_names=[MAT_VARIABLE, LABEL_VARIABLE])
        if MAT_VARIABLE not in data or LABEL_VARIABLE not in data:
            raise ValueError(f"{filename} must contain {MAT_VARIABLE} and {LABEL_VARIABLE}")

        csi = np.asarray(data[MAT_VARIABLE])
        labels = np.asarray(data[LABEL_VARIABLE]).reshape(-1)
        expected_shape = (labels.size, 625, 90)
        if csi.shape != expected_shape:
            raise ValueError(f"{filename} has CSI shape {csi.shape}; expected {expected_shape}")
        if not np.issubdtype(csi.dtype, np.number) or not np.isfinite(csi).all():
            raise ValueError(f"{filename} CSI must contain finite numeric amplitudes")
        if not np.isin(labels, (0, 1)).all():
            raise ValueError(f"{filename} labels must be 0 (non-fall) or 1 (fall)")

        # Source columns interleave three receivers for each of 30 subcarriers.
        streams = csi.reshape(labels.size, 625, 30, 3).transpose(0, 1, 3, 2)
        for index, raw_label in enumerate(labels):
            yield {
                "event_id": f"{Path(filename).stem}:{index:04d}",
                "room": room,
                "label": "fall" if int(raw_label) == 1 else "non-fall",
                # Clip negative interpolation artifacts; source signal is amplitude.
                "amplitude": np.maximum(streams[index], 0.0).astype(np.float32, copy=True),
            }


def extract_event_features(amplitude: np.ndarray) -> np.ndarray:
    """Summarize each native receiver/subcarrier stream into one event vector.

    Input shape is samples x 3 receivers x 30 subcarriers. Statistics are
    computed across time, then concatenated statistic-first, yielding 450
    features. No samples from other events are used by this transform.
    """

    values = np.asarray(amplitude, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 30) or values.shape[0] < 2:
        raise ValueError(
            "amplitude must have shape (at least 2 samples, 3 receivers, 30 subcarriers)"
        )
    if not np.isfinite(values).all():
        raise ValueError("amplitude contains non-finite values")

    delta_std = np.diff(values, axis=0).std(axis=0)
    return np.concatenate(
        (
            values.mean(axis=0).ravel(),
            values.std(axis=0).ravel(),
            values.min(axis=0).ravel(),
            values.max(axis=0).ravel(),
            delta_std.ravel(),
        )
    )


def load_features(dataset_root: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Read events once, validate published counts, and group by fixed split."""

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for event in iter_enetfall_events(dataset_root):
        split = ROOM_SPLITS.get(event["room"])
        if split is None:
            raise ValueError(f"No split configured for ENetFall room {event['room']!r}")
        splits[split].append(
            {
                "event_id": event["event_id"],
                "room": event["room"],
                "label": event["label"],
                "features": extract_event_features(event["amplitude"]),
            }
        )

    actual_counts = {
        room: dict(
            Counter(
                record["label"]
                for records in splits.values()
                for record in records
                if record["room"] == room
            )
        )
        for room in ROOM_SPLITS
    }
    if actual_counts != EXPECTED_ROOM_COUNTS:
        raise ValueError(f"Unexpected ENetFall room/class counts: {actual_counts}")
    if {record["label"] for record in splits["train"]} != set(LABELS):
        raise ValueError("Training rooms must contain both fall and non-fall events")
    return splits


def _evaluate(
    estimator: Pipeline,
    records: list[dict[str, Any]],
    split: str,
    common_metadata: dict[str, Any],
) -> dict[str, Any]:
    features = np.vstack([record["features"] for record in records])
    y_true = np.asarray([record["label"] for record in records], dtype=object)
    y_pred = estimator.predict(features)
    raw_scores = np.asarray(estimator.decision_function(features), dtype=float)
    positive_scores = -raw_scores if estimator.classes_[0] == "fall" else raw_scores

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )[:3]
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
    )[:3]
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label="fall", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label="fall", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label="fall", zero_division=0)),
        "support": int(y_true.size),
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
        "roc_auc": float(roc_auc_score(y_true == "fall", positive_scores)),
        "pr_auc": float(average_precision_score(y_true == "fall", positive_scores)),
    }
    evaluation = {
        "labels": list(LABELS),
        "observation_ids": [record["event_id"] for record in records],
        "group_ids": [record["room"] for record in records],
        "y_true": y_true.tolist(),
        "y_pred": np.asarray(y_pred).tolist(),
        "scores": raw_scores.tolist(),
        "positive_scores": positive_scores.tolist(),
        "score_type": "decision_function",
        "metrics": metrics,
        "confusion_matrix": matrix.tolist(),
    }
    return {**common_metadata, "evaluation_split": split, "evaluations": {split: evaluation}}


def train_svm(
    dataset_root: str | Path = "data/raw/ENetFall",
    output_dir: str | Path = "data/processed/enetfall/rbf-svm-run",
) -> dict[str, Any]:
    """Fit on training rooms only; evaluate fixed validation and test rooms."""

    splits = load_features(dataset_root)
    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced"),
            ),
        ]
    )
    train_records = splits["train"]
    x_train = np.vstack([record["features"] for record in train_records])
    y_train = np.asarray([record["label"] for record in train_records], dtype=object)
    estimator.fit(x_train, y_train)

    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.joblib"
    joblib.dump(estimator, model_path)

    room_counts = {
        split: dict(sorted(Counter(record["room"] for record in records).items()))
        for split, records in splits.items()
    }
    class_counts = {
        split: dict(sorted(Counter(record["label"] for record in records).items()))
        for split, records in splits.items()
    }
    split_metadata = {
        "strategy": "fixed environment-held-out",
        "room_assignments": ROOM_SPLITS,
        "room_counts": room_counts,
        "class_counts": class_counts,
        "seed": None,
        "note": (
            "No random event split: all events from a nominal environment stay together. "
            "Participant IDs are unavailable, so participant independence is not guaranteed."
        ),
    }
    model_metadata = {
        "family": "svm",
        "name": "SVC",
        "artifact": model_path.name,
        "parameters": {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "class_weight": "balanced",
        },
    }
    feature_metadata = {
        "name": "enetfall_receiver_subcarrier_statistics_v1",
        "signal": "native ENetFall CSI amplitude",
        "statistics": list(FEATURE_STATISTICS),
        "receiver_count": 3,
        "subcarrier_count": 30,
        "features_per_event": int(x_train.shape[1]),
        "aggregation": "per receiver and subcarrier across 625 time samples",
    }
    preprocessing_metadata = {
        "steps": ["event-level amplitude statistics", "StandardScaler", "RBF SVC"],
        "fit_split": "train rooms only",
    }
    dataset_metadata = {
        "name": "ENetFall",
        "source_format": "MAT, dataset_CSI_t and dataset_labels",
        "event_count": sum(len(records) for records in splits.values()),
        "source_shape_per_event": [625, 3, 30],
    }
    task_metadata = {
        "name": "binary_fall_detection",
        "signal": "WiFi CSI amplitude",
        "labels": list(LABELS),
        "positive_label": "fall",
    }
    common_metadata = {
        "schema_version": 1,
        "run_id": "enetfall-rbf-svm-environment-held-out-v1",
        "dataset": dataset_metadata,
        "task": task_metadata,
        "model": model_metadata,
        "features": feature_metadata,
        "preprocessing": preprocessing_metadata,
        "split": split_metadata,
    }
    evaluations = {}
    for split in ("validation", "test"):
        evaluations[split] = _evaluate(
            estimator, splits[split], split, common_metadata
        )["evaluations"][split]

    test_evaluation = evaluations["test"]
    artifact = {
        **common_metadata,
        "evaluation_split": "test",
        "evaluations": evaluations,
        "metrics": test_evaluation["metrics"],
        "confusion_matrix": test_evaluation["confusion_matrix"],
    }
    with (output / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", nargs="?", default="data/raw/ENetFall")
    parser.add_argument(
        "output_dir", nargs="?", default="data/processed/enetfall/rbf-svm-run"
    )
    args = parser.parse_args()
    artifact = train_svm(args.dataset_root, args.output_dir)
    print(f"Saved ENetFall RBF SVM artifacts to {Path(args.output_dir).expanduser()}")
    print(f"Held-out test metrics: {artifact['metrics']}")


if __name__ == "__main__":
    main()
