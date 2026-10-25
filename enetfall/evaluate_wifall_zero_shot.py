"""Evaluate ENetFall-trained Random Forest on held-out WiFall recordings.

This is an explicitly approximate zero-shot bridge: WiFall's single 52-bin
stream is interpolated to 30 bins, upsampled from approximately 84 to 125 Hz,
and copied into three pseudo-receiver channels expected by the ENetFall model.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.signal import resample_poly
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

from .compare_models import extract_windowed_features
from .train_random_forest import EnetFallRandomForest
from .train_svm import LABELS

SOURCE_SUBCARRIERS = 52
TARGET_SUBCARRIERS = 30
PSEUDO_RECEIVERS = 3
WIFALL_SAMPLE_RATE_HZ = 84
ENETFALL_SAMPLE_RATE_HZ = 125
CSI_COLUMN = "data"
ACTIVITY_COLUMN = "taget"


def parse_wifall_csi_field(value: str) -> np.ndarray:
    """Parse WiFall's alternating real/imaginary CSI values into amplitude."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("CSI field must be a non-empty string")
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValueError("CSI field is not a valid list") from error
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 2 * SOURCE_SUBCARRIERS:
        raise ValueError("CSI field must contain 104 real/imaginary values (52 subcarriers)")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in parsed):
        raise ValueError("CSI values must all be integers")
    values = np.asarray(parsed, dtype=np.float64)
    complex_csi = values[::2] + 1j * values[1::2]
    return np.abs(complex_csi)


def adapt_wifall_amplitude(amplitude: np.ndarray) -> np.ndarray:
    """Approximate WiFall input as ENetFall samples x 3 pseudo-RX x 30 bins."""

    values = np.asarray(amplitude, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != SOURCE_SUBCARRIERS or values.shape[0] < 4:
        raise ValueError("amplitude must have shape (at least 4 samples, 52 subcarriers)")
    if not np.isfinite(values).all():
        raise ValueError("amplitude contains non-finite values")

    source_grid = np.arange(SOURCE_SUBCARRIERS, dtype=np.float64)
    target_grid = np.linspace(0.0, SOURCE_SUBCARRIERS - 1, TARGET_SUBCARRIERS)
    mapped = np.stack([np.interp(target_grid, source_grid, row) for row in values])
    mapped = resample_poly(
        mapped,
        up=ENETFALL_SAMPLE_RATE_HZ,
        down=WIFALL_SAMPLE_RATE_HZ,
        axis=0,
    )
    mapped = np.maximum(mapped, 0.0).astype(np.float32, copy=False)
    return np.repeat(mapped[:, None, :], PSEUDO_RECEIVERS, axis=1)


def _read_wifall_recording(
    path: Path,
    expected_target: str,
    expected_rows: int | None = None,
) -> np.ndarray:
    amplitudes: list[np.ndarray] = []
    sequences: list[int] = []
    activities: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {CSI_COLUMN, ACTIVITY_COLUMN, "seq"}.issubset(
            reader.fieldnames
        ):
            raise ValueError(f"WiFall recording lacks required CSI, activity, or sequence columns: {path}")
        for row in reader:
            if not row or not any((value or "").strip() for value in row.values()):
                continue
            activity = row[ACTIVITY_COLUMN].strip()
            if not activity:
                raise ValueError(f"Missing activity label in {path}")
            activities.add(activity.casefold())
            try:
                sequences.append(int(row["seq"]))
            except ValueError as error:
                raise ValueError(f"Invalid sequence number in {path}: {row['seq']!r}") from error
            amplitudes.append(parse_wifall_csi_field(row[CSI_COLUMN]))

    if not amplitudes:
        raise ValueError(f"WiFall recording contains no samples: {path}")
    if len(activities) != 1:
        raise ValueError(f"WiFall recording contains mixed activities: {path}")
    actual_target = "fall" if next(iter(activities)) == "fall" else "non-fall"
    if actual_target != expected_target:
        raise ValueError(f"WiFall target mismatch in {path}: {actual_target} != {expected_target}")
    if any(right <= left for left, right in pairwise(sequences)):
        raise ValueError(f"WiFall sequence numbers are not strictly increasing: {path}")
    if expected_rows is not None and len(amplitudes) != expected_rows:
        raise ValueError(
            f"WiFall row-count mismatch in {path}: {len(amplitudes)} != {expected_rows}"
        )
    return np.stack(amplitudes)


def load_wifall_records(
    dataset_root: str | Path,
    split_manifest: str | Path,
    *,
    evaluation_scope: str = "test",
) -> tuple[list[dict[str, str]], np.ndarray, dict[str, Any]]:
    root = Path(dataset_root).expanduser().resolve()
    manifest = json.loads(Path(split_manifest).expanduser().read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or "recordings" not in manifest:
        raise ValueError("Unsupported or incomplete WiFall split manifest")
    if evaluation_scope not in ("test", "all"):
        raise ValueError("evaluation_scope must be 'test' or 'all'")

    records: list[dict[str, str]] = []
    feature_rows = []
    for entry in manifest["recordings"]:
        if evaluation_scope != "all" and entry.get("split") != evaluation_scope:
            continue
        relative_path = Path(entry["relative_path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe WiFall recording path in split manifest: {relative_path}")
        target = entry["target"]
        if target not in LABELS:
            raise ValueError(f"Invalid target in WiFall split manifest: {target!r}")
        amplitude = _read_wifall_recording(
            root / relative_path,
            target,
            expected_rows=entry.get("row_count"),
        )
        approximate_input = adapt_wifall_amplitude(amplitude)
        feature_rows.append(
            extract_windowed_features(approximate_input, robust_normalize=True)
        )
        records.append(
            {
                "event_id": entry["recording_id"],
                "participant_id": entry["participant_id"],
                "label": target,
            }
        )

    if not records or {record["label"] for record in records} != set(LABELS):
        raise ValueError(
            f"WiFall {evaluation_scope} selection must contain fall and non-fall recordings"
        )
    return records, np.vstack(feature_rows), manifest


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


def _evaluate(
    records: list[dict[str, str]],
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    labels = np.asarray([record["label"] for record in records], dtype=object)
    positive_scores = probabilities[:, 1]
    predictions = np.where(positive_scores >= threshold, "fall", "non-fall")
    matrix = confusion_matrix(labels, predictions, labels=LABELS)
    participant_metrics = {}
    for participant in sorted({record["participant_id"] for record in records}):
        indices = np.asarray(
            [record["participant_id"] == participant for record in records], dtype=bool
        )
        group_predictions = predictions[indices]
        tn, fp, fn, tp = confusion_matrix(
            labels[indices], group_predictions, labels=LABELS
        ).ravel()
        participant_metrics[participant] = {
            "metrics": {
                "accuracy": float(accuracy_score(labels[indices], group_predictions)),
                "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
                "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
                "f1": float(f1_score(labels[indices], group_predictions, pos_label="fall", zero_division=0)),
            }
        }
    return {
        "labels": list(LABELS),
        "observation_ids": [record["event_id"] for record in records],
        "group_ids": [record["participant_id"] for record in records],
        "y_true": labels.tolist(),
        "y_pred": predictions.tolist(),
        "scores": probabilities.tolist(),
        "positive_scores": positive_scores.tolist(),
        "score_type": "predict_proba",
        "metrics": _metrics(labels, predictions, positive_scores),
        "confusion_matrix": matrix.tolist(),
        "participant_metrics": participant_metrics,
    }


def evaluate_enetfall_random_forest_on_wifall(
    model_path: str | Path = "data/processed/enetfall/random-forest-run/model.joblib",
    dataset_root: str | Path = "data/raw/wifall/WiFall",
    split_manifest: str | Path = "data/processed/wifall/split.json",
    output_dir: str | Path | None = None,
    *,
    evaluation_scope: str = "test",
) -> dict[str, Any]:
    """Evaluate the saved ENetFall forest on WiFall test participants, without fitting."""

    model = joblib.load(Path(model_path).expanduser())
    if not isinstance(model, EnetFallRandomForest):
        raise TypeError("Model artifact is not an EnetFallRandomForest")
    if model.estimator.n_features_in_ != 216:
        raise ValueError(f"Expected 216-feature ENetFall forest, got {model.estimator.n_features_in_}")

    records, features, manifest = load_wifall_records(
        dataset_root, split_manifest, evaluation_scope=evaluation_scope
    )
    probabilities = model.estimator.predict_proba(features)
    classes = list(model.estimator.classes_)
    probabilities = probabilities[:, [classes.index("non-fall"), classes.index("fall")]]
    evaluation = _evaluate(records, probabilities, model.threshold)
    split_metadata = {
        "strategy": (
            "all WiFall recordings; zero-shot ENetFall model"
            if evaluation_scope == "all"
            else "WiFall participant-held-out test split; zero-shot ENetFall model"
        ),
        "source_split_config": manifest["split_config"],
        "source_splits_included": (
            ["train", "validation", "test"] if evaluation_scope == "all" else ["test"]
        ),
        "evaluated_participants": sorted({record["participant_id"] for record in records}),
        "evaluated_recordings": len(records),
        "threshold_source": "fixed from ENetFall validation; no WiFall labels used for fitting or threshold selection",
        "warning": (
            "This all-data aggregate includes every participant used in WiFall's original train/validation/test partitions."
            if evaluation_scope == "all"
            else "WiFall test participants only."
        ),
    }
    if evaluation_scope not in ("test", "all"):
        raise ValueError("evaluation_scope must be 'test' or 'all'")
    if output_dir is None:
        output_dir = (
            "data/processed/enetfall/wifall-zero-shot-all-run"
            if evaluation_scope == "all"
            else "data/processed/enetfall/wifall-zero-shot-run"
        )
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": 1,
        "run_id": f"enetfall-random-forest-wifall-zero-shot-approx-{evaluation_scope}-v1",
        "dataset": {
            "name": "WiFall",
            "participants": manifest["dataset"]["participants"],
            "recordings": manifest["dataset"]["recordings"],
            "evaluated_participants": len(split_metadata["evaluated_participants"]),
            "evaluated_recordings": len(records),
        },
        "task": {
            "name": "binary_fall_detection",
            "signal": "WiFi CSI amplitude",
            "labels": list(LABELS),
            "positive_label": "fall",
        },
        "model": {
            "family": "random_forest",
            "name": "ENetFall RandomForestClassifier (zero-shot)",
            "artifact": str(Path(model_path).name),
            "trained_on": "ENetFall",
            "parameters": {
                "n_estimators": model.estimator.n_estimators,
                "max_features": model.estimator.max_features,
                "min_samples_leaf": model.estimator.min_samples_leaf,
                "class_weight": model.estimator.class_weight,
                "threshold": model.threshold,
            },
        },
        "features": {
            "name": "enetfall_window_robust_temporal_v1_with_approximate_wifall_adapter",
            "count": int(model.estimator.n_features_in_),
            "source_shape": ["samples", SOURCE_SUBCARRIERS],
            "model_shape": ["samples", PSEUDO_RECEIVERS, TARGET_SUBCARRIERS],
        },
        "preprocessing": {
            "steps": [
                "WiFall complex CSI converted to amplitude",
                "linear interpolation by subcarrier index: 52 to 30",
                "polyphase time resampling: approximately 84 Hz to 125 Hz",
                "duplicate single stream into 3 pseudo-receiver channels",
                "ENetFall per-event median/MAD normalization and 216 window features",
            ],
            "approximation_warning": (
                "WiFall has one 52-subcarrier stream; ENetFall model expects three 30-subcarrier receivers. "
                "Pseudo-receiver duplication and index interpolation are not physical alignment."
            ),
        },
        "split": split_metadata,
        "evaluation_split": evaluation_scope,
        "evaluations": {evaluation_scope: evaluation},
        "metrics": evaluation["metrics"],
        "confusion_matrix": evaluation["confusion_matrix"],
    }
    with (output / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Saved zero-shot evaluation: {output / 'run.json'}")
    print(f"Fixed ENetFall threshold: {model.threshold:.6f}")
    print(f"WiFall {evaluation_scope} metrics: {evaluation['metrics']}")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_path", nargs="?", default="data/processed/enetfall/random-forest-run/model.joblib"
    )
    parser.add_argument("dataset_root", nargs="?", default="data/raw/wifall/WiFall")
    parser.add_argument("split_manifest", nargs="?", default="data/processed/wifall/split.json")
    parser.add_argument("output_dir", nargs="?", default=None)
    parser.add_argument(
        "--all-recordings",
        action="store_true",
        help="evaluate every recording, including WiFall train and validation participants",
    )
    args = parser.parse_args()
    evaluate_enetfall_random_forest_on_wifall(
        args.model_path,
        args.dataset_root,
        args.split_manifest,
        args.output_dir,
        evaluation_scope="all" if args.all_recordings else "test",
    )


if __name__ == "__main__":
    main()
