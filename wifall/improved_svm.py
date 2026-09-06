"""Separate improved WiFall SVM training pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluation import evaluate_classifier, render_html_report, save_evaluation_result

from .audit import audit_wifall
from .loader import (
    DatasetFormatError,
    Recording,
    discover_recordings,
    load_recording,
    load_split_manifest,
)

LABELS = ("non-fall", "fall")
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class ImprovedFeatureConfig:
    """Windowed per-recording normalization and CSI feature configuration."""

    name: str = "csi_normalized_temporal_v1"
    signal: str = "normalized CSI amplitude"
    window_count: int = 4
    statistics: tuple[str, ...] = ("mean", "std", "min", "max", "delta_std")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"statistics": list(self.statistics)}


@dataclass(frozen=True)
class ImprovedSvmConfig:
    """Configuration for improved SVM selection and thresholding."""

    run_id: str = "wifall-improved-svm"
    c_values: tuple[float, ...] = (0.1, 1.0, 10.0)
    gamma_values: tuple[str, ...] = ("scale",)
    class_weight: str = "balanced"
    feature: ImprovedFeatureConfig = ImprovedFeatureConfig()
    threshold_policy: str = "maximize_validation_f1_then_sensitivity"

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "c_values": list(self.c_values),
            "gamma_values": list(self.gamma_values),
            "class_weight": self.class_weight,
            "feature": self.feature.as_dict(),
            "threshold_policy": self.threshold_policy,
        }


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    f1: float
    sensitivity: float
    specificity: float
    precision: float
    false_positive_rate: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ImprovedSvmRun:
    artifact_path: Path
    report_path: Path
    model_path: Path
    audit_path: Path
    artifact: dict[str, Any]


@dataclass
class ThresholdedSvm:
    """Persistable SVM wrapper with positive-class score orientation."""

    estimator: Pipeline
    threshold: float
    positive_label: str = "fall"
    negative_label: str = "non-fall"
    classes_: np.ndarray = field(default_factory=lambda: np.asarray(LABELS))

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.estimator.decision_function(features), dtype=float)
        classes = [str(value) for value in self.estimator.classes_]
        if classes[0] == self.positive_label:
            return -raw
        return raw

    def predict(self, features: np.ndarray) -> np.ndarray:
        scores = self.decision_function(features)
        return np.where(scores >= self.threshold, self.positive_label, self.negative_label)


@dataclass
class ImprovedWiFallSvmPipeline:
    """Persistable improved feature extraction and thresholded SVM pipeline."""

    estimator: ThresholdedSvm
    feature_config: ImprovedFeatureConfig

    def features(self, recordings: list[Recording]) -> np.ndarray:
        return np.vstack(
            [extract_improved_features(recording, self.feature_config) for recording in recordings]
        )

    def predict_recordings(self, recordings: list[Recording]) -> np.ndarray:
        return self.estimator.predict(self.features(recordings))

    def decision_function_recordings(self, recordings: list[Recording]) -> np.ndarray:
        return self.estimator.decision_function(self.features(recordings))


def _normalized_amplitude(recording: Recording) -> np.ndarray:
    amplitude = np.abs(recording.csi).astype(np.float64)
    center = np.median(amplitude, axis=0)
    scale = np.median(np.abs(amplitude - center), axis=0)
    scale[scale < 1e-8] = 1.0
    return (amplitude - center) / scale


def extract_improved_features(
    recording: Recording, config: ImprovedFeatureConfig | None = None
) -> np.ndarray:
    """Extract normalized, windowed temporal features from one recording."""

    config = config or ImprovedFeatureConfig()
    if config.window_count < 1:
        raise ValueError("window_count must be at least 1")
    expected = ("mean", "std", "min", "max", "delta_std")
    if config.statistics != expected:
        raise ValueError(f"Unsupported feature statistics: {config.statistics}")

    amplitude = _normalized_amplitude(recording)
    windows = np.array_split(amplitude, config.window_count, axis=0)
    feature_parts: list[np.ndarray] = []
    for window in windows:
        if window.shape[0] == 0:
            feature_parts.append(np.zeros(amplitude.shape[1] * len(expected)))
            continue
        delta = np.diff(window, axis=0)
        delta_std = delta.std(axis=0) if delta.size else np.zeros(window.shape[1])
        feature_parts.append(
            np.concatenate(
                (
                    window.mean(axis=0),
                    window.std(axis=0),
                    window.min(axis=0),
                    window.max(axis=0),
                    delta_std,
                )
            )
        )
    return np.concatenate(feature_parts).astype(np.float64, copy=False)


def _load_split_records(
    dataset_root: Path, manifest: dict[str, Any]
) -> dict[str, list[tuple[str, str, Recording]]]:
    references = {reference.recording_id: reference for reference in discover_recordings(dataset_root)}
    split_records: dict[str, list[tuple[str, str, Recording]]] = {split: [] for split in SPLITS}
    for entry in manifest["recordings"]:
        recording_id = entry["recording_id"]
        split = entry["split"]
        if split not in split_records:
            raise DatasetFormatError(f"Unknown split in manifest: {split!r}")
        reference = references.get(recording_id)
        if reference is None:
            raise DatasetFormatError(f"Manifest recording is missing from dataset: {recording_id}")
        recording = load_recording(reference)
        if recording.target != entry["target"] or recording.row_count != entry["row_count"]:
            raise DatasetFormatError(f"Manifest metadata mismatch: {recording_id}")
        split_records[split].append((reference.participant_id, reference.activity, recording))
    return split_records


def _validate_participant_splits(
    split_records: dict[str, list[tuple[str, str, Recording]]]
) -> None:
    participants = {
        split: {participant for participant, _, _ in records}
        for split, records in split_records.items()
    }
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            if participants[left].intersection(participants[right]):
                raise DatasetFormatError("Participant appears in multiple evaluation splits")


def _oriented_scores(estimator: Pipeline, features: np.ndarray) -> np.ndarray:
    raw = np.asarray(estimator.decision_function(features), dtype=float)
    classes = [str(value) for value in estimator.classes_]
    return -raw if classes[0] == "fall" else raw


def _threshold_candidates(scores: np.ndarray) -> np.ndarray:
    values = np.unique(scores)
    if values.size == 1:
        margin = max(abs(float(values[0])) * 1e-6, 1e-6)
        return np.asarray([values[0] - margin, values[0], values[0] + margin])
    midpoints = (values[:-1] + values[1:]) / 2
    margin = max(abs(float(values[0])) * 1e-6, 1e-6)
    return np.concatenate(([values[0] - margin], midpoints, [values[-1] + margin]))


def _threshold_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> ThresholdSelection:
    predictions = np.where(scores >= threshold, "fall", "non-fall")
    matrix = confusion_matrix(labels, predictions, labels=LABELS)
    true_negative, false_positive, _, _ = matrix.ravel()
    negative_total = true_negative + false_positive
    return ThresholdSelection(
        threshold=float(threshold),
        f1=float(f1_score(labels, predictions, pos_label="fall", zero_division=0)),
        sensitivity=float(recall_score(labels, predictions, pos_label="fall", zero_division=0)),
        specificity=float(true_negative / negative_total) if negative_total else 0.0,
        precision=float(precision_score(labels, predictions, pos_label="fall", zero_division=0)),
        false_positive_rate=float(false_positive / negative_total) if negative_total else 0.0,
    )


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[ThresholdSelection, list[dict[str, float]]]:
    """Select validation threshold by F1, then sensitivity."""

    candidates = [_threshold_metrics(labels, scores, threshold) for threshold in _threshold_candidates(scores)]
    selected = max(candidates, key=lambda result: (result.f1, result.sensitivity, -result.false_positive_rate))
    return selected, [candidate.as_dict() for candidate in candidates]


def _fit_estimator(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    c: float,
    gamma: str,
    class_weight: str,
) -> Pipeline:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(C=c, gamma=gamma, kernel="rbf", class_weight=class_weight),
            ),
        ]
    )
    pipeline.fit(train_features, train_labels)
    return pipeline


def _participant_results(
    split: str,
    records: list[tuple[str, str, Recording]],
    features: np.ndarray,
    model: ThresholdedSvm,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for index, (participant, _, _) in enumerate(records):
        groups.setdefault(participant, []).append(index)
    results = {}
    for participant, indices in sorted(groups.items()):
        result = evaluate_classifier(
            model,
            features[indices],
            np.asarray([records[index][2].target for index in indices], dtype=object),
            label_order=LABELS,
            positive_label="fall",
            run_id=metadata["run_id"],
            split_name=split,
            dataset_metadata=metadata["dataset"],
            task_metadata=metadata["task"],
            model_metadata=metadata["model"],
            feature_metadata=metadata["features"],
            preprocessing_metadata=metadata["preprocessing"],
            split_metadata=metadata["split"],
            group_ids=[participant] * len(indices),
        )
        results[participant] = result["evaluations"][split]
    return results


def _score_distribution(
    scores: np.ndarray, records: list[tuple[str, str, Recording]]
) -> dict[str, Any]:
    by_participant: dict[str, list[float]] = {}
    for score, (participant, _, _) in zip(scores, records):
        by_participant.setdefault(participant, []).append(float(score))
    return {
        "overall": {
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
        },
        "by_participant": {
            participant: {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
            for participant, values in sorted(by_participant.items())
        },
    }


def train_improved_svm(
    dataset_root: str | Path,
    split_manifest: str | Path,
    output_dir: str | Path,
    config: ImprovedSvmConfig | None = None,
) -> ImprovedSvmRun:
    """Train improved SVM and persist model, result, audit, and HTML report."""

    config = config or ImprovedSvmConfig()
    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_split_manifest(split_manifest)
    split_records = _load_split_records(root, manifest)
    _validate_participant_splits(split_records)
    audit = audit_wifall(root)
    audit_path = output / "dataset_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    train_records = split_records["train"]
    validation_records = split_records["validation"]
    if not train_records or not validation_records:
        raise DatasetFormatError("Improved SVM requires train and validation recordings")
    train_features = np.vstack(
        [extract_improved_features(record, config.feature) for _, _, record in train_records]
    )
    validation_features = np.vstack(
        [extract_improved_features(record, config.feature) for _, _, record in validation_records]
    )
    train_labels = np.asarray([record.target for _, _, record in train_records], dtype=object)
    validation_labels = np.asarray(
        [record.target for _, _, record in validation_records], dtype=object
    )

    candidates = []
    for c in config.c_values:
        for gamma in config.gamma_values:
            estimator = _fit_estimator(
                train_features, train_labels, c, gamma, config.class_weight
            )
            validation_scores = _oriented_scores(estimator, validation_features)
            threshold, threshold_table = select_threshold(validation_labels, validation_scores)
            candidates.append(
                {
                    "c": c,
                    "gamma": gamma,
                    "threshold": threshold.threshold,
                    "threshold_metrics": threshold.as_dict(),
                    "threshold_table": threshold_table,
                    "validation_f1": threshold.f1,
                    "validation_sensitivity": threshold.sensitivity,
                }
            )
    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["validation_f1"],
            candidate["validation_sensitivity"],
        ),
    )
    final_estimator = _fit_estimator(
        train_features,
        train_labels,
        selected["c"],
        selected["gamma"],
        config.class_weight,
    )
    thresholded = ThresholdedSvm(final_estimator, selected["threshold"])
    feature_pipeline = ImprovedWiFallSvmPipeline(thresholded, config.feature)
    model_path = output / "model.joblib"
    joblib.dump(feature_pipeline, model_path)

    model_metadata = {
        "family": "svm",
        "name": "improved_thresholded_rbf_svm",
        "training_code": "wifall.improved_svm",
        "artifact": model_path.name,
        "parameters": {
            "C": selected["c"],
            "gamma": selected["gamma"],
            "kernel": "rbf",
            "class_weight": config.class_weight,
        },
        "decision_threshold": selected["threshold"],
        "threshold_policy": config.threshold_policy,
    }
    feature_metadata = config.feature.as_dict() | {"count": int(train_features.shape[1])}
    preprocessing_metadata = {
        "steps": [
            "per-recording CSI amplitude median/MAD normalization",
            "four temporal windows",
            "StandardScaler",
            "RBF SVC",
        ],
        "fit_split": "train",
    }
    split_metadata = {
        "strategy": "participant-held-out",
        "config": manifest["split_config"],
        "participants": manifest["participants"],
        "counts": manifest["counts"],
    }
    metadata = {
        "run_id": config.run_id,
        "dataset": manifest["dataset"],
        "task": {"name": "binary_fall_detection", "signal": "WiFi CSI"},
        "model": model_metadata,
        "features": feature_metadata,
        "preprocessing": preprocessing_metadata,
        "split": split_metadata,
    }
    evaluations: dict[str, dict[str, Any]] = {}
    for split, records in split_records.items():
        if not records:
            continue
        features = np.vstack(
            [extract_improved_features(record, config.feature) for _, _, record in records]
        )
        labels = np.asarray([record.target for _, _, record in records], dtype=object)
        result = evaluate_classifier(
            thresholded,
            features,
            labels,
            label_order=LABELS,
            positive_label="fall",
            run_id=config.run_id,
            split_name=split,
            dataset_metadata=manifest["dataset"],
            task_metadata=metadata["task"],
            model_metadata=model_metadata,
            feature_metadata=feature_metadata,
            preprocessing_metadata=preprocessing_metadata,
            split_metadata=split_metadata,
            observation_ids=[record.reference.recording_id for _, _, record in records],
            group_ids=[participant for participant, _, _ in records],
        )
        evaluation = result["evaluations"][split]
        scores = np.asarray(evaluation["scores"], dtype=float)
        evaluation["participant_metrics"] = _participant_results(
            split, records, features, thresholded, metadata
        )
        evaluation["score_distribution"] = _score_distribution(scores, records)
        evaluation["threshold_metrics"] = (
            selected["threshold_table"] if split == "validation" else []
        )
        evaluations[split] = evaluation

    primary_split = "test" if "test" in evaluations else "validation"
    primary = evaluations[primary_split]
    artifact = result | {
        "evaluation_split": primary_split,
        "evaluations": evaluations,
        "metrics": primary["metrics"],
        "confusion_matrix": primary["confusion_matrix"],
        "data_audit": audit,
        "selection": {
            "candidates": candidates,
            "selected": selected,
            "selection_split": "validation",
        },
    }
    artifact_path = output / "run.json"
    save_evaluation_result(artifact, artifact_path)
    report_path = output / "report.html"
    report_path.write_text(render_html_report(artifact), encoding="utf-8")
    return ImprovedSvmRun(
        artifact_path=artifact_path,
        report_path=report_path,
        model_path=model_path,
        audit_path=audit_path,
        artifact=artifact,
    )


def load_improved_svm(path: str | Path) -> ImprovedWiFallSvmPipeline:
    """Load an improved SVM pipeline."""

    pipeline = joblib.load(Path(path).expanduser())
    if not isinstance(pipeline, ImprovedWiFallSvmPipeline):
        raise TypeError("Model artifact is not an ImprovedWiFallSvmPipeline")
    return pipeline
