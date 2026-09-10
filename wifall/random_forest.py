"""Separate Random Forest training pipeline for WiFall fall detection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
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
from sklearn.model_selection import LeaveOneGroupOut

from evaluation import (
    analyze_hard_examples,
    evaluate_classifier,
    render_html_report,
    save_evaluation_result,
)

from .audit import audit_wifall
from .augmentation import CsiAugmentationConfig, augment_split_records
from .diagnostics import recording_observations
from .improved_svm import ImprovedFeatureConfig
from .loader import DatasetFormatError, Recording, load_split_manifest
from .training import load_split_records, validate_participant_splits

LABELS = ("non-fall", "fall")
_BASELINE_STATISTICS = ("mean", "std", "min", "max", "delta_std")
_MOTION_STATISTICS = _BASELINE_STATISTICS + (
    "delta_mean_abs",
    "slope",
    "energy",
    "peak_to_peak",
    "change_point",
)


@dataclass(frozen=True)
class RandomForestFeatureConfig:
    """Configurable normalized CSI temporal representation for Random Forest.

    ``window_count`` controls temporal resolution. ``statistics`` controls the
    ordered per-window features, including baseline summaries and motion
    features such as slope, energy, and change-point location.
    """

    name: str = "csi_normalized_temporal_v1"
    signal: str = "normalized CSI amplitude"
    window_count: int = 4
    statistics: tuple[str, ...] = _BASELINE_STATISTICS

    def __post_init__(self) -> None:
        if self.window_count < 1:
            raise ValueError("window_count must be at least 1")
        unsupported = set(self.statistics).difference(
            _BASELINE_STATISTICS + _MOTION_STATISTICS
        )
        if unsupported:
            raise ValueError(f"Unsupported feature statistics: {sorted(unsupported)}")
        if len(set(self.statistics)) != len(self.statistics):
            raise ValueError("statistics must not contain duplicates")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "signal": self.signal,
            "window_count": self.window_count,
            "statistics": list(self.statistics),
        }


def _default_motion_feature_config() -> RandomForestFeatureConfig:
    return RandomForestFeatureConfig(
        name="csi_normalized_temporal_motion_v2",
        window_count=8,
        statistics=_MOTION_STATISTICS,
    )


@dataclass(frozen=True)
class RandomForestConfig:
    """Configuration for Random Forest selection and thresholding."""

    run_id: str = "wifall-random-forest"
    n_estimators: tuple[int, ...] = (100,)
    max_depth: tuple[int | None, ...] = (None, 16)
    min_samples_leaf: tuple[int, ...] = (1, 4)
    max_features: tuple[str | float | None, ...] = ("sqrt",)
    max_samples: tuple[float | None, ...] = (None, 0.75)
    criterion: tuple[str, ...] = ("gini", "entropy")
    class_weight: str = "balanced"
    random_state: int = 42
    feature: RandomForestFeatureConfig = field(default_factory=RandomForestFeatureConfig)
    feature_variants: tuple[RandomForestFeatureConfig, ...] = ()
    calibration_method: str = "sigmoid"
    threshold_policy: str = "maximize_grouped_validation_f1_then_sensitivity"
    minimum_recall: float = 0.9
    augmentation: CsiAugmentationConfig = field(default_factory=CsiAugmentationConfig)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "n_estimators": list(self.n_estimators),
            "max_depth": list(self.max_depth),
            "min_samples_leaf": list(self.min_samples_leaf),
            "max_features": list(self.max_features),
            "max_samples": list(self.max_samples),
            "criterion": list(self.criterion),
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "feature": self.feature.as_dict(),
            "feature_variants": [variant.as_dict() for variant in self.feature_configs()],
            "calibration_method": self.calibration_method,
            "threshold_policy": self.threshold_policy,
            "minimum_recall": self.minimum_recall,
            "augmentation": self.augmentation.as_dict(),
        }

    def feature_configs(self) -> tuple[RandomForestFeatureConfig, ...]:
        if self.feature_variants:
            return self.feature_variants
        if self.feature == RandomForestFeatureConfig():
            return (self.feature, _default_motion_feature_config())
        return (self.feature,)


@dataclass(frozen=True)
class RandomForestRun:
    """Paths and result artifact produced by Random Forest training."""

    artifact_path: Path
    report_path: Path
    model_path: Path
    audit_path: Path
    artifact: dict[str, Any]


@dataclass
class ScoreCalibrator:
    """Persistable calibration model for positive-class Random Forest scores."""

    method: str
    estimator: LogisticRegression | IsotonicRegression

    def transform(self, scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=float)
        if self.method == "sigmoid":
            return self.estimator.predict_proba(values.reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return np.asarray(self.estimator.predict(values), dtype=float)
        raise DatasetFormatError(f"Unsupported score calibration method: {self.method!r}")


@dataclass
class ThresholdedRandomForest:
    """Persistable Random Forest wrapper with calibrated fall-probability scores."""

    estimator: RandomForestClassifier
    threshold: float
    calibrator: ScoreCalibrator | None = None
    positive_label: str = "fall"
    negative_label: str = "non-fall"
    classes_: np.ndarray = field(default_factory=lambda: np.asarray(LABELS))

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        """Return probability of the explicitly configured positive class."""

        probabilities = np.asarray(self.estimator.predict_proba(features), dtype=float)
        estimator_classes = [str(value) for value in self.estimator.classes_]
        try:
            positive_index = estimator_classes.index(self.positive_label)
        except ValueError as error:
            raise DatasetFormatError("Random Forest has no fall class") from error
        scores = probabilities[:, positive_index]
        return self.calibrator.transform(scores) if self.calibrator else scores

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return probabilities in the stable ``non-fall``, ``fall`` order."""

        scores = self.decision_function(features)
        return np.column_stack((1.0 - scores, scores))

    def predict(self, features: np.ndarray) -> np.ndarray:
        scores = self.decision_function(features)
        return np.where(scores >= self.threshold, self.positive_label, self.negative_label)


@dataclass
class WiFallRandomForestPipeline:
    """Persistable improved-feature extraction and thresholded Random Forest."""

    estimator: ThresholdedRandomForest
    feature_config: RandomForestFeatureConfig

    def features(self, recordings: list[Recording]) -> np.ndarray:
        return np.vstack(
            [extract_random_forest_features(recording, self.feature_config) for recording in recordings]
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


def _feature_config(
    config: RandomForestFeatureConfig | ImprovedFeatureConfig | None,
) -> RandomForestFeatureConfig:
    if config is None:
        return RandomForestFeatureConfig()
    if isinstance(config, RandomForestFeatureConfig):
        return config
    return RandomForestFeatureConfig(
        name=config.name,
        signal=config.signal,
        window_count=config.window_count,
        statistics=config.statistics,
    )


def _window_statistics(window: np.ndarray) -> dict[str, np.ndarray]:
    subcarriers = window.shape[1]
    if window.shape[0] == 0:
        return {
            statistic: np.zeros(subcarriers)
            for statistic in _MOTION_STATISTICS
        }
    delta = np.diff(window, axis=0)
    delta_abs = np.abs(delta)
    if window.shape[0] > 1:
        positions = np.linspace(-1.0, 1.0, window.shape[0])
        centered = positions - positions.mean()
        denominator = float(np.dot(centered, centered))
        slope = centered @ window / denominator if denominator else np.zeros(subcarriers)
        change_index = float(np.argmax(delta_abs.mean(axis=1))) / max(delta.shape[0] - 1, 1)
    else:
        slope = np.zeros(subcarriers)
        change_index = 0.0
    return {
        "mean": window.mean(axis=0),
        "std": window.std(axis=0),
        "min": window.min(axis=0),
        "max": window.max(axis=0),
        "delta_std": delta.std(axis=0) if delta.size else np.zeros(subcarriers),
        "delta_mean_abs": delta_abs.mean(axis=0) if delta_abs.size else np.zeros(subcarriers),
        "slope": slope,
        "energy": np.mean(np.square(window), axis=0),
        "peak_to_peak": window.max(axis=0) - window.min(axis=0),
        "change_point": np.full(subcarriers, change_index),
    }


def extract_random_forest_features(
    recording: Recording,
    config: RandomForestFeatureConfig | ImprovedFeatureConfig | None = None,
) -> np.ndarray:
    """Extract normalized temporal CSI features for one recording."""

    feature_config = _feature_config(config)
    amplitude = _normalized_amplitude(recording)
    windows = np.array_split(amplitude, feature_config.window_count, axis=0)
    feature_parts = [
        np.concatenate(
            [_window_statistics(window)[statistic] for statistic in feature_config.statistics]
        )
        for window in windows
    ]
    return np.concatenate(feature_parts).astype(np.float64, copy=False)


def _fit_estimator(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_features: str | float | None,
    max_samples: float | None,
    criterion: str,
    config: RandomForestConfig,
    sample_weight: np.ndarray | None = None,
) -> RandomForestClassifier:
    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_samples=max_samples,
        criterion=criterion,
        class_weight=config.class_weight,
        random_state=config.random_state,
        n_jobs=1,
    )
    estimator.fit(train_features, train_labels, sample_weight=sample_weight)
    return estimator


def _participant_weights(
    records: list[tuple[str, str, Recording]],
) -> tuple[np.ndarray, dict[str, dict[str, float | int]]]:
    counts: dict[str, int] = {}
    for participant, _, _ in records:
        counts[participant] = counts.get(participant, 0) + 1
    scale = len(records) / len(counts)
    weights = np.asarray(
        [scale / counts[participant] for participant, _, _ in records],
        dtype=np.float64,
    )
    metadata = {
        participant: {
            "recordings": count,
            "weight_per_recording": scale / count,
            "total_weight": scale,
        }
        for participant, count in sorted(counts.items())
    }
    return weights, metadata


def _grouped_fold_indices(
    records: list[tuple[str, str, Recording]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    participants = np.asarray([participant for participant, _, _ in records], dtype=object)
    if len(set(participants.tolist())) < 2:
        raise DatasetFormatError("Grouped validation requires at least two participants")
    splitter = LeaveOneGroupOut()
    placeholder = np.zeros((len(records), 1), dtype=np.float64)
    return [
        (train_indices, validation_indices)
        for train_indices, validation_indices in splitter.split(
            placeholder, groups=participants
        )
    ]


def _selection_protocol(
    records: list[tuple[str, str, Recording]],
    test_participants: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    fold_metadata = []
    for index, (train_indices, validation_indices) in enumerate(folds):
        train_records = [records[position] for position in train_indices]
        validation_records = [records[position] for position in validation_indices]
        _, participant_weights = _participant_weights(train_records)
        fold_metadata.append(
            {
                "index": index,
                "train_participants": sorted({record[0] for record in train_records}),
                "validation_participants": sorted(
                    {record[0] for record in validation_records}
                ),
                "train_recordings": len(train_records),
                "validation_recordings": len(validation_records),
                "participant_weights": participant_weights,
            }
        )
    _, final_weights = _participant_weights(records)
    return {
        "method": "leave_one_participant_out",
        "selection_splits": ["train", "validation"],
        "participants": sorted({participant for participant, _, _ in records}),
        "test_participants": sorted(test_participants),
        "participant_weight_policy": "equal_participant_contribution",
        "final_fit_participant_weights": final_weights,
        "folds": fold_metadata,
    }


def _threshold_candidates(scores: np.ndarray) -> np.ndarray:
    values = np.unique(np.asarray(scores, dtype=float))
    if values.size == 1:
        margin = max(abs(float(values[0])) * 1e-6, 1e-6)
        return np.asarray([values[0] - margin, values[0], values[0] + margin])
    midpoints = (values[:-1] + values[1:]) / 2
    margin = max(abs(float(values[0])) * 1e-6, 1e-6)
    return np.concatenate(([values[0] - margin], midpoints, [values[-1] + margin]))


def _threshold_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = np.where(scores >= threshold, "fall", "non-fall")
    matrix = confusion_matrix(labels, predictions, labels=LABELS)
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    negative_total = true_negative + false_positive
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, pos_label="fall", zero_division=0)),
        "sensitivity": float(
            recall_score(labels, predictions, pos_label="fall", zero_division=0)
        ),
        "specificity": float(true_negative / negative_total) if negative_total else 0.0,
        "precision": float(
            precision_score(labels, predictions, pos_label="fall", zero_division=0)
        ),
        "false_positive_rate": float(false_positive / negative_total)
        if negative_total
        else 0.0,
        "support": float(true_negative + false_positive + false_negative + true_positive),
    }


def select_random_forest_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    policy: str,
    minimum_recall: float = 0.9,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Select threshold using an explicit grouped-validation objective."""

    if not 0.0 <= minimum_recall <= 1.0:
        raise ValueError("minimum_recall must be between 0 and 1")
    candidates = [
        _threshold_metrics(labels, scores, threshold)
        for threshold in _threshold_candidates(scores)
    ]
    if policy in {"maximize_f1_then_sensitivity", "maximize_grouped_validation_f1_then_sensitivity"}:
        selected = max(
            candidates,
            key=lambda result: (
                result["f1"],
                result["sensitivity"],
                -result["false_positive_rate"],
            ),
        )
    elif policy in {
        "maximize_balanced_accuracy_then_sensitivity",
        "maximize_grouped_validation_balanced_accuracy_then_sensitivity",
    }:
        selected = max(
            candidates,
            key=lambda result: (
                result["balanced_accuracy"],
                result["sensitivity"],
                -result["false_positive_rate"],
            ),
        )
    elif policy in {
        "maximize_recall_constrained_accuracy",
        "maximize_grouped_validation_recall_constrained_accuracy",
    }:
        eligible = [
            result for result in candidates if result["sensitivity"] >= minimum_recall
        ]
        selected = max(
            eligible or candidates,
            key=lambda result: (
                result["accuracy"],
                result["sensitivity"],
                -result["false_positive_rate"],
            ),
        )
    else:
        raise ValueError(f"Unsupported threshold policy: {policy!r}")
    return selected, candidates


def _candidate_selection_key(
    candidate: dict[str, Any], policy: str
) -> tuple[float, float, float]:
    if "balanced_accuracy" in policy:
        primary = candidate["threshold_metrics"]["balanced_accuracy"]
    elif "recall_constrained" in policy:
        primary = candidate["threshold_metrics"]["accuracy"]
    else:
        primary = candidate["threshold_metrics"]["f1"]
    return (
        primary,
        candidate["threshold_metrics"]["sensitivity"],
        -candidate["threshold_metrics"]["false_positive_rate"],
    )


def _fit_score_calibrator(
    scores: np.ndarray,
    labels: np.ndarray,
    method: str,
    random_state: int,
) -> ScoreCalibrator | None:
    if method == "none":
        return None
    binary_labels = (labels == "fall").astype(np.int8)
    if method == "sigmoid":
        estimator: LogisticRegression | IsotonicRegression = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            random_state=random_state,
        )
        estimator.fit(scores.reshape(-1, 1), binary_labels)
    elif method == "isotonic":
        estimator = IsotonicRegression(out_of_bounds="clip")
        estimator.fit(scores, binary_labels)
    else:
        raise ValueError(f"Unsupported score calibration method: {method!r}")
    return ScoreCalibrator(method, estimator)


def _participant_results(
    split: str,
    records: list[tuple[str, str, Recording]],
    features: np.ndarray,
    model: ThresholdedRandomForest,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for index, (participant, _, _) in enumerate(records):
        groups.setdefault(participant, []).append(index)

    results = {}
    for participant, indexes in sorted(groups.items()):
        result = evaluate_classifier(
            model,
            features[indexes],
            np.asarray([records[index][2].target for index in indexes], dtype=object),
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
            group_ids=[participant] * len(indexes),
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


def train_random_forest(
    dataset_root: str | Path,
    split_manifest: str | Path,
    output_dir: str | Path,
    config: RandomForestConfig | None = None,
    *,
    processed_dataset: str | Path | None = None,
) -> RandomForestRun:
    """Train, threshold, evaluate, and persist a separate Random Forest run."""

    config = config or RandomForestConfig()
    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_split_manifest(split_manifest)
    split_records = load_split_records(
        root, manifest, processed_dataset=processed_dataset
    )
    validate_participant_splits(split_records)
    augmentation_result = augment_split_records(split_records, config.augmentation)
    fit_split_records = {
        split: list(records) for split, records in augmentation_result.splits.items()
    }

    audit = audit_wifall(root)
    audit_path = output / "dataset_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    selection_records = fit_split_records["train"] + fit_split_records["validation"]
    if not selection_records:
        raise DatasetFormatError("Random Forest requires non-test selection recordings")
    feature_configs = config.feature_configs()
    feature_names = [feature_config.name for feature_config in feature_configs]
    if len(set(feature_names)) != len(feature_names):
        raise DatasetFormatError("Random Forest feature variant names must be unique")
    feature_sets = {
        feature_config.name: np.vstack(
            [
                extract_random_forest_features(record, feature_config)
                for _, _, record in selection_records
            ]
        )
        for feature_config in feature_configs
    }
    selection_labels = np.asarray(
        [record.target for _, _, record in selection_records], dtype=object
    )
    if set(selection_labels) != set(LABELS):
        raise DatasetFormatError("Selection participants must contain fall and non-fall recordings")

    folds = _grouped_fold_indices(selection_records)
    selection_protocol = _selection_protocol(
        selection_records,
        manifest["participants"]["test"],
        folds,
    )
    candidates: list[dict[str, Any]] = []
    calibrators: list[ScoreCalibrator | None] = []
    for feature_config in feature_configs:
        selection_features = feature_sets[feature_config.name]
        for n_estimators in config.n_estimators:
            for max_depth in config.max_depth:
                for min_samples_leaf in config.min_samples_leaf:
                    for max_features in config.max_features:
                        for max_samples in config.max_samples:
                            for criterion in config.criterion:
                                out_of_fold_scores = np.empty(
                                    selection_labels.shape[0], dtype=np.float64
                                )
                                for train_indices, validation_indices in folds:
                                    fold_records = [
                                        selection_records[index] for index in train_indices
                                    ]
                                    fold_weights, _ = _participant_weights(fold_records)
                                    estimator = _fit_estimator(
                                        selection_features[train_indices],
                                        selection_labels[train_indices],
                                        n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_samples_leaf=min_samples_leaf,
                                        max_features=max_features,
                                        max_samples=max_samples,
                                        criterion=criterion,
                                        config=config,
                                        sample_weight=fold_weights,
                                    )
                                    out_of_fold_scores[validation_indices] = ThresholdedRandomForest(
                                        estimator, 0.5
                                    ).decision_function(selection_features[validation_indices])
                                calibrator = _fit_score_calibrator(
                                    out_of_fold_scores,
                                    selection_labels,
                                    config.calibration_method,
                                    config.random_state,
                                )
                                calibrated_scores = (
                                    calibrator.transform(out_of_fold_scores)
                                    if calibrator
                                    else out_of_fold_scores
                                )
                                threshold, threshold_table = select_random_forest_threshold(
                                    selection_labels,
                                    calibrated_scores,
                                    config.threshold_policy,
                                    config.minimum_recall,
                                )
                                selection_roc_auc = float(
                                    roc_auc_score(
                                        selection_labels == "fall", calibrated_scores
                                    )
                                )
                                selection_pr_auc = float(
                                    average_precision_score(
                                        selection_labels == "fall", calibrated_scores
                                    )
                                )
                                candidates.append(
                                    {
                                        "feature": feature_config.as_dict(),
                                        "feature_name": feature_config.name,
                                        "n_estimators": n_estimators,
                                        "max_depth": max_depth,
                                        "min_samples_leaf": min_samples_leaf,
                                        "max_features": max_features,
                                        "max_samples": max_samples,
                                        "criterion": criterion,
                                        "calibration_method": config.calibration_method,
                                        "threshold": threshold["threshold"],
                                        "threshold_metrics": threshold,
                                        "threshold_table": threshold_table,
                                        "validation_f1": threshold["f1"],
                                        "validation_sensitivity": threshold["sensitivity"],
                                        "selection_metric": (
                                            "balanced_accuracy"
                                            if "balanced_accuracy" in config.threshold_policy
                                            else "accuracy"
                                            if "recall_constrained" in config.threshold_policy
                                            else "f1"
                                        ),
                                        "selection_metrics": {
                                            "split": "grouped_validation",
                                            "accuracy": threshold["accuracy"],
                                            "balanced_accuracy": threshold["balanced_accuracy"],
                                            "precision": threshold["precision"],
                                            "recall": threshold["sensitivity"],
                                            "f1": threshold["f1"],
                                            "sensitivity": threshold["sensitivity"],
                                            "specificity": threshold["specificity"],
                                            "false_positive_rate": threshold["false_positive_rate"],
                                            "roc_auc": selection_roc_auc,
                                            "pr_auc": selection_pr_auc,
                                            "minimum_recall": config.minimum_recall,
                                        },
                                    }
                                )
                                calibrators.append(calibrator)

    if not candidates:
        raise DatasetFormatError("Random Forest configuration produced no candidates")
    selected_index = max(
        range(len(candidates)),
        key=lambda index: _candidate_selection_key(candidates[index], config.threshold_policy),
    )
    selected = candidates[selected_index]
    selected_calibrator = calibrators[selected_index]
    selected_feature_config = next(
        feature_config
        for feature_config in feature_configs
        if feature_config.name == selected["feature_name"]
    )
    selected_features = feature_sets[selected_feature_config.name]
    final_weights, _ = _participant_weights(selection_records)
    final_estimator = _fit_estimator(
        selected_features,
        selection_labels,
        n_estimators=selected["n_estimators"],
        max_depth=selected["max_depth"],
        min_samples_leaf=selected["min_samples_leaf"],
        max_features=selected["max_features"],
        max_samples=selected["max_samples"],
        criterion=selected["criterion"],
        config=config,
        sample_weight=final_weights,
    )
    thresholded = ThresholdedRandomForest(
        final_estimator,
        selected["threshold"],
        calibrator=selected_calibrator,
    )
    feature_pipeline = WiFallRandomForestPipeline(thresholded, selected_feature_config)
    model_path = output / "model.joblib"
    joblib.dump(feature_pipeline, model_path)

    model_metadata = {
        "family": "random_forest",
        "name": "thresholded_random_forest",
        "training_code": "wifall.random_forest",
        "artifact": model_path.name,
        "parameters": {
            "n_estimators": selected["n_estimators"],
            "max_depth": selected["max_depth"],
            "min_samples_leaf": selected["min_samples_leaf"],
            "max_features": selected["max_features"],
            "max_samples": selected["max_samples"],
            "criterion": selected["criterion"],
            "class_weight": config.class_weight,
            "random_state": config.random_state,
            "n_jobs": 1,
        },
        "decision_threshold": selected["threshold"],
        "threshold_policy": config.threshold_policy,
        "minimum_recall": config.minimum_recall,
        "calibration": {
            "method": config.calibration_method,
            "fit_split": "grouped_validation_oof",
            "positive_label": "fall",
        },
        "fit_splits": ["train", "validation"],
        "fit_participants": selection_protocol["participants"],
        "score_orientation": {
            "positive_label": "fall",
            "score": "fall_probability",
            "source": "RandomForestClassifier.predict_proba",
        },
    }
    feature_metadata = selected_feature_config.as_dict() | {
        "count": int(selected_features.shape[1])
    }
    preprocessing_metadata = {
        "steps": [
            "per-recording CSI amplitude median/MAD normalization",
            f"{selected_feature_config.window_count} temporal windows",
            "RandomForestClassifier with participant-balanced sample weights",
            f"{config.calibration_method} score calibration",
        ],
        "fit_split": "train+validation",
        "augmentation": augmentation_result.metadata,
    }
    split_metadata = {
        "strategy": "participant-held-out",
        "config": manifest["split_config"],
        "participants": manifest["participants"],
        "counts": manifest["counts"],
        "selection_participants": selection_protocol["participants"],
        "test_participants": selection_protocol["test_participants"],
    }
    metadata = {
        "run_id": config.run_id,
        "dataset": manifest["dataset"],
        "task": {"name": "binary_fall_detection", "signal": "WiFi CSI"},
        "model": model_metadata,
        "features": feature_metadata,
        "preprocessing": preprocessing_metadata,
        "split": split_metadata,
        "augmentation": augmentation_result.metadata,
        "data_source": {
            "type": "duration_normalized_dataset"
            if processed_dataset is not None
            else "raw_recordings",
            "path": str(Path(processed_dataset)) if processed_dataset is not None else None,
        },
    }

    evaluations: dict[str, dict[str, Any]] = {}
    evaluation_artifacts: dict[str, dict[str, Any]] = {}
    for split, records in split_records.items():
        if not records:
            continue
        features = np.vstack(
            [
                extract_random_forest_features(record, selected_feature_config)
                for _, _, record in records
            ]
        )
        labels = np.asarray([record.target for _, _, record in records], dtype=object)
        evaluation_result = evaluate_classifier(
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
        evaluation = evaluation_result["evaluations"][split]
        evaluation["fit_status"] = "fit_data" if split in {"train", "validation"} else "held_out"
        evaluation["participant_metrics"] = _participant_results(
            split, records, features, thresholded, metadata
        )
        scores = np.asarray(evaluation["scores"], dtype=float)
        evaluation["score_distribution"] = _score_distribution(scores, records)
        evaluation["threshold_metrics"] = selected["threshold_table"] if split == "validation" else []
        # Diagnostics consume finalized predictions only. They never participate
        # in feature, estimator, calibration, or threshold selection above.
        evaluation["hard_example_analysis"] = analyze_hard_examples(
            evaluation,
            recording_observations(records),
            run_id=config.run_id,
            split_name=split,
            threshold=selected["threshold"],
        )
        evaluations[split] = evaluation
        evaluation_artifacts[split] = evaluation_result

    primary_split = "test" if "test" in evaluations else "validation"
    primary = evaluations[primary_split]
    feature_comparison = []
    for feature_config in feature_configs:
        variant_candidates = [
            candidate
            for candidate in candidates
            if candidate["feature_name"] == feature_config.name
        ]
        best_variant = max(
            variant_candidates,
            key=lambda candidate: _candidate_selection_key(
                candidate, config.threshold_policy
            ),
        )
        feature_comparison.append(
            {
                "name": feature_config.name,
                "feature": feature_config.as_dict(),
                "candidate_count": len(variant_candidates),
                "best_selection_metrics": best_variant["selection_metrics"],
                "selected": best_variant is selected,
            }
        )
    artifact = evaluation_artifacts[primary_split] | {
        "evaluation_split": primary_split,
        "augmentation": augmentation_result.metadata,
        "data_source": metadata["data_source"],
        "evaluations": evaluations,
        "metrics": primary["metrics"],
        "confusion_matrix": primary["confusion_matrix"],
        "data_audit": audit,
        "selection": {
            "candidates": candidates,
            "selected": selected,
            "feature_comparison": feature_comparison,
            "selection_split": "grouped_validation",
            "protocol": selection_protocol,
        },
    }
    artifact_path = output / "run.json"
    save_evaluation_result(artifact, artifact_path)
    report_path = output / "report.html"
    report_path.write_text(render_html_report(artifact), encoding="utf-8")
    return RandomForestRun(
        artifact_path=artifact_path,
        report_path=report_path,
        model_path=model_path,
        audit_path=audit_path,
        artifact=artifact,
    )


def load_random_forest(path: str | Path) -> WiFallRandomForestPipeline:
    """Load a persisted Random Forest feature and prediction pipeline."""

    pipeline = joblib.load(Path(path).expanduser())
    if not isinstance(pipeline, WiFallRandomForestPipeline):
        raise TypeError("Model artifact is not a WiFallRandomForestPipeline")
    return pipeline
