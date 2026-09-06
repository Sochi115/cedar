"""Separate Random Forest training pipeline for WiFall fall detection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut

from evaluation import evaluate_classifier, render_html_report, save_evaluation_result

from .audit import audit_wifall
from .improved_svm import (
    ImprovedFeatureConfig,
    extract_improved_features,
    select_threshold,
)
from .loader import DatasetFormatError, Recording, load_split_manifest
from .training import load_split_records, validate_participant_splits

LABELS = ("non-fall", "fall")


@dataclass(frozen=True)
class RandomForestConfig:
    """Configuration for Random Forest selection and thresholding."""

    run_id: str = "wifall-random-forest"
    n_estimators: tuple[int, ...] = (100,)
    max_depth: tuple[int | None, ...] = (None, 20)
    min_samples_leaf: tuple[int, ...] = (1,)
    max_features: tuple[str | float | None, ...] = ("sqrt",)
    class_weight: str = "balanced"
    random_state: int = 42
    feature: ImprovedFeatureConfig = field(default_factory=ImprovedFeatureConfig)
    threshold_policy: str = "maximize_grouped_validation_f1_then_sensitivity"

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "n_estimators": list(self.n_estimators),
            "max_depth": list(self.max_depth),
            "min_samples_leaf": list(self.min_samples_leaf),
            "max_features": list(self.max_features),
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "feature": self.feature.as_dict(),
            "threshold_policy": self.threshold_policy,
        }


@dataclass(frozen=True)
class RandomForestRun:
    """Paths and result artifact produced by Random Forest training."""

    artifact_path: Path
    report_path: Path
    model_path: Path
    audit_path: Path
    artifact: dict[str, Any]


@dataclass
class ThresholdedRandomForest:
    """Persistable Random Forest wrapper with fall-probability scores."""

    estimator: RandomForestClassifier
    threshold: float
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
        return probabilities[:, positive_index]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return probabilities in the stable ``non-fall``, ``fall`` order."""

        probabilities = np.asarray(self.estimator.predict_proba(features), dtype=float)
        estimator_classes = [str(value) for value in self.estimator.classes_]
        try:
            indexes = [estimator_classes.index(label) for label in LABELS]
        except ValueError as error:
            raise DatasetFormatError("Random Forest classes do not match WiFall labels") from error
        return probabilities[:, indexes]

    def predict(self, features: np.ndarray) -> np.ndarray:
        scores = self.decision_function(features)
        return np.where(scores >= self.threshold, self.positive_label, self.negative_label)


@dataclass
class WiFallRandomForestPipeline:
    """Persistable improved-feature extraction and thresholded Random Forest."""

    estimator: ThresholdedRandomForest
    feature_config: ImprovedFeatureConfig

    def features(self, recordings: list[Recording]) -> np.ndarray:
        return np.vstack(
            [extract_random_forest_features(recording, self.feature_config) for recording in recordings]
        )

    def predict_recordings(self, recordings: list[Recording]) -> np.ndarray:
        return self.estimator.predict(self.features(recordings))

    def decision_function_recordings(self, recordings: list[Recording]) -> np.ndarray:
        return self.estimator.decision_function(self.features(recordings))


def extract_random_forest_features(
    recording: Recording, config: ImprovedFeatureConfig | None = None
) -> np.ndarray:
    """Extract exactly the improved SVM feature representation."""

    return extract_improved_features(recording, config)


def _fit_estimator(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_features: str | float | None,
    config: RandomForestConfig,
    sample_weight: np.ndarray | None = None,
) -> RandomForestClassifier:
    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
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
    weights = np.asarray(
        [1.0 / counts[participant] for participant, _, _ in records],
        dtype=np.float64,
    )
    metadata = {
        participant: {
            "recordings": count,
            "weight_per_recording": 1.0 / count,
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
) -> RandomForestRun:
    """Train, threshold, evaluate, and persist a separate Random Forest run."""

    config = config or RandomForestConfig()
    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_split_manifest(split_manifest)
    split_records = load_split_records(root, manifest)
    validate_participant_splits(split_records)

    audit = audit_wifall(root)
    audit_path = output / "dataset_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    selection_records = split_records["train"] + split_records["validation"]
    if not selection_records:
        raise DatasetFormatError("Random Forest requires non-test selection recordings")
    selection_features = np.vstack(
        [
            extract_random_forest_features(record, config.feature)
            for _, _, record in selection_records
        ]
    )
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
    for n_estimators in config.n_estimators:
        for max_depth in config.max_depth:
            for min_samples_leaf in config.min_samples_leaf:
                for max_features in config.max_features:
                    out_of_fold_scores = np.empty(selection_labels.shape[0], dtype=np.float64)
                    for train_indices, validation_indices in folds:
                        fold_records = [selection_records[index] for index in train_indices]
                        fold_weights, _ = _participant_weights(fold_records)
                        estimator = _fit_estimator(
                            selection_features[train_indices],
                            selection_labels[train_indices],
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            max_features=max_features,
                            config=config,
                            sample_weight=fold_weights,
                        )
                        out_of_fold_scores[validation_indices] = ThresholdedRandomForest(
                            estimator, 0.5
                        ).decision_function(selection_features[validation_indices])
                    threshold, threshold_table = select_threshold(
                        selection_labels, out_of_fold_scores
                    )
                    candidates.append(
                        {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "min_samples_leaf": min_samples_leaf,
                            "max_features": max_features,
                            "threshold": threshold.threshold,
                            "threshold_metrics": threshold.as_dict(),
                            "threshold_table": threshold_table,
                            "validation_f1": threshold.f1,
                            "validation_sensitivity": threshold.sensitivity,
                            "selection_metrics": {
                                "split": "grouped_validation",
                                "f1": threshold.f1,
                                "sensitivity": threshold.sensitivity,
                                "false_positive_rate": threshold.false_positive_rate,
                            },
                        }
                    )

    if not candidates:
        raise DatasetFormatError("Random Forest configuration produced no candidates")
    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["validation_f1"],
            candidate["validation_sensitivity"],
            -candidate["threshold_metrics"]["false_positive_rate"],
        ),
    )
    final_weights, _ = _participant_weights(selection_records)
    final_estimator = _fit_estimator(
        selection_features,
        selection_labels,
        n_estimators=selected["n_estimators"],
        max_depth=selected["max_depth"],
        min_samples_leaf=selected["min_samples_leaf"],
        max_features=selected["max_features"],
        config=config,
        sample_weight=final_weights,
    )
    thresholded = ThresholdedRandomForest(final_estimator, selected["threshold"])
    feature_pipeline = WiFallRandomForestPipeline(thresholded, config.feature)
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
            "class_weight": config.class_weight,
            "random_state": config.random_state,
            "n_jobs": 1,
        },
        "decision_threshold": selected["threshold"],
        "threshold_policy": config.threshold_policy,
        "fit_splits": ["train", "validation"],
        "fit_participants": selection_protocol["participants"],
        "score_orientation": {
            "positive_label": "fall",
            "score": "fall_probability",
            "source": "RandomForestClassifier.predict_proba",
        },
    }
    feature_metadata = config.feature.as_dict() | {"count": int(selection_features.shape[1])}
    preprocessing_metadata = {
        "steps": [
            "per-recording CSI amplitude median/MAD normalization",
            "four temporal windows",
            "RandomForestClassifier",
        ],
        "fit_split": "train+validation",
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
    }

    evaluations: dict[str, dict[str, Any]] = {}
    evaluation_artifacts: dict[str, dict[str, Any]] = {}
    for split, records in split_records.items():
        if not records:
            continue
        features = np.vstack(
            [extract_random_forest_features(record, config.feature) for _, _, record in records]
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
        evaluations[split] = evaluation
        evaluation_artifacts[split] = evaluation_result

    primary_split = "test" if "test" in evaluations else "validation"
    primary = evaluations[primary_split]
    artifact = evaluation_artifacts[primary_split] | {
        "evaluation_split": primary_split,
        "evaluations": evaluations,
        "metrics": primary["metrics"],
        "confusion_matrix": primary["confusion_matrix"],
        "data_audit": audit,
        "selection": {
            "candidates": candidates,
            "selected": selected,
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
