"""Small participant-held-out MLP pipeline for WiFall fall detection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from evaluation import (
    analyze_hard_examples,
    evaluate_classifier,
    render_html_report,
    save_evaluation_result,
)

from .audit import audit_wifall
from .diagnostics import recording_observations
from .improved_svm import ImprovedFeatureConfig
from .loader import DatasetFormatError, Recording, load_split_manifest
from .random_forest import RandomForestFeatureConfig, extract_random_forest_features
from .training import load_split_records, validate_participant_splits

LABELS = ("non-fall", "fall")


@dataclass(frozen=True)
class MlpFeatureConfig:
    """Fixed-size normalized CSI temporal features used by the MLP."""

    name: str = "csi_normalized_temporal_mlp_v1"
    signal: str = "normalized CSI amplitude"
    window_count: int = 4
    statistics: tuple[str, ...] = ("mean", "std", "min", "max", "delta_std")

    def __post_init__(self) -> None:
        if self.window_count < 1:
            raise ValueError("window_count must be at least 1")
        allowed = {"mean", "std", "min", "max", "delta_std"}
        unsupported = set(self.statistics).difference(allowed)
        if unsupported:
            raise ValueError(f"Unsupported MLP feature statistics: {sorted(unsupported)}")
        if len(set(self.statistics)) != len(self.statistics):
            raise ValueError("statistics must not contain duplicates")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "signal": self.signal,
            "window_count": self.window_count,
            "statistics": list(self.statistics),
        }

    def random_forest_config(self) -> RandomForestFeatureConfig:
        return RandomForestFeatureConfig(
            name=self.name,
            signal=self.signal,
            window_count=self.window_count,
            statistics=self.statistics,
        )


@dataclass(frozen=True)
class MlpConfig:
    """Configuration for MLP search and safety-oriented threshold selection."""

    run_id: str = "wifall-mlp"
    hidden_layer_sizes: tuple[tuple[int, ...], ...] = ((32, 8),)
    alpha_values: tuple[float, ...] = (1e-3,)
    activation: str = "relu"
    solver: str = "lbfgs"
    learning_rate_init: float = 1e-3
    max_iter: int = 300
    class_balance: str = "oversample"
    minimum_recall: float = 0.95
    feature: MlpFeatureConfig = field(default_factory=MlpFeatureConfig)
    random_state: int = 42

    def __post_init__(self) -> None:
        if not self.hidden_layer_sizes:
            raise ValueError("hidden_layer_sizes cannot be empty")
        if any(not architecture or any(width < 1 for width in architecture) for architecture in self.hidden_layer_sizes):
            raise ValueError("MLP layer sizes must contain positive widths")
        if not self.alpha_values or any(alpha < 0.0 for alpha in self.alpha_values):
            raise ValueError("alpha_values must contain non-negative values")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if self.class_balance not in {"none", "oversample"}:
            raise ValueError("class_balance must be 'none' or 'oversample'")
        if not 0.0 <= self.minimum_recall <= 1.0:
            raise ValueError("minimum_recall must be between 0 and 1")

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "hidden_layer_sizes": [list(architecture) for architecture in self.hidden_layer_sizes],
            "alpha_values": list(self.alpha_values),
            "activation": self.activation,
            "solver": self.solver,
            "learning_rate_init": self.learning_rate_init,
            "max_iter": self.max_iter,
            "class_balance": self.class_balance,
            "minimum_recall": self.minimum_recall,
            "feature": self.feature.as_dict(),
            "random_state": self.random_state,
        }


@dataclass(frozen=True)
class MlpRun:
    """Paths and result artifact produced by MLP training."""

    artifact_path: Path
    report_path: Path
    model_path: Path
    audit_path: Path
    artifact: dict[str, Any]


@dataclass
class WiFallMlpPipeline:
    """Persistable MLP feature extraction, scaling, scoring, and thresholding."""

    scaler: StandardScaler
    estimator: MLPClassifier
    feature_config: MlpFeatureConfig
    threshold: float
    positive_label: str = "fall"
    negative_label: str = "non-fall"

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray((self.negative_label, self.positive_label))

    def features(self, recordings: list[Recording]) -> np.ndarray:
        return np.vstack([extract_mlp_features(recording, self.feature_config) for recording in recordings])

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(features)
        probabilities = np.asarray(self.estimator.predict_proba(scaled), dtype=float)
        classes = [str(value) for value in self.estimator.classes_]
        try:
            positive_index = classes.index(self.positive_label)
        except ValueError as error:
            raise DatasetFormatError("MLP has no fall class") from error
        return probabilities[:, positive_index]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        scores = self.decision_function(features)
        return np.column_stack((1.0 - scores, scores))

    def predict(self, features: np.ndarray) -> np.ndarray:
        scores = self.decision_function(features)
        return np.where(scores >= self.threshold, self.positive_label, self.negative_label)

    def predict_recordings(self, recordings: list[Recording]) -> np.ndarray:
        return self.predict(self.features(recordings))

    def decision_function_recordings(self, recordings: list[Recording]) -> np.ndarray:
        return self.decision_function(self.features(recordings))


def extract_mlp_features(
    recording: Recording,
    config: MlpFeatureConfig | ImprovedFeatureConfig | None = None,
) -> np.ndarray:
    """Extract one deterministic, fixed-size normalized temporal feature vector."""

    if config is None:
        mlp_config = MlpFeatureConfig()
    elif isinstance(config, MlpFeatureConfig):
        mlp_config = config
    else:
        mlp_config = MlpFeatureConfig(
            name=config.name,
            signal=config.signal,
            window_count=config.window_count,
            statistics=config.statistics,
        )
    return extract_random_forest_features(recording, mlp_config.random_forest_config())


def _balanced_rows(
    features: np.ndarray,
    labels: np.ndarray,
    random_state: int,
    method: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    if method == "none":
        return features, labels, 0
    rng = np.random.default_rng(random_state)
    class_indices = [np.flatnonzero(labels == label) for label in LABELS]
    if any(indexes.size == 0 for indexes in class_indices):
        raise DatasetFormatError("MLP training requires both fall and non-fall classes")
    target_count = max(indexes.size for indexes in class_indices)
    selected_indices = np.concatenate(
        [rng.choice(indexes, size=target_count, replace=True) for indexes in class_indices]
    )
    return features[selected_indices], labels[selected_indices], int(selected_indices.size - labels.size)


def _fall_scores(estimator: MLPClassifier, scaled_features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(scaled_features), dtype=float)
    classes = [str(value) for value in estimator.classes_]
    try:
        index = classes.index("fall")
    except ValueError as error:
        raise DatasetFormatError("MLP estimator has no fall class") from error
    return probabilities[:, index]


def _threshold_candidates(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(scores)
    if unique.size == 0:
        raise DatasetFormatError("Cannot select an MLP threshold from empty scores")
    candidates = {0.0, 0.5, 1.0, *unique.tolist()}
    candidates.update(float((left + right) / 2.0) for left, right in pairwise(unique))
    return np.asarray(sorted(candidates), dtype=float)


def _specificity(labels: np.ndarray, predictions: np.ndarray) -> float:
    negative = labels == "non-fall"
    if not negative.any():
        return 0.0
    return float(np.mean(predictions[negative] == "non-fall"))


def _threshold_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: list[str],
    threshold: float,
) -> dict[str, float | bool]:
    predictions = np.where(scores >= threshold, "fall", "non-fall")
    group_recalls: list[float] = []
    group_specificities: list[float] = []
    for group in sorted(set(groups)):
        indexes = np.asarray([index for index, value in enumerate(groups) if value == group])
        group_labels = labels[indexes]
        group_predictions = predictions[indexes]
        falls = group_labels == "fall"
        group_recalls.append(float(np.mean(group_predictions[falls] == "fall")) if falls.any() else 0.0)
        group_specificities.append(_specificity(group_labels, group_predictions))
    minimum_group_recall = min(group_recalls) if group_recalls else 0.0
    true_negative = np.sum((labels == "non-fall") & (predictions == "non-fall"))
    false_positive = np.sum((labels == "non-fall") & (predictions == "fall"))
    recall = float(np.mean(predictions[labels == "fall"] == "fall")) if np.any(labels == "fall") else 0.0
    precision = float(
        np.sum((labels == "fall") & (predictions == "fall"))
        / max(np.sum(predictions == "fall"), 1)
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(np.mean(labels == predictions)),
        "balanced_accuracy": float(
            (np.mean(predictions[labels == "fall"] == "fall") if np.any(labels == "fall") else 0.0)
            + _specificity(labels, predictions)
        )
        / 2.0,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "f1": f1,
        "specificity": _specificity(labels, predictions),
        "false_positive_rate": float(false_positive / max(true_negative + false_positive, 1)),
        "minimum_group_recall": float(minimum_group_recall),
        "group_specificity": float(np.mean(group_specificities)) if group_specificities else 0.0,
        "eligible": bool(minimum_group_recall >= 0.0),
    }


def select_mlp_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: list[str],
    minimum_recall: float = 0.95,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select threshold requiring minimum recall for every validation group."""

    if labels.size != scores.size or labels.size != len(groups):
        raise ValueError("MLP labels, scores, and groups must have equal lengths")
    if not 0.0 <= minimum_recall <= 1.0:
        raise ValueError("minimum_recall must be between 0 and 1")
    candidates = [
        _threshold_metrics(labels, scores, groups, threshold)
        for threshold in _threshold_candidates(scores)
    ]
    for candidate in candidates:
        candidate["eligible"] = bool(candidate["minimum_group_recall"] >= minimum_recall)
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = max(
        eligible or candidates,
        key=lambda candidate: (
            candidate["eligible"],
            candidate["minimum_group_recall"],
            candidate["group_specificity"],
            candidate["f1"],
            -candidate["threshold"],
        ),
    )
    return selected, candidates


def _selection_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    selected = candidate["threshold_metrics"]
    return (
        float(selected["eligible"]),
        float(selected["minimum_group_recall"]),
        float(selected["group_specificity"]),
        float(selected["specificity"]),
        -float(selected["false_positive_rate"]),
    )


def _fit_candidate(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    config: MlpConfig,
    architecture: tuple[int, ...],
    alpha: float,
) -> tuple[StandardScaler, MLPClassifier, int]:
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_features)
    balanced_features, balanced_labels, generated_count = _balanced_rows(
        scaled_train,
        train_labels,
        config.random_state,
        config.class_balance,
    )
    estimator = MLPClassifier(
        hidden_layer_sizes=architecture,
        activation=config.activation,
        solver=config.solver,
        alpha=alpha,
        learning_rate_init=config.learning_rate_init,
        max_iter=config.max_iter,
        random_state=config.random_state,
    )
    estimator.fit(balanced_features, balanced_labels)
    return scaler, estimator, generated_count


def _grouped_oof_scores(
    features: np.ndarray,
    labels: np.ndarray,
    groups: list[str],
    config: MlpConfig,
    architecture: tuple[int, ...],
    alpha: float,
) -> tuple[np.ndarray, int]:
    """Fit one model per held-out non-test participant and return OOF scores."""

    scores = np.empty(labels.size, dtype=float)
    generated_count = 0
    for held_out_group in sorted(set(groups)):
        validation_indexes = np.asarray(
            [index for index, group in enumerate(groups) if group == held_out_group]
        )
        train_indexes = np.asarray(
            [index for index, group in enumerate(groups) if group != held_out_group]
        )
        scaler, estimator, fold_generated = _fit_candidate(
            features[train_indexes],
            labels[train_indexes],
            config,
            architecture,
            alpha,
        )
        scores[validation_indexes] = _fall_scores(
            estimator,
            scaler.transform(features[validation_indexes]),
        )
        generated_count += fold_generated
    return scores, generated_count


def train_mlp(
    dataset_root: str | Path,
    split_manifest: str | Path,
    output_dir: str | Path,
    config: MlpConfig | None = None,
) -> MlpRun:
    """Train, select, evaluate, and persist a small MLP pipeline."""

    config = config or MlpConfig()
    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_split_manifest(split_manifest)
    split_records = load_split_records(root, manifest)
    validate_participant_splits(split_records)
    train_records = split_records["train"]
    validation_records = split_records["validation"]
    if not train_records or not validation_records:
        raise DatasetFormatError("MLP requires train and validation recordings")

    audit = audit_wifall(root)
    audit_path = output / "dataset_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    train_features = np.vstack([extract_mlp_features(record, config.feature) for _, _, record in train_records])
    train_labels = np.asarray([record.target for _, _, record in train_records], dtype=object)
    selection_records = train_records + validation_records
    selection_features = np.vstack(
        [extract_mlp_features(record, config.feature) for _, _, record in selection_records]
    )
    selection_labels = np.asarray(
        [record.target for _, _, record in selection_records], dtype=object
    )
    selection_groups = [participant for participant, _, _ in selection_records]

    candidates: list[dict[str, Any]] = []
    for architecture in config.hidden_layer_sizes:
        for alpha in config.alpha_values:
            selection_scores, generated_count = _grouped_oof_scores(
                selection_features,
                selection_labels,
                selection_groups,
                config,
                architecture,
                alpha,
            )
            selected_threshold, threshold_table = select_mlp_threshold(
                selection_labels,
                selection_scores,
                selection_groups,
                config.minimum_recall,
            )
            candidates.append(
                {
                    "architecture": list(architecture),
                    "alpha": alpha,
                    "threshold": selected_threshold["threshold"],
                    "threshold_metrics": selected_threshold,
                    "threshold_table": threshold_table,
                    "generated_training_examples": generated_count,
                    "selection_split": "grouped_validation",
                }
            )

    if not candidates:
        raise DatasetFormatError("MLP configuration produced no candidates")
    selected_index = max(range(len(candidates)), key=lambda index: _selection_key(candidates[index]))
    selected = candidates[selected_index]
    selected_scaler, selected_estimator, generated_count = _fit_candidate(
        train_features,
        train_labels,
        config,
        tuple(selected["architecture"]),
        selected["alpha"],
    )
    feature_pipeline = WiFallMlpPipeline(
        selected_scaler,
        selected_estimator,
        config.feature,
        selected["threshold"],
    )
    model_path = output / "model.joblib"
    joblib.dump(feature_pipeline, model_path)

    model_metadata = {
        "family": "mlp",
        "name": "small_regularized_mlp",
        "training_code": "wifall.mlp",
        "artifact": model_path.name,
        "architecture": selected["architecture"],
        "parameters": {
            "activation": config.activation,
            "solver": config.solver,
            "alpha": selected["alpha"],
            "learning_rate_init": config.learning_rate_init,
            "max_iter": config.max_iter,
            "random_state": config.random_state,
        },
        "balancing": {
            "method": config.class_balance,
            "generated_training_examples": generated_count,
        },
        "decision_threshold": selected["threshold"],
        "threshold_policy": "minimum_grouped_validation_recall_then_specificity",
        "minimum_recall": config.minimum_recall,
        "calibration": {"method": "native_predict_proba", "positive_label": "fall"},
        "fit_splits": ["train"],
        "fit_participants": manifest["participants"]["train"],
        "score_orientation": {"positive_label": "fall", "score": "fall_probability"},
    }
    feature_metadata = config.feature.as_dict() | {"count": int(train_features.shape[1])}
    preprocessing_metadata = {
        "steps": [
            "per-recording CSI amplitude median/MAD normalization",
            f"{config.feature.window_count} temporal windows",
            "training-only StandardScaler",
            "training-only class balancing",
            "regularized MLP",
        ],
        "fit_split": "train",
    }
    split_metadata = {
        "strategy": "participant-held-out",
        "config": manifest["split_config"],
        "participants": manifest["participants"],
        "counts": manifest["counts"],
        "selection_participants": sorted(set(selection_groups)),
        "test_participants": manifest["participants"]["test"],
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
        features = np.vstack([extract_mlp_features(record, config.feature) for _, _, record in records])
        labels = np.asarray([record.target for _, _, record in records], dtype=object)
        evaluation_result = evaluate_classifier(
            feature_pipeline,
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
        evaluation["fit_status"] = "fit_data" if split == "train" else "held_out"
        evaluation["threshold_metrics"] = selected["threshold_table"] if split == "validation" else []
        evaluation["hard_example_analysis"] = analyze_hard_examples(
            evaluation,
            recording_observations(records),
            run_id=config.run_id,
            split_name=split,
            threshold=selected["threshold"],
        )
        evaluations[split] = evaluation
        evaluation_artifacts[split] = evaluation_result

    primary = evaluations["test"] if "test" in evaluations else evaluations["validation"]
    artifact = evaluation_artifacts["test" if "test" in evaluations else "validation"] | {
        "evaluation_split": "test" if "test" in evaluations else "validation",
        "evaluations": evaluations,
        "metrics": primary["metrics"],
        "confusion_matrix": primary["confusion_matrix"],
        "data_audit": audit,
        "selection": {
            "candidates": candidates,
            "selected": selected,
            "selection_split": "grouped_validation",
            "protocol": {
                "method": "participant_grouped_validation",
                "participants": sorted(set(selection_groups)),
                "test_participants": manifest["participants"]["test"],
                "fit_participants": manifest["participants"]["train"],
            },
        },
    }
    artifact_path = output / "run.json"
    save_evaluation_result(artifact, artifact_path)
    report_path = output / "report.html"
    report_path.write_text(render_html_report(artifact), encoding="utf-8")
    return MlpRun(model_path=model_path, artifact_path=artifact_path, report_path=report_path, audit_path=audit_path, artifact=artifact)


def load_mlp(path: str | Path) -> WiFallMlpPipeline:
    """Load a persisted MLP feature and prediction pipeline."""

    pipeline = joblib.load(Path(path).expanduser())
    if not isinstance(pipeline, WiFallMlpPipeline):
        raise TypeError("Model artifact is not a WiFallMlpPipeline")
    return pipeline
