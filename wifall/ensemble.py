"""Participant-safe score blending for compatible WiFall run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from evaluation import (
    analyze_hard_examples,
    evaluate_classifier,
    render_html_report,
    save_evaluation_result,
)
from evaluation.core import load_evaluation_result

from .diagnostics import recording_observations
from .loader import DatasetFormatError, load_split_manifest
from .training import load_split_records, validate_participant_splits

LABELS = ("non-fall", "fall")


@dataclass(frozen=True)
class EnsembleConfig:
    """Validation-only search configuration for score blending."""

    run_id: str = "wifall-svm-random-forest-ensemble"
    blend_weights: tuple[float, ...] = tuple(index / 10 for index in range(11))
    calibration_method: str = "sigmoid_logistic"
    selection_metric: str = "f1"
    random_state: int = 42

    def __post_init__(self) -> None:
        if not self.blend_weights:
            raise ValueError("blend_weights cannot be empty")
        if any(weight < 0.0 or weight > 1.0 for weight in self.blend_weights):
            raise ValueError("blend_weights must be between 0 and 1")
        if self.selection_metric not in {"f1", "balanced_accuracy", "recall"}:
            raise ValueError(f"Unsupported ensemble selection metric: {self.selection_metric}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "blend_weights": list(self.blend_weights),
            "calibration_method": self.calibration_method,
            "selection_metric": self.selection_metric,
            "random_state": self.random_state,
        }


@dataclass(frozen=True)
class EnsembleRun:
    artifact_path: Path
    report_path: Path
    artifact: dict[str, Any]


@dataclass(frozen=True)
class _Calibrator:
    estimator: LogisticRegression

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(scores.reshape(-1, 1))[:, 1]


def _source_path(path: str | Path) -> Path:
    source = Path(path).expanduser()
    if source.is_dir():
        source = source / "run.json"
    if not source.is_file():
        raise FileNotFoundError(f"Source run artifact does not exist: {source}")
    return source.resolve()


def _positive_scores(evaluation: dict[str, Any]) -> np.ndarray:
    values = evaluation.get("positive_scores")
    if values is None:
        values = evaluation.get("scores")
    if values is None:
        raise DatasetFormatError("Source evaluation has no decision scores")
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise DatasetFormatError(
            "Source evaluation must contain one-dimensional positive-class scores"
        )
    if not np.isfinite(array).all():
        raise DatasetFormatError("Source evaluation contains non-finite scores")
    return array


def _check_compatible(svm: dict[str, Any], random_forest: dict[str, Any]) -> None:
    if svm.get("dataset") != random_forest.get("dataset"):
        raise DatasetFormatError("Source runs use different datasets")
    if svm.get("task") != random_forest.get("task"):
        raise DatasetFormatError("Source runs use different task definitions")
    for split in ("train", "validation", "test"):
        left = svm.get("evaluations", {}).get(split)
        right = random_forest.get("evaluations", {}).get(split)
        if left is None or right is None:
            raise DatasetFormatError(f"Both source runs require {split} evaluations")
        if left.get("observation_ids") != right.get("observation_ids"):
            raise DatasetFormatError(f"Source observation IDs differ for {split}")
        if left.get("y_true") != right.get("y_true"):
            raise DatasetFormatError(f"Source labels differ for {split}")
        if left.get("group_ids") != right.get("group_ids"):
            raise DatasetFormatError(f"Source groups differ for {split}")


def _fit_calibrator(
    source: dict[str, Any], random_state: int
) -> _Calibrator:
    scores = np.concatenate(
        [_positive_scores(source["evaluations"][split]) for split in ("train", "validation")]
    )
    labels = np.concatenate(
        [
            np.asarray(source["evaluations"][split]["y_true"], dtype=object) == "fall"
            for split in ("train", "validation")
        ]
    ).astype(int)
    estimator = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )
    estimator.fit(scores.reshape(-1, 1), labels)
    return _Calibrator(estimator)


def _thresholds(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(scores)
    if unique.size == 0:
        raise DatasetFormatError("Cannot select threshold from empty validation scores")
    candidates = {0.0, 0.5, 1.0, *unique.tolist()}
    candidates.update(float((left + right) / 2.0) for left, right in pairwise(unique))
    return np.asarray(sorted(candidates), dtype=float)


def _specificity(labels: np.ndarray, predictions: np.ndarray) -> float:
    negatives = labels == "non-fall"
    if not negatives.any():
        return 0.0
    return float(np.sum(predictions[negatives] == "non-fall") / np.sum(negatives))


def _grouped_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: list[str],
    threshold: float,
) -> dict[str, float]:
    predictions = np.where(scores >= threshold, "fall", "non-fall")
    by_group: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        by_group.setdefault(group, []).append(index)
    values = {name: [] for name in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "specificity")}
    for indexes in by_group.values():
        group_labels = labels[indexes]
        group_predictions = predictions[indexes]
        values["accuracy"].append(float(accuracy_score(group_labels, group_predictions)))
        values["balanced_accuracy"].append(
            float(balanced_accuracy_score(group_labels, group_predictions))
        )
        values["precision"].append(
            float(precision_score(group_labels, group_predictions, zero_division=0, pos_label="fall"))
        )
        values["recall"].append(
            float(recall_score(group_labels, group_predictions, zero_division=0, pos_label="fall"))
        )
        values["f1"].append(
            float(f1_score(group_labels, group_predictions, zero_division=0, pos_label="fall"))
        )
        values["specificity"].append(_specificity(group_labels, group_predictions))
    return {name: float(np.mean(group_values)) for name, group_values in values.items()}


def _select_blend(
    svm_scores: np.ndarray,
    random_forest_scores: np.ndarray,
    labels: np.ndarray,
    groups: list[str],
    config: EnsembleConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for weight in config.blend_weights:
        blended = weight * svm_scores + (1.0 - weight) * random_forest_scores
        for threshold in _thresholds(blended):
            metrics = _grouped_metrics(labels, blended, groups, float(threshold))
            candidates.append(
                {
                    "svm_weight": float(weight),
                    "random_forest_weight": float(1.0 - weight),
                    "threshold": float(threshold),
                    "metrics": metrics,
                    "selection_metric": config.selection_metric,
                    "selection_split": "grouped_validation",
                }
            )
    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["metrics"][config.selection_metric],
            candidate["metrics"]["recall"],
            candidate["metrics"]["specificity"],
            -abs(candidate["svm_weight"] - 0.5),
        ),
    )
    return selected, candidates


def _evaluate_scores(
    labels: list[str],
    scores: np.ndarray,
    *,
    threshold: float,
    run_id: str,
    split: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    class ScoreModel:
        classes_ = np.asarray(LABELS)

        def predict(self, features: np.ndarray) -> np.ndarray:
            return np.where(scores[np.asarray(features, dtype=int)] >= threshold, "fall", "non-fall")

        def decision_function(self, features: np.ndarray) -> np.ndarray:
            return scores[np.asarray(features, dtype=int)]

    result = evaluate_classifier(
        ScoreModel(),
        np.arange(len(labels)),
        labels,
        label_order=LABELS,
        positive_label="fall",
        run_id=run_id,
        split_name=split,
        dataset_metadata=source["dataset"],
        task_metadata=source["task"],
        model_metadata={"family": "score_ensemble", "name": "calibrated_svm_random_forest_blend"},
        feature_metadata={"name": "calibrated_fall_score_blend", "sources": ["improved_svm", "random_forest"]},
        preprocessing_metadata={"steps": ["source score calibration", "weighted score blend"]},
        split_metadata=source["split"],
        observation_ids=source["evaluations"][split]["observation_ids"],
        group_ids=source["evaluations"][split]["group_ids"],
    )
    return result["evaluations"][split]


def train_ensemble(
    dataset_root: str | Path,
    split_manifest: str | Path,
    improved_svm_run: str | Path,
    random_forest_run: str | Path,
    output_dir: str | Path,
    config: EnsembleConfig | None = None,
) -> EnsembleRun:
    """Create a calibrated SVM/Random Forest ensemble artifact."""

    config = config or EnsembleConfig()
    svm_path = _source_path(improved_svm_run)
    random_forest_path = _source_path(random_forest_run)
    svm_result = load_evaluation_result(svm_path)
    random_forest_result = load_evaluation_result(random_forest_path)
    _check_compatible(svm_result, random_forest_result)
    manifest = load_split_manifest(split_manifest)
    split_records = load_split_records(Path(dataset_root).expanduser().resolve(), manifest)
    validate_participant_splits(split_records)
    if manifest["dataset"] != svm_result.get("dataset"):
        raise DatasetFormatError("Split manifest does not match source run dataset")

    svm_calibrator = _fit_calibrator(svm_result, config.random_state)
    random_forest_calibrator = _fit_calibrator(random_forest_result, config.random_state)
    calibrated_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in ("train", "validation", "test"):
        svm_scores = svm_calibrator.transform(_positive_scores(svm_result["evaluations"][split]))
        random_forest_scores = random_forest_calibrator.transform(
            _positive_scores(random_forest_result["evaluations"][split])
        )
        calibrated_scores[split] = (svm_scores, random_forest_scores)

    validation_labels = np.asarray(svm_result["evaluations"]["validation"]["y_true"], dtype=object)
    validation_groups = [
        str(group) for group in svm_result["evaluations"]["validation"]["group_ids"]
    ]
    selected, candidates = _select_blend(
        *calibrated_scores["validation"],
        validation_labels,
        validation_groups,
        config,
    )
    selected_weight = selected["svm_weight"]
    selected_threshold = selected["threshold"]

    source_metadata = svm_result
    evaluations: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        svm_scores, random_forest_scores = calibrated_scores[split]
        blended = selected_weight * svm_scores + (1.0 - selected_weight) * random_forest_scores
        labels = list(svm_result["evaluations"][split]["y_true"])
        evaluation = _evaluate_scores(
            labels,
            blended,
            threshold=selected_threshold,
            run_id=config.run_id,
            split=split,
            source=source_metadata,
        )
        evaluation["calibrated_source_scores"] = {
            "improved_svm": svm_scores.tolist(),
            "random_forest": random_forest_scores.tolist(),
        }
        evaluation["fit_status"] = "diagnostic" if split == "test" else "selection_source"
        evaluation["hard_example_analysis"] = analyze_hard_examples(
            evaluation,
            recording_observations(split_records[split]),
            run_id=config.run_id,
            split_name=split,
            threshold=selected_threshold,
        )
        evaluations[split] = evaluation

    model_metadata = {
        "family": "score_ensemble",
        "name": "calibrated_svm_random_forest_blend",
        "training_code": "wifall.ensemble",
        "source_runs": {
            "improved_svm": {"run_id": svm_result["run_id"], "artifact": svm_path.as_posix()},
            "random_forest": {
                "run_id": random_forest_result["run_id"],
                "artifact": random_forest_path.as_posix(),
            },
        },
        "calibration": {
            "method": config.calibration_method,
            "fit_splits": ["train", "validation"],
            "positive_label": "fall",
        },
        "blend_weight": {
            "improved_svm": selected_weight,
            "random_forest": 1.0 - selected_weight,
        },
        "decision_threshold": selected_threshold,
    }
    artifact = {
        "schema_version": 1,
        "run_id": config.run_id,
        "dataset": source_metadata["dataset"],
        "task": source_metadata["task"],
        "model": model_metadata,
        "features": {"name": "calibrated_fall_score_blend"},
        "preprocessing": {
            "steps": ["source score calibration", "weighted score blend"],
            "fit_splits": ["train", "validation"],
        },
        "split": source_metadata["split"],
        "evaluation_split": "test",
        "evaluations": evaluations,
        "metrics": evaluations["test"]["metrics"],
        "confusion_matrix": evaluations["test"]["confusion_matrix"],
        "selection": {
            "selection_split": "grouped_validation",
            "test_participants": manifest["participants"]["test"],
            "selected": selected,
            "candidates": candidates,
            "protocol": {
                "method": "equal_participant_contribution",
                "fit_splits": ["train", "validation"],
                "selection_split": "validation",
                "test_participants": manifest["participants"]["test"],
            },
        },
    }
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = output / "run.json"
    save_evaluation_result(artifact, artifact_path)
    report_path = output / "report.html"
    report_path.write_text(render_html_report(artifact), encoding="utf-8")
    return EnsembleRun(artifact_path, report_path, artifact)