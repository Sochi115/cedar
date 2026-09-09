"""Train and evaluate a reproducible WiFall SVM fall detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluation import (
    analyze_hard_examples,
    evaluate_classifier,
    save_evaluation_result,
)

from .diagnostics import recording_observations
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
class FeatureConfig:
    """Configuration for deterministic recording-level CSI features."""

    name: str = "csi_amplitude_statistics_v1"
    signal: str = "CSI amplitude"
    statistics: tuple[str, ...] = ("mean", "std", "min", "max", "delta_std")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"statistics": list(self.statistics)}


@dataclass(frozen=True)
class SvmConfig:
    """Configuration for the reproducible baseline SVM."""

    run_id: str = "wifall-svm-baseline"
    c: float = 1.0
    kernel: str = "rbf"
    gamma: str = "scale"
    class_weight: str = "balanced"
    feature: FeatureConfig = FeatureConfig()

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "c": self.c,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "class_weight": self.class_weight,
            "feature": self.feature.as_dict(),
        }


@dataclass(frozen=True)
class SvmRun:
    """Paths and run artifact produced by SVM training."""

    artifact_path: Path
    model_path: Path
    artifact: dict[str, Any]


def extract_recording_features(
    recording: Recording, config: FeatureConfig | None = None
) -> np.ndarray:
    """Extract one fixed-size feature vector from one recording.

    The baseline uses per-subcarrier amplitude statistics: mean, standard
    deviation, minimum, maximum, and standard deviation of adjacent-sample
    differences. Statistics are concatenated in that order, producing 260
    features for WiFall's 52 subcarriers. Scaling is fitted later on training
    recordings only by the persisted ``StandardScaler``.
    """

    config = config or FeatureConfig()
    expected_statistics = ("mean", "std", "min", "max", "delta_std")
    if config.statistics != expected_statistics:
        raise ValueError(f"Unsupported feature statistics: {config.statistics}")

    amplitude = np.abs(recording.csi)
    delta = np.diff(amplitude, axis=0)
    delta_std = delta.std(axis=0) if delta.size else np.zeros(amplitude.shape[1])
    features = np.concatenate(
        (
            amplitude.mean(axis=0),
            amplitude.std(axis=0),
            amplitude.min(axis=0),
            amplitude.max(axis=0),
            delta_std,
        )
    )
    return features.astype(np.float64, copy=False)


@dataclass
class WiFallSvmPipeline:
    """Persistable feature extraction, scaling, and SVM prediction pipeline."""

    estimator: Pipeline
    feature_config: FeatureConfig

    def features(self, recordings: list[Recording]) -> np.ndarray:
        return np.vstack(
            [extract_recording_features(recording, self.feature_config) for recording in recordings]
        )

    def predict_recordings(self, recordings: list[Recording]) -> np.ndarray:
        return self.estimator.predict(self.features(recordings))

    def decision_function_recordings(self, recordings: list[Recording]) -> np.ndarray:
        return self.estimator.decision_function(self.features(recordings))


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
        if recording_id not in references:
            raise DatasetFormatError(f"Manifest recording is missing from dataset: {recording_id}")
        reference = references[recording_id]
        recording = load_recording(reference)
        if recording.target != entry["target"]:
            raise DatasetFormatError(f"Target mismatch for manifest recording: {recording_id}")
        if recording.row_count != entry["row_count"]:
            raise DatasetFormatError(f"Row count mismatch for manifest recording: {recording_id}")
        split_records[split].append((reference.participant_id, reference.activity, recording))
    return split_records


def _validate_participant_splits(
    split_records: dict[str, list[tuple[str, str, Recording]]]
) -> None:
    participants = {
        split: {participant for participant, _, _ in records}
        for split, records in split_records.items()
    }
    for left_index, left_split in enumerate(SPLITS):
        for right_split in SPLITS[left_index + 1 :]:
            if participants[left_split].intersection(participants[right_split]):
                raise DatasetFormatError("Participant appears in multiple evaluation splits")


def train_svm(
    dataset_root: str | Path,
    split_manifest: str | Path,
    output_dir: str | Path,
    config: SvmConfig | None = None,
) -> SvmRun:
    """Train SVM on train participants and persist model plus generic results."""

    config = config or SvmConfig()
    root = Path(dataset_root).expanduser().resolve()
    manifest = load_split_manifest(split_manifest)
    split_records = _load_split_records(root, manifest)
    _validate_participant_splits(split_records)

    train_records = split_records["train"]
    if not train_records:
        raise DatasetFormatError("Split manifest has no training recordings")
    train_targets = {record.target for _, _, record in train_records}
    if train_targets != set(LABELS):
        raise DatasetFormatError("Training split must contain fall and non-fall recordings")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    C=config.c,
                    kernel=config.kernel,
                    gamma=config.gamma,
                    class_weight=config.class_weight,
                ),
            ),
        ]
    )
    feature_pipeline = WiFallSvmPipeline(pipeline, config.feature)
    train_features = feature_pipeline.features([record for _, _, record in train_records])
    train_labels = np.asarray(
        [record.target for _, _, record in train_records],
        dtype=object,
    )
    pipeline.fit(train_features, train_labels)

    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.joblib"
    joblib.dump(feature_pipeline, model_path)

    model_metadata = {
        "family": "svm",
        "name": "SVC",
        "artifact": model_path.name,
        "parameters": {
            "C": config.c,
            "kernel": config.kernel,
            "gamma": config.gamma,
            "class_weight": config.class_weight,
        },
    }
    feature_metadata = config.feature.as_dict() | {"count": int(train_features.shape[1])}
    preprocessing_metadata = {
        "steps": ["CSI amplitude statistics", "StandardScaler", "SVC"],
        "fit_split": "train",
    }
    split_metadata = {
        "strategy": "participant-held-out",
        "config": manifest["split_config"],
        "participants": manifest["participants"],
        "counts": manifest["counts"],
    }
    evaluation_results: dict[str, dict[str, Any]] = {}
    for split, records in split_records.items():
        if not records:
            continue
        features = feature_pipeline.features([record for _, _, record in records])
        labels = np.asarray(
            [record.target for _, _, record in records],
            dtype=object,
        )
        result = evaluate_classifier(
            pipeline,
            features,
            labels,
            label_order=LABELS,
            positive_label="fall",
            run_id=config.run_id,
            split_name=split,
            dataset_metadata=manifest["dataset"],
            task_metadata={"name": "binary_fall_detection", "signal": "WiFi CSI"},
            model_metadata=model_metadata,
            feature_metadata=feature_metadata,
            preprocessing_metadata=preprocessing_metadata,
            split_metadata=split_metadata,
            observation_ids=[record.reference.recording_id for _, _, record in records],
            group_ids=[participant for participant, _, _ in records],
        )
        evaluation = result["evaluations"][split]
        evaluation["hard_example_analysis"] = analyze_hard_examples(
            evaluation,
            recording_observations(records),
            run_id=config.run_id,
            split_name=split,
        )
        evaluation_results[split] = evaluation

    primary_split = "test" if "test" in evaluation_results else "train"
    primary_evaluation = evaluation_results[primary_split]
    artifact = result | {
        "evaluation_split": primary_split,
        "evaluations": evaluation_results,
        "metrics": primary_evaluation["metrics"],
        "confusion_matrix": primary_evaluation["confusion_matrix"],
    }
    artifact_path = output / "run.json"
    save_evaluation_result(artifact, artifact_path)
    return SvmRun(artifact_path=artifact_path, model_path=model_path, artifact=artifact)


def load_trained_svm(path: str | Path) -> WiFallSvmPipeline:
    """Load a persisted WiFall SVM feature and prediction pipeline."""

    pipeline = joblib.load(Path(path).expanduser())
    if not isinstance(pipeline, WiFallSvmPipeline):
        raise TypeError("Model artifact is not a WiFallSvmPipeline")
    return pipeline
