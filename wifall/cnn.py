"""Small participant-held-out CNN pipeline for WiFall fall detection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation import (
    analyze_hard_examples,
    evaluate_classifier,
    render_html_report,
    save_evaluation_result,
)

from .audit import audit_wifall
from .diagnostics import recording_observations
from .loader import DatasetFormatError, Recording, load_split_manifest
from .training import load_split_records, validate_participant_splits

LABELS = ("non-fall", "fall")


@dataclass(frozen=True)
class CnnFeatureConfig:
    """Fixed tensor representation preserving temporal and subcarrier axes."""

    name: str = "csi_normalized_amplitude_tensor_v1"
    signal: str = "normalized CSI amplitude"
    sample_count: int = 128
    subcarrier_count: int = 52

    def __post_init__(self) -> None:
        if self.sample_count < 4:
            raise ValueError("sample_count must be at least 4")
        if self.subcarrier_count != 52:
            raise ValueError("WiFall CNN currently requires 52 subcarriers")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "signal": self.signal,
            "sample_count": self.sample_count,
            "subcarrier_count": self.subcarrier_count,
            "shape": [1, self.sample_count, self.subcarrier_count],
        }


@dataclass(frozen=True)
class CnnConfig:
    """Configuration for small CNN search and safety threshold selection."""

    run_id: str = "wifall-cnn"
    channel_variants: tuple[tuple[int, ...], ...] = ((8, 16),)
    dropout_values: tuple[float, ...] = (0.1,)
    kernel_size: int = 3
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    positive_weight: float = 1.5
    minimum_recall: float = 0.95
    feature: CnnFeatureConfig = field(default_factory=CnnFeatureConfig)
    random_state: int = 42

    def __post_init__(self) -> None:
        if not self.channel_variants or any(
            len(channels) != 2 or any(channel < 1 for channel in channels)
            for channels in self.channel_variants
        ):
            raise ValueError("channel_variants must contain two positive channel counts")
        if not self.dropout_values or any(not 0.0 <= value < 1.0 for value in self.dropout_values):
            raise ValueError("dropout_values must be between 0 and 1")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.positive_weight <= 0.0:
            raise ValueError("positive_weight must be positive")
        if not 0.0 <= self.minimum_recall <= 1.0:
            raise ValueError("minimum_recall must be between 0 and 1")

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "channel_variants": [list(channels) for channels in self.channel_variants],
            "dropout_values": list(self.dropout_values),
            "kernel_size": self.kernel_size,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "positive_weight": self.positive_weight,
            "minimum_recall": self.minimum_recall,
            "feature": self.feature.as_dict(),
            "random_state": self.random_state,
        }


@dataclass(frozen=True)
class CnnRun:
    """Paths and result artifact produced by CNN training."""

    artifact_path: Path
    report_path: Path
    model_path: Path
    audit_path: Path
    artifact: dict[str, Any]


class SmallCnn(nn.Module):
    """Small 2D CNN over temporal samples by subcarrier."""

    def __init__(self, channels: tuple[int, int], kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(channels[0], channels[1], kernel_size, padding=padding),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[1], 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(values)).squeeze(1)


@dataclass
class WiFallCnnPipeline:
    """Persistable CNN tensor extraction, scoring, and thresholding pipeline."""

    network: SmallCnn
    feature_config: CnnFeatureConfig
    threshold: float
    positive_label: str = "fall"
    negative_label: str = "non-fall"

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray((self.negative_label, self.positive_label))

    def features(self, recordings: list[Recording]) -> np.ndarray:
        return np.stack([extract_cnn_tensor(recording, self.feature_config) for recording in recordings])

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        values_array = np.asarray(features, dtype=np.float32)
        expected_shape = (self.feature_config.sample_count, self.feature_config.subcarrier_count)
        if values_array.ndim != 4 or values_array.shape[1:] != (1, *expected_shape):
            raise ValueError(
                "CNN features must have shape "
                f"(observations, 1, {expected_shape[0]}, {expected_shape[1]})"
            )
        values = torch.from_numpy(values_array)
        self.network.eval()
        with torch.no_grad():
            return torch.sigmoid(self.network(values)).cpu().numpy()

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


def _normalized_amplitude(recording: Recording) -> np.ndarray:
    amplitude = np.abs(recording.csi).astype(np.float64)
    center = np.median(amplitude, axis=0)
    scale = np.median(np.abs(amplitude - center), axis=0)
    scale[scale < 1e-8] = 1.0
    return (amplitude - center) / scale


def extract_cnn_tensor(
    recording: Recording,
    config: CnnFeatureConfig | None = None,
) -> np.ndarray:
    """Resample one recording to deterministic ``[channel, time, subcarrier]``."""

    config = config or CnnFeatureConfig()
    if recording.csi.ndim != 2 or recording.csi.shape[1] != config.subcarrier_count:
        raise DatasetFormatError(
            f"CNN input requires {config.subcarrier_count} subcarriers"
        )
    amplitude = _normalized_amplitude(recording)
    source_positions = np.linspace(0.0, 1.0, amplitude.shape[0])
    target_positions = np.linspace(0.0, 1.0, config.sample_count)
    tensor = np.stack(
        [
            np.interp(target_positions, source_positions, amplitude[:, subcarrier])
            for subcarrier in range(config.subcarrier_count)
        ],
        axis=1,
    )
    return tensor[np.newaxis, :, :].astype(np.float32, copy=False)


def _seed(random_state: int) -> None:
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _fit_network(
    tensors: np.ndarray,
    labels: np.ndarray,
    config: CnnConfig,
    channels: tuple[int, int],
    dropout: float,
) -> SmallCnn:
    _seed(config.random_state)
    network = SmallCnn(channels, config.kernel_size, dropout)
    values = torch.from_numpy(np.asarray(tensors, dtype=np.float32))
    targets = torch.from_numpy((labels == "fall").astype(np.float32))
    dataset = TensorDataset(values, targets)
    generator = torch.Generator().manual_seed(config.random_state)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(config.positive_weight, dtype=torch.float32)
    )
    network.train()
    for _ in range(config.epochs):
        for batch_values, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(network(batch_values), batch_targets)
            loss.backward()
            optimizer.step()
    network.eval()
    return network


def _scores(network: SmallCnn, tensors: np.ndarray) -> np.ndarray:
    network.eval()
    with torch.no_grad():
        values = torch.from_numpy(np.asarray(tensors, dtype=np.float32))
        return torch.sigmoid(network(values)).cpu().numpy()


def _threshold_candidates(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(scores)
    if unique.size == 0:
        raise DatasetFormatError("Cannot select a CNN threshold from empty scores")
    candidates = {0.0, 0.5, 1.0, *unique.tolist()}
    candidates.update(float((left + right) / 2.0) for left, right in pairwise(unique))
    return np.asarray(sorted(candidates), dtype=float)


def _specificity(labels: np.ndarray, predictions: np.ndarray) -> float:
    negative = labels == "non-fall"
    return float(np.mean(predictions[negative] == "non-fall")) if negative.any() else 0.0


def _threshold_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: list[str],
    threshold: float,
) -> dict[str, Any]:
    predictions = np.where(scores >= threshold, "fall", "non-fall")
    group_recalls = []
    group_specificities = []
    for group in sorted(set(groups)):
        indexes = np.asarray([index for index, value in enumerate(groups) if value == group])
        group_labels = labels[indexes]
        group_predictions = predictions[indexes]
        falls = group_labels == "fall"
        group_recalls.append(float(np.mean(group_predictions[falls] == "fall")) if falls.any() else 0.0)
        group_specificities.append(_specificity(group_labels, group_predictions))
    recall = float(np.mean(predictions[labels == "fall"] == "fall")) if np.any(labels == "fall") else 0.0
    precision = float(
        np.sum((labels == "fall") & (predictions == "fall")) / max(np.sum(predictions == "fall"), 1)
    )
    specificity = _specificity(labels, predictions)
    false_positive_rate = 1.0 - specificity
    return {
        "threshold": float(threshold),
        "accuracy": float(np.mean(labels == predictions)),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "f1": float(f1_score(labels, predictions, pos_label="fall", zero_division=0)),
        "minimum_group_recall": min(group_recalls) if group_recalls else 0.0,
        "group_specificity": float(np.mean(group_specificities)) if group_specificities else 0.0,
        "eligible": False,
    }


def select_cnn_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: list[str],
    minimum_recall: float = 0.95,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select threshold requiring minimum recall for every validation group."""

    if labels.size != scores.size or labels.size != len(groups):
        raise ValueError("CNN labels, scores, and groups must have equal lengths")
    if not 0.0 <= minimum_recall <= 1.0:
        raise ValueError("minimum_recall must be between 0 and 1")
    candidates = [_threshold_metrics(labels, scores, groups, threshold) for threshold in _threshold_candidates(scores)]
    for candidate in candidates:
        candidate["eligible"] = candidate["minimum_group_recall"] >= minimum_recall
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


def _grouped_oof_scores(
    tensors: np.ndarray,
    labels: np.ndarray,
    groups: list[str],
    config: CnnConfig,
    channels: tuple[int, int],
    dropout: float,
) -> np.ndarray:
    scores = np.empty(labels.size, dtype=float)
    for held_out_group in sorted(set(groups)):
        validation_indexes = np.asarray([index for index, group in enumerate(groups) if group == held_out_group])
        train_indexes = np.asarray([index for index, group in enumerate(groups) if group != held_out_group])
        network = _fit_network(tensors[train_indexes], labels[train_indexes], config, channels, dropout)
        scores[validation_indexes] = _scores(network, tensors[validation_indexes])
    return scores


def train_cnn(
    dataset_root: str | Path,
    split_manifest: str | Path,
    output_dir: str | Path,
    config: CnnConfig | None = None,
) -> CnnRun:
    """Train, select, evaluate, and persist a small CNN pipeline."""

    config = config or CnnConfig()
    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_split_manifest(split_manifest)
    split_records = load_split_records(root, manifest)
    validate_participant_splits(split_records)
    if not split_records["train"] or not split_records["validation"]:
        raise DatasetFormatError("CNN requires train and validation recordings")

    audit = audit_wifall(root)
    audit_path = output / "dataset_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selection_records = split_records["train"] + split_records["validation"]
    selection_tensors = np.stack([extract_cnn_tensor(record, config.feature) for _, _, record in selection_records])
    selection_labels = np.asarray([record.target for _, _, record in selection_records], dtype=object)
    selection_groups = [participant for participant, _, _ in selection_records]

    candidates: list[dict[str, Any]] = []
    for channels in config.channel_variants:
        for dropout in config.dropout_values:
            selection_scores = _grouped_oof_scores(
                selection_tensors,
                selection_labels,
                selection_groups,
                config,
                channels,
                dropout,
            )
            selected_threshold, threshold_table = select_cnn_threshold(
                selection_labels,
                selection_scores,
                selection_groups,
                config.minimum_recall,
            )
            candidates.append(
                {
                    "channels": list(channels),
                    "dropout": dropout,
                    "threshold": selected_threshold["threshold"],
                    "threshold_metrics": selected_threshold,
                    "threshold_table": threshold_table,
                    "selection_split": "grouped_validation",
                }
            )

    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["threshold_metrics"]["eligible"],
            candidate["threshold_metrics"]["minimum_group_recall"],
            candidate["threshold_metrics"]["group_specificity"],
            candidate["threshold_metrics"]["f1"],
        ),
    )
    selected_channels = tuple(selected["channels"])
    selected_dropout = selected["dropout"]
    final_network = _fit_network(
        selection_tensors,
        selection_labels,
        config,
        selected_channels,
        selected_dropout,
    )
    pipeline = WiFallCnnPipeline(
        final_network,
        config.feature,
        selected["threshold"],
    )
    model_path = output / "model.joblib"
    joblib.dump(pipeline, model_path)

    model_metadata = {
        "family": "cnn",
        "name": "small_2d_cnn",
        "training_code": "wifall.cnn",
        "artifact": model_path.name,
        "architecture": {
            "type": "2d_convolution",
            "channels": list(selected_channels),
            "kernel_size": config.kernel_size,
            "pooling": "max_pool_then_adaptive_average_pool",
            "dropout": selected_dropout,
        },
        "parameters": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "positive_weight": config.positive_weight,
            "random_state": config.random_state,
        },
        "decision_threshold": selected["threshold"],
        "threshold_policy": "minimum_grouped_validation_recall_then_specificity",
        "minimum_recall": config.minimum_recall,
        "calibration": {"method": "native_sigmoid", "positive_label": "fall"},
        "fit_splits": ["train", "validation"],
        "fit_participants": sorted(set(selection_groups)),
        "score_orientation": {"positive_label": "fall", "score": "fall_probability"},
    }
    feature_metadata = config.feature.as_dict()
    preprocessing_metadata = {
        "steps": [
            "per-recording CSI amplitude median/MAD normalization",
            "deterministic temporal resampling",
            "training-only participant-grouped selection",
            "CNN native sigmoid score",
        ],
        "fit_split": "train+validation_after_selection",
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
        tensors = np.stack([extract_cnn_tensor(record, config.feature) for _, _, record in records])
        labels = np.asarray([record.target for _, _, record in records], dtype=object)
        evaluation_result = evaluate_classifier(
            pipeline,
            tensors,
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
            "protocol": {
                "method": "participant_grouped_oof",
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
    return CnnRun(artifact_path, report_path, model_path, audit_path, artifact)


def load_cnn(path: str | Path) -> WiFallCnnPipeline:
    """Load a persisted CNN feature and prediction pipeline."""

    pipeline = joblib.load(Path(path).expanduser())
    if not isinstance(pipeline, WiFallCnnPipeline):
        raise TypeError("Model artifact is not a WiFallCnnPipeline")
    return pipeline
