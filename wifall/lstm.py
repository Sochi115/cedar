"""Participant-held-out causal LSTM pipeline for duration-normalized WiFall CSI."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime
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
from .processed import load_duration_normalized_dataset
from .training import load_split_records, validate_participant_splits

LABELS = ("non-fall", "fall")


@dataclass(frozen=True)
class LstmFeatureConfig:
    """Causal fixed-rate sequence and rolling alert window configuration."""

    name: str = "duration_normalized_csi_amplitude_sequence_v3"
    signal: str = "per-window normalized CSI amplitude"
    sample_count: int = 128
    subcarrier_count: int = 52
    causal_window: int = 32
    window_stride: int = 4

    def __post_init__(self) -> None:
        if self.sample_count < 2:
            raise ValueError("sample_count must be at least 2")
        if self.subcarrier_count != 52:
            raise ValueError("WiFall LSTM currently requires 52 subcarriers")
        if self.causal_window < 2 or self.causal_window > self.sample_count:
            raise ValueError("causal_window must be between 2 and sample_count")
        if self.window_stride < 1:
            raise ValueError("window_stride must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "signal": self.signal,
            "sample_count": self.sample_count,
            "subcarrier_count": self.subcarrier_count,
            "causal_window": self.causal_window,
            "window_stride": self.window_stride,
            "shape": [self.sample_count, self.subcarrier_count],
            "sequence_semantics": "fixed-rate normalized-time sample by subcarrier",
            "normalization_semantics": "median/MAD over current rolling window only",
        }


@dataclass(frozen=True)
class LstmConfig:
    """Configuration for grouped causal architecture and threshold selection."""

    run_id: str = "wifall-lstm-duration-normalized"
    hidden_sizes: tuple[int, ...] = (32,)
    layer_counts: tuple[int, ...] = (1,)
    dropout_values: tuple[float, ...] = (0.1,)
    optimizer_names: tuple[str, ...] = ("adam",)
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    positive_weight: float = 1.5
    minimum_recall: float = 0.95
    early_stopping_patience: int = 5
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    min_epochs: int = 1
    feature: LstmFeatureConfig = field(default_factory=LstmFeatureConfig)
    random_state: int = 42

    def __post_init__(self) -> None:
        if not self.hidden_sizes or any(size < 1 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive values")
        if not self.layer_counts or any(count < 1 for count in self.layer_counts):
            raise ValueError("layer_counts must contain positive values")
        if not self.dropout_values or any(not 0.0 <= value < 1.0 for value in self.dropout_values):
            raise ValueError("dropout_values must be between 0 and 1")
        if not self.optimizer_names or any(name not in {"adam", "adamw"} for name in self.optimizer_names):
            raise ValueError("optimizer_names must contain only 'adam' or 'adamw'")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.positive_weight <= 0.0:
            raise ValueError("positive_weight must be positive")
        if not 0.0 <= self.minimum_recall <= 1.0:
            raise ValueError("minimum_recall must be between 0 and 1")
        if self.early_stopping_patience < 1 or self.scheduler_patience < 0:
            raise ValueError("stopping patience must be non-negative and early stopping positive")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("scheduler_factor must be between 0 and 1")
        if self.min_epochs < 1 or self.min_epochs > self.epochs:
            raise ValueError("min_epochs must be between 1 and epochs")

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "hidden_sizes": list(self.hidden_sizes),
            "layer_counts": list(self.layer_counts),
            "dropout_values": list(self.dropout_values),
            "optimizer_names": list(self.optimizer_names),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "positive_weight": self.positive_weight,
            "minimum_recall": self.minimum_recall,
            "early_stopping_patience": self.early_stopping_patience,
            "scheduler_patience": self.scheduler_patience,
            "scheduler_factor": self.scheduler_factor,
            "min_epochs": self.min_epochs,
            "feature": self.feature.as_dict(),
            "random_state": self.random_state,
        }


@dataclass(frozen=True)
class LstmRun:
    """Paths and result artifact produced by LSTM training."""

    artifact_path: Path
    report_path: Path
    model_path: Path
    audit_path: Path
    artifact: dict[str, Any]


class SmallLstm(nn.Module):
    """Small unidirectional sequence classifier over normalized time."""

    def __init__(self, input_size: int, hidden_size: int, layer_count: int, dropout: float) -> None:
        super().__init__()
        self.recurrent = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layer_count,
            batch_first=True,
            dropout=dropout if layer_count > 1 else 0.0,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.recurrent(values)
        return self.classifier(hidden[-1]).squeeze(1)


@dataclass
class WiFallLstmPipeline:
    """Persistable causal rolling-window LSTM pipeline."""

    network: SmallLstm
    feature_config: LstmFeatureConfig
    threshold: float
    positive_label: str = "fall"
    negative_label: str = "non-fall"

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray((self.negative_label, self.positive_label))

    def features(self, recordings: list[Recording]) -> np.ndarray:
        expected_rows = self.feature_config.sample_count
        if any(recording.row_count != expected_rows for recording in recordings):
            raise DatasetFormatError(
                "Causal LSTM inference requires fixed-rate duration-normalized recordings "
                f"with {expected_rows} samples"
            )
        return np.stack([extract_lstm_sequence(recording, self.feature_config) for recording in recordings])

    def _validate_features(self, features: np.ndarray) -> np.ndarray:
        values_array = np.asarray(features, dtype=np.float32)
        expected_shape = (self.feature_config.sample_count, self.feature_config.subcarrier_count)
        if values_array.ndim != 3 or values_array.shape[1:] != expected_shape:
            raise ValueError(
                "LSTM features must have shape "
                f"(observations, {expected_shape[0]}, {expected_shape[1]})"
            )
        return values_array

    def causal_scores(self, sequence: np.ndarray) -> np.ndarray:
        """Return one score per sample; each score uses current and past samples only."""

        values = np.asarray(sequence, dtype=np.float32)
        expected_shape = (self.feature_config.sample_count, self.feature_config.subcarrier_count)
        if values.ndim != 2 or values.shape != expected_shape:
            raise ValueError(
                "LSTM sequence must have shape "
                f"({expected_shape[0]}, {expected_shape[1]})"
            )
        windows, starts = _rolling_windows(
            values,
            self.feature_config.causal_window,
            stride=1,
        )
        scores = _scores(self.network, _normalize_windows(windows))
        trace = np.full(values.shape[0], np.nan, dtype=float)
        ends = starts + self.feature_config.causal_window - 1
        trace[ends] = scores
        return trace

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        """Return recording scores as maximum causal rolling-window score."""

        values = self._validate_features(features)
        return np.asarray([np.nanmax(self.causal_scores(sequence)) for sequence in values])

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


def _amplitude_sequence(recording: Recording) -> np.ndarray:
    """Return raw amplitude; normalization happens inside each causal window."""

    return np.abs(recording.csi).astype(np.float64)


def _normalize_windows(windows: np.ndarray) -> np.ndarray:
    """Normalize each rolling window without reading outside that window."""

    center = np.median(windows, axis=1, keepdims=True)
    scale = np.median(np.abs(windows - center), axis=1, keepdims=True)
    scale[scale < 1e-8] = 1.0
    return ((windows - center) / scale).astype(np.float32, copy=False)


def extract_lstm_sequence(
    recording: Recording,
    config: LstmFeatureConfig | None = None,
) -> np.ndarray:
    """Extract a fixed-rate causal ``[time, subcarrier]`` amplitude sequence.

    Persisted duration-normalized recordings already have ``sample_count`` rows.
    Shorter streams are right-padded with their latest sample; longer streams use
    their latest fixed-size history. Neither operation reads future samples.
    """

    config = config or LstmFeatureConfig()
    if recording.csi.ndim != 2 or recording.csi.shape[1] != config.subcarrier_count:
        raise DatasetFormatError(
            f"LSTM input requires {config.subcarrier_count} subcarriers"
        )
    amplitude = _amplitude_sequence(recording)
    if amplitude.shape[0] > config.sample_count:
        sequence = amplitude[-config.sample_count :]
    elif amplitude.shape[0] < config.sample_count:
        sequence = np.pad(
            amplitude,
            ((0, config.sample_count - amplitude.shape[0]), (0, 0)),
            mode="edge",
        )
    else:
        sequence = amplitude
    return sequence.astype(np.float32, copy=False)


def _rolling_windows(
    sequence: np.ndarray,
    lookback: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sequence.ndim != 2 or sequence.shape[0] < lookback:
        raise ValueError("sequence must contain at least lookback samples")
    starts = list(range(0, sequence.shape[0] - lookback + 1, stride))
    final_start = sequence.shape[0] - lookback
    if starts[-1] != final_start:
        starts.append(final_start)
    starts_array = np.asarray(starts, dtype=np.int64)
    windows = np.stack([sequence[start : start + lookback] for start in starts_array])
    return windows, starts_array


def _window_dataset(
    sequences: np.ndarray,
    labels: np.ndarray,
    config: LstmConfig,
) -> tuple[np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    window_labels: list[np.ndarray] = []
    for sequence, label in zip(sequences, labels):
        values, _ = _rolling_windows(
            sequence,
            config.feature.causal_window,
            config.feature.window_stride,
        )
        windows.append(_normalize_windows(values))
        window_labels.append(np.full(values.shape[0], label, dtype=object))
    return np.concatenate(windows), np.concatenate(window_labels)


def _seed(random_state: int) -> None:
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _loss(
    network: SmallLstm,
    values: np.ndarray,
    labels: np.ndarray,
    loss_function: nn.Module,
) -> float:
    network.eval()
    with torch.no_grad():
        logits = network(torch.from_numpy(np.asarray(values, dtype=np.float32)))
        targets = torch.from_numpy((labels == "fall").astype(np.float32))
        return float(loss_function(logits, targets).item())


def _fit_network(
    sequences: np.ndarray,
    labels: np.ndarray,
    config: LstmConfig,
    hidden_size: int,
    layer_count: int,
    dropout: float,
    optimizer_name: str,
    *,
    validation_sequences: np.ndarray | None = None,
    validation_labels: np.ndarray | None = None,
    epochs_override: int | None = None,
) -> tuple[SmallLstm, int]:
    if np.unique(labels).size < 2:
        raise DatasetFormatError("LSTM training requires both fall and non-fall classes")
    if (validation_sequences is None) != (validation_labels is None):
        raise ValueError("validation sequences and labels must be provided together")
    _seed(config.random_state)
    network = SmallLstm(config.feature.subcarrier_count, hidden_size, layer_count, dropout)
    values = torch.from_numpy(np.asarray(sequences, dtype=np.float32))
    targets = torch.from_numpy((labels == "fall").astype(np.float32))
    dataset = TensorDataset(values, targets)
    generator = torch.Generator().manual_seed(config.random_state)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
    optimizer_type = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[optimizer_name]
    optimizer = optimizer_type(
        network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(config.positive_weight, dtype=torch.float32)
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
        )
        if validation_sequences is not None
        else None
    )
    epochs = epochs_override or config.epochs
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = epochs
    stale_epochs = 0
    for epoch in range(epochs):
        network.train()
        for batch_values, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(network(batch_values), batch_targets)
            loss.backward()
            optimizer.step()
        if validation_sequences is None:
            continue
        validation_loss = _loss(network, validation_sequences, validation_labels, loss_function)
        scheduler.step(validation_loss)
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(network.state_dict())
            stale_epochs = 0
        elif epoch + 1 >= config.min_epochs:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break
    if best_state is not None:
        network.load_state_dict(best_state)
    network.eval()
    return network, best_epoch


def _scores(network: SmallLstm, sequences: np.ndarray) -> np.ndarray:
    network.eval()
    with torch.no_grad():
        values = torch.from_numpy(np.asarray(sequences, dtype=np.float32))
        return torch.sigmoid(network(values)).cpu().numpy()


def _recording_scores(
    network: SmallLstm,
    sequences: np.ndarray,
    config: LstmConfig,
) -> np.ndarray:
    return np.asarray(
        [
            float(
                np.max(
                    _scores(
                        network,
                        _normalize_windows(
                            _rolling_windows(sequence, config.feature.causal_window, 1)[0]
                        ),
                    )
                )
            )
            for sequence in sequences
        ]
    )


def _threshold_candidates(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(scores)
    if unique.size == 0:
        raise DatasetFormatError("Cannot select an LSTM threshold from empty scores")
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
    specificity = _specificity(labels, predictions)
    precision = float(
        np.sum((labels == "fall") & (predictions == "fall")) / max(np.sum(predictions == "fall"), 1)
    )
    return {
        "threshold": float(threshold),
        "accuracy": float(np.mean(labels == predictions)),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "false_positive_rate": 1.0 - specificity,
        "f1": float(f1_score(labels, predictions, pos_label="fall", zero_division=0)),
        "minimum_group_recall": min(group_recalls) if group_recalls else 0.0,
        "group_specificity": float(np.mean(group_specificities)) if group_specificities else 0.0,
        "eligible": False,
    }


def select_lstm_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: list[str],
    minimum_recall: float = 0.95,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select threshold requiring minimum recall for every validation participant."""

    if labels.size != scores.size or labels.size != len(groups):
        raise ValueError("LSTM labels, scores, and groups must have equal lengths")
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
    sequences: np.ndarray,
    labels: np.ndarray,
    groups: list[str],
    config: LstmConfig,
    hidden_size: int,
    layer_count: int,
    dropout: float,
    optimizer_name: str,
) -> tuple[np.ndarray, list[int]]:
    """Return outer grouped scores and inner-validation stopping epochs."""

    scores = np.empty(labels.size, dtype=float)
    stopping_epochs: list[int] = []
    for held_out_group in sorted(set(groups)):
        validation_indexes = np.asarray([index for index, group in enumerate(groups) if group == held_out_group])
        train_indexes = np.asarray([index for index, group in enumerate(groups) if group != held_out_group])
        train_groups = sorted({groups[index] for index in train_indexes})
        if len(train_groups) > 1:
            inner_group = train_groups[-1]
            inner_indexes = np.asarray([index for index in train_indexes if groups[index] == inner_group])
            fit_indexes = np.asarray([index for index in train_indexes if groups[index] != inner_group])
        else:
            inner_indexes = None
            fit_indexes = train_indexes
        fit_windows, fit_labels = _window_dataset(sequences[fit_indexes], labels[fit_indexes], config)
        validation_windows = validation_labels = None
        if inner_indexes is not None:
            validation_windows, validation_labels = _window_dataset(
                sequences[inner_indexes], labels[inner_indexes], config
            )
        network, best_epoch = _fit_network(
            fit_windows,
            fit_labels,
            config,
            hidden_size,
            layer_count,
            dropout,
            optimizer_name,
            validation_sequences=validation_windows,
            validation_labels=validation_labels,
        )
        stopping_epochs.append(best_epoch)
        scores[validation_indexes] = _recording_scores(network, sequences[validation_indexes], config)
    return scores, stopping_epochs


def _alerting_diagnostics(
    pipeline: WiFallLstmPipeline,
    sequences: np.ndarray,
    labels: np.ndarray,
    observation_ids: list[str],
    groups: list[str],
    durations: list[float],
) -> dict[str, Any]:
    observation_alerts = []
    first_alerts: list[int] = []
    for observation_id, group, label, sequence, duration in zip(
        observation_ids, groups, labels, sequences, durations
    ):
        trace = pipeline.causal_scores(sequence)
        alert_indexes = np.flatnonzero(np.isfinite(trace) & (trace >= pipeline.threshold))
        first_alert = int(alert_indexes[0]) if alert_indexes.size else None
        if first_alert is not None:
            first_alerts.append(first_alert)
        sample_period = duration / max(sequence.shape[0] - 1, 1)
        observation_alerts.append(
            {
                "observation_id": observation_id,
                "group_id": group,
                "label": str(label),
                "first_alert_sample": first_alert,
                "first_alert_seconds": (
                    float(first_alert * sample_period) if first_alert is not None else None
                ),
            }
        )
    fall_alerts = [
        item for item in observation_alerts if item["label"] == "fall"
    ]
    non_fall_alerts = [
        item for item in observation_alerts if item["label"] == "non-fall"
    ]
    alerted_falls = [item for item in fall_alerts if item["first_alert_sample"] is not None]
    false_alerts = [item for item in non_fall_alerts if item["first_alert_sample"] is not None]
    latencies = [item["first_alert_sample"] for item in alerted_falls]
    latency_seconds = [item["first_alert_seconds"] for item in alerted_falls]
    return {
        "causal": True,
        "alert_cadence_samples": 1,
        "max_lookback_samples": pipeline.feature_config.causal_window,
        "warmup_samples": pipeline.feature_config.causal_window - 1,
        "state_reset": "stateless rolling window resets at each recording or stream session",
        "indexing": "zero-based sample index",
        "observation_alerts": observation_alerts,
        "summary": {
            "fall_alert_rate": float(len(alerted_falls) / max(len(fall_alerts), 1)),
            "false_alarm_rate": float(len(false_alerts) / max(len(non_fall_alerts), 1)),
            "median_first_alert_sample": float(np.median(latencies)) if latencies else None,
            "minimum_first_alert_sample": min(latencies) if latencies else None,
            "maximum_first_alert_sample": max(latencies) if latencies else None,
            "median_first_alert_seconds": float(np.median(latency_seconds)) if latency_seconds else None,
            "minimum_first_alert_seconds": min(latency_seconds) if latency_seconds else None,
            "maximum_first_alert_seconds": max(latency_seconds) if latency_seconds else None,
            "alert_count": len(first_alerts),
        },
    }


def train_lstm(
    dataset_root: str | Path,
    split_manifest: str | Path,
    output_dir: str | Path,
    config: LstmConfig | None = None,
    *,
    processed_dataset: str | Path | None = None,
) -> LstmRun:
    """Train, select, evaluate, and persist a causal LSTM from persisted CSI data."""

    config = config or LstmConfig()
    if processed_dataset is None:
        raise DatasetFormatError("LSTM requires a persisted --processed-dataset")
    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_split_manifest(split_manifest)
    split_records = load_split_records(root, manifest, processed_dataset=processed_dataset)
    validate_participant_splits(split_records)
    processed = load_duration_normalized_dataset(processed_dataset)
    if processed.config.sample_count != config.feature.sample_count:
        raise DatasetFormatError(
            "LSTM feature sample_count must match persisted duration-normalized dataset: "
            f"{config.feature.sample_count} != {processed.config.sample_count}"
        )
    if not split_records["train"] or not split_records["validation"]:
        raise DatasetFormatError("LSTM requires train and validation recordings")

    audit = audit_wifall(root)
    audit_path = output / "dataset_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selection_records = split_records["train"] + split_records["validation"]
    selection_sequences = np.stack(
        [extract_lstm_sequence(record, config.feature) for _, _, record in selection_records]
    )
    selection_labels = np.asarray([record.target for _, _, record in selection_records], dtype=object)
    selection_groups = [participant for participant, _, _ in selection_records]

    candidates: list[dict[str, Any]] = []
    for hidden_size in config.hidden_sizes:
        for layer_count in config.layer_counts:
            for dropout in config.dropout_values:
                for optimizer_name in config.optimizer_names:
                    selection_scores, stopping_epochs = _grouped_oof_scores(
                    selection_sequences,
                    selection_labels,
                    selection_groups,
                    config,
                    hidden_size,
                    layer_count,
                    dropout,
                    optimizer_name,
                )
                selected_threshold, threshold_table = select_lstm_threshold(
                    selection_labels,
                    selection_scores,
                    selection_groups,
                    config.minimum_recall,
                )
                candidates.append(
                    {
                        "hidden_size": hidden_size,
                        "layer_count": layer_count,
                        "dropout": dropout,
                        "optimizer": optimizer_name,
                        "threshold": selected_threshold["threshold"],
                        "threshold_metrics": selected_threshold,
                        "threshold_table": threshold_table,
                        "effective_epochs": round(np.median(stopping_epochs)),
                        "stopping_epochs": stopping_epochs,
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
            -candidate["threshold"],
        ),
    )
    selection_windows, selection_window_labels = _window_dataset(
        selection_sequences,
        selection_labels,
        config,
    )
    final_network, _ = _fit_network(
        selection_windows,
        selection_window_labels,
        config,
        selected["hidden_size"],
        selected["layer_count"],
        selected["dropout"],
        selected["optimizer"],
        epochs_override=selected["effective_epochs"],
    )
    pipeline = WiFallLstmPipeline(final_network, config.feature, selected["threshold"])
    model_path = output / "model.joblib"
    joblib.dump(pipeline, model_path)

    model_metadata = {
        "family": "lstm",
        "name": "small_causal_lstm_duration_normalized",
        "training_code": "wifall.lstm",
        "artifact": model_path.name,
        "architecture": {
            "type": "unidirectional_lstm_rolling_window_classifier",
            "input_size": config.feature.subcarrier_count,
            "hidden_size": selected["hidden_size"],
            "layer_count": selected["layer_count"],
            "dropout": selected["dropout"],
            "bidirectional": False,
        },
        "parameters": {
            "epochs": config.epochs,
            "effective_epochs": selected["effective_epochs"],
            "optimizer": selected["optimizer"],
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "positive_weight": config.positive_weight,
            "random_state": config.random_state,
        },
        "real_time": {
            "causal": True,
            "max_lookback_samples": config.feature.causal_window,
            "alert_cadence_samples": 1,
            "warmup_samples": config.feature.causal_window - 1,
            "state_reset": "clear rolling window at recording or stream session boundary",
            "expected_alert_delay": "at least the rolling-window warmup; report estimates seconds from source duration or stream rate",
            "input_contract": "caller supplies fixed-rate duration-normalized samples; no end-of-recording resampling at inference",
        },
        "decision_threshold": selected["threshold"],
        "threshold_policy": "minimum_grouped_validation_recall_then_specificity",
        "minimum_recall": config.minimum_recall,
        "calibration": {"method": "native_sigmoid", "positive_label": "fall"},
        "fit_splits": ["train", "validation"],
        "fit_participants": sorted(set(selection_groups)),
        "score_orientation": {"positive_label": "fall", "score": "fall_probability"},
    }
    training_metadata = {
        "early_stopping": {
            "enabled": True,
            "patience": config.early_stopping_patience,
            "min_epochs": config.min_epochs,
            "selection": "inner_grouped_non_test_validation",
            "effective_epochs": selected["effective_epochs"],
        },
        "scheduler": {
            "method": "ReduceLROnPlateau",
            "patience": config.scheduler_patience,
            "factor": config.scheduler_factor,
            "fit_split": "inner_grouped_non_test_validation",
        },
        "window_training": {
            "label_scope": "recording label repeated over causal windows",
            "window_stride": config.feature.window_stride,
        },
    }
    feature_metadata = config.feature.as_dict()
    preprocessing_metadata = {
        "steps": [
            "persisted duration-normalized fixed-rate CSI sequence",
            "per-window median/MAD amplitude normalization using current and past samples",
            "causal rolling-window training and inference",
            "training-only participant-grouped architecture and threshold selection",
            "LSTM native sigmoid score",
        ],
        "fit_split": "train+validation_after_selection",
        "duration_normalized_dataset": processed.metadata,
        "real_time_contract": model_metadata["real_time"],
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
        "data_source": {"type": "duration_normalized_dataset", "path": str(Path(processed_dataset))},
    }

    evaluations: dict[str, dict[str, Any]] = {}
    evaluation_artifacts: dict[str, dict[str, Any]] = {}
    for split, records in split_records.items():
        if not records:
            continue
        sequences = np.stack([extract_lstm_sequence(record, config.feature) for _, _, record in records])
        labels = np.asarray([record.target for _, _, record in records], dtype=object)
        observation_ids = [record.reference.recording_id for _, _, record in records]
        group_ids = [participant for participant, _, _ in records]
        durations = [
            (datetime.fromisoformat(record.timestamps[-1]) - datetime.fromisoformat(record.timestamps[0])).total_seconds()
            for _, _, record in records
        ]
        evaluation_result = evaluate_classifier(
            pipeline,
            sequences,
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
            observation_ids=observation_ids,
            group_ids=group_ids,
        )
        evaluation = evaluation_result["evaluations"][split]
        evaluation["fit_status"] = "fit_data" if split in {"train", "validation"} else "held_out"
        evaluation["threshold_metrics"] = selected["threshold_table"] if split == "validation" else []
        evaluation["alerting"] = _alerting_diagnostics(
            pipeline,
            sequences,
            labels,
            observation_ids,
            group_ids,
            durations,
        )
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
        "data_source": metadata["data_source"],
        "training": training_metadata,
        "evaluations": evaluations,
        "metrics": primary["metrics"],
        "confusion_matrix": primary["confusion_matrix"],
        "data_audit": audit,
        "selection": {
            "candidates": candidates,
            "selected": selected,
            "selection_split": "grouped_validation",
            "protocol": {
                "method": "participant_grouped_oof_with_inner_stopping_validation",
                "participants": sorted(set(selection_groups)),
                "test_participants": manifest["participants"]["test"],
                "fit_participants": sorted(set(selection_groups)),
            },
        },
    }
    artifact_path = output / "run.json"
    save_evaluation_result(artifact, artifact_path)
    report_path = output / "report.html"
    report_path.write_text(render_html_report(artifact), encoding="utf-8")
    return LstmRun(artifact_path, report_path, model_path, audit_path, artifact)


def load_lstm(path: str | Path) -> WiFallLstmPipeline:
    """Load a persisted causal LSTM feature and prediction pipeline."""

    pipeline = joblib.load(Path(path).expanduser())
    if not isinstance(pipeline, WiFallLstmPipeline):
        raise TypeError("Model artifact is not a WiFallLstmPipeline")
    return pipeline
