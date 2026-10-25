"""Participant-held-out causal LSTM pipeline over raw CSI windows."""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
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
from .causal_windows import (
    CausalWindowConfig,
    extract_causal_windows,
    load_causal_window_dataset,
)
from .diagnostics import recording_observations
from .loader import DatasetFormatError, Recording, load_split_manifest
from .training import load_split_records, validate_participant_splits

LABELS = ("non-fall", "fall")


@dataclass(frozen=True)
class LstmFeatureConfig:
    """Consecutive raw-row window used by causal LSTM inference."""

    name: str = "raw_csi_causal_window_v1"
    signal: str = "per-window normalized raw CSI amplitude"
    window_samples: int = 32
    window_stride: int = 4
    window_seconds: float | None = None
    subcarrier_count: int = 52

    def __post_init__(self) -> None:
        if self.window_samples < 2:
            raise ValueError("window_samples must be at least 2")
        if self.window_stride < 1:
            raise ValueError("window_stride must be positive")
        if self.window_seconds is not None and self.window_seconds <= 0.0:
            raise ValueError("window_seconds must be positive when provided")
        if self.subcarrier_count != 52:
            raise ValueError("WiFall LSTM currently requires 52 subcarriers")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "signal": self.signal,
            "window_samples": self.window_samples,
            "window_stride": self.window_stride,
            "window_seconds": self.window_seconds,
            "subcarrier_count": self.subcarrier_count,
            "shape": [self.window_samples, self.subcarrier_count],
            "sequence_semantics": "consecutive current-and-past raw CSI rows",
            "normalization_semantics": "median/MAD over current window only",
        }

    def window_config(self) -> CausalWindowConfig:
        return CausalWindowConfig(
            window_samples=self.window_samples,
            stride_samples=self.window_stride,
            window_seconds=self.window_seconds,
            pad_short_recordings=True,
        )


@dataclass(frozen=True)
class LstmConfig:
    """Configuration for grouped causal architecture and threshold selection."""

    run_id: str = "wifall-lstm-causal-windows"
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
    artifact_path: Path
    report_path: Path
    model_path: Path
    audit_path: Path
    artifact: dict[str, Any]


class SmallLstm(nn.Module):
    """Small unidirectional sequence classifier."""

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


def _normalize_windows(windows: np.ndarray) -> np.ndarray:
    center = np.median(windows, axis=1, keepdims=True)
    scale = np.median(np.abs(windows - center), axis=1, keepdims=True)
    scale[scale < 1e-8] = 1.0
    return ((windows - center) / scale).astype(np.float32, copy=False)


def _recording_windows(recording: Recording, config: LstmFeatureConfig) -> tuple[np.ndarray, np.ndarray]:
    windows, ends = extract_causal_windows(recording, config.window_config())
    return _normalize_windows(windows), ends


@dataclass
class WiFallLstmPipeline:
    """Persistable stateless rolling-window LSTM pipeline."""

    network: SmallLstm
    feature_config: LstmFeatureConfig
    threshold: float
    positive_label: str = "fall"
    negative_label: str = "non-fall"

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray((self.negative_label, self.positive_label))

    def features(self, recordings: list[Recording]) -> list[np.ndarray]:
        return [_recording_windows(recording, self.feature_config)[0] for recording in recordings]

    def causal_scores(self, windows: np.ndarray) -> np.ndarray:
        values = np.asarray(windows, dtype=np.float32)
        expected = (self.feature_config.window_samples, self.feature_config.subcarrier_count)
        if values.ndim != 3 or values.shape[1:] != expected:
            raise ValueError(f"LSTM windows must have shape (windows, {expected[0]}, {expected[1]})")
        return _scores(self.network, values)

    def decision_function(self, features: Sequence[np.ndarray]) -> np.ndarray:
        return np.asarray([float(np.max(self.causal_scores(windows))) for windows in features])

    def predict_proba(self, features: Sequence[np.ndarray]) -> np.ndarray:
        scores = self.decision_function(features)
        return np.column_stack((1.0 - scores, scores))

    def predict(self, features: Sequence[np.ndarray]) -> np.ndarray:
        scores = self.decision_function(features)
        return np.where(scores >= self.threshold, self.positive_label, self.negative_label)

    def predict_recordings(self, recordings: list[Recording]) -> np.ndarray:
        return self.predict(self.features(recordings))

    def decision_function_recordings(self, recordings: list[Recording]) -> np.ndarray:
        return self.decision_function(self.features(recordings))


def _seed(random_state: int) -> None:
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _window_dataset(
    recording_windows: list[np.ndarray],
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not recording_windows:
        raise DatasetFormatError("Cannot train LSTM on empty causal windows")
    windows = np.concatenate(recording_windows)
    window_labels = np.concatenate(
        [np.full(values.shape[0], label, dtype=object) for values, label in zip(recording_windows, labels)]
    )
    return windows, window_labels


def _loss(network: SmallLstm, values: np.ndarray, labels: np.ndarray, loss_function: nn.Module) -> float:
    network.eval()
    with torch.no_grad():
        logits = network(torch.from_numpy(np.asarray(values, dtype=np.float32)))
        targets = torch.from_numpy((labels == "fall").astype(np.float32))
        return float(loss_function(logits, targets).item())


def _fit_network(
    windows: np.ndarray,
    labels: np.ndarray,
    config: LstmConfig,
    hidden_size: int,
    layer_count: int,
    dropout: float,
    optimizer_name: str,
    *,
    validation_windows: np.ndarray | None = None,
    validation_labels: np.ndarray | None = None,
    epochs_override: int | None = None,
) -> tuple[SmallLstm, int]:
    if np.unique(labels).size < 2:
        raise DatasetFormatError("LSTM training requires both fall and non-fall classes")
    _seed(config.random_state)
    network = SmallLstm(config.feature.subcarrier_count, hidden_size, layer_count, dropout)
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(windows, dtype=np.float32)),
        torch.from_numpy((labels == "fall").astype(np.float32)),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.random_state),
    )
    optimizer_type = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[optimizer_name]
    optimizer = optimizer_type(network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.positive_weight, dtype=torch.float32))
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
        )
        if validation_windows is not None
        else None
    )
    epochs = epochs_override or config.epochs
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = epochs
    stale_epochs = 0
    for epoch in range(epochs):
        network.train()
        for batch_windows, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(network(batch_windows), batch_labels)
            loss.backward()
            optimizer.step()
        if validation_windows is None:
            continue
        validation_loss = _loss(network, validation_windows, validation_labels, loss_function)
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


def _scores(network: SmallLstm, windows: np.ndarray) -> np.ndarray:
    network.eval()
    with torch.no_grad():
        return torch.sigmoid(network(torch.from_numpy(np.asarray(windows, dtype=np.float32)))).cpu().numpy()


def _recording_scores(network: SmallLstm, recording_windows: list[np.ndarray]) -> np.ndarray:
    return np.asarray([float(np.max(_scores(network, windows))) for windows in recording_windows])


def _threshold_candidates(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(scores)
    if unique.size == 0:
        raise DatasetFormatError("Cannot select an LSTM threshold from empty scores")
    candidates = {0.0, 0.5, 1.0, *unique.tolist()}
    candidates.update(float((left + right) / 2.0) for left, right in pairwise(unique))
    return np.asarray(sorted(candidates), dtype=float)


def _specificity(labels: np.ndarray, predictions: np.ndarray) -> float:
    negatives = labels == "non-fall"
    return float(np.mean(predictions[negatives] == "non-fall")) if negatives.any() else 0.0


def _threshold_metrics(labels: np.ndarray, scores: np.ndarray, groups: list[str], threshold: float) -> dict[str, Any]:
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
    precision = float(np.sum((labels == "fall") & (predictions == "fall")) / max(np.sum(predictions == "fall"), 1))
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


def select_lstm_threshold(labels: np.ndarray, scores: np.ndarray, groups: list[str], minimum_recall: float = 0.95) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
            candidate["eligible"], candidate["minimum_group_recall"], candidate["group_specificity"], candidate["f1"], -candidate["threshold"],
        ),
    )
    return selected, candidates


def _grouped_oof_scores(
    windows_by_recording: list[np.ndarray],
    labels: np.ndarray,
    groups: list[str],
    config: LstmConfig,
    hidden_size: int,
    layer_count: int,
    dropout: float,
    optimizer_name: str,
) -> tuple[np.ndarray, list[int]]:
    scores = np.empty(labels.size, dtype=float)
    stopping_epochs: list[int] = []
    for held_out_group in sorted(set(groups)):
        validation_indexes = np.asarray([i for i, group in enumerate(groups) if group == held_out_group])
        train_indexes = np.asarray([i for i, group in enumerate(groups) if group != held_out_group])
        train_groups = sorted({groups[i] for i in train_indexes})
        if len(train_groups) > 1:
            inner_group = train_groups[-1]
            inner_indexes = np.asarray([i for i in train_indexes if groups[i] == inner_group])
            fit_indexes = np.asarray([i for i in train_indexes if groups[i] != inner_group])
        else:
            inner_indexes = None
            fit_indexes = train_indexes
        fit_windows, fit_labels = _window_dataset([windows_by_recording[i] for i in fit_indexes], labels[fit_indexes])
        validation_windows = validation_labels = None
        if inner_indexes is not None:
            validation_windows, validation_labels = _window_dataset(
                [windows_by_recording[i] for i in inner_indexes], labels[inner_indexes]
            )
        network, best_epoch = _fit_network(
            fit_windows,
            fit_labels,
            config,
            hidden_size,
            layer_count,
            dropout,
            optimizer_name,
            validation_windows=validation_windows,
            validation_labels=validation_labels,
        )
        stopping_epochs.append(best_epoch)
        scores[validation_indexes] = _recording_scores(network, [windows_by_recording[i] for i in validation_indexes])
    return scores, stopping_epochs


def _alerting_diagnostics(
    pipeline: WiFallLstmPipeline,
    recordings: list[tuple[str, str, Recording]],
) -> dict[str, Any]:
    observation_alerts = []
    for participant, _, recording in recordings:
        windows, ends = _recording_windows(recording, pipeline.feature_config)
        scores = pipeline.causal_scores(windows)
        indexes = np.flatnonzero(scores >= pipeline.threshold)
        first = int(indexes[0]) if indexes.size else None
        first_sequence = int(ends[first]) if first is not None else None
        duration = (datetime.fromisoformat(recording.timestamps[-1]) - datetime.fromisoformat(recording.timestamps[0])).total_seconds()
        first_seconds = duration * first_sequence / max(recording.row_count - 1, 1) if first_sequence is not None else None
        observation_alerts.append({
            "observation_id": recording.reference.recording_id,
            "group_id": participant,
            "label": recording.target,
            "first_alert_sequence": first_sequence,
            "first_alert_seconds": first_seconds,
        })
    falls = [x for x in observation_alerts if x["label"] == "fall"]
    non_falls = [x for x in observation_alerts if x["label"] == "non-fall"]
    alerted_falls = [x for x in falls if x["first_alert_sequence"] is not None]
    false_alerts = [x for x in non_falls if x["first_alert_sequence"] is not None]
    seconds = [x["first_alert_seconds"] for x in alerted_falls]
    return {
        "causal": True,
        "max_lookback_samples": pipeline.feature_config.window_samples,
        "alert_cadence_samples": pipeline.feature_config.window_stride,
        "state_reset": "clear rolling window at recording or stream session boundary",
        "weak_recording_labels": True,
        "observation_alerts": observation_alerts,
        "summary": {
            "fall_alert_rate": len(alerted_falls) / max(len(falls), 1),
            "false_alarm_rate": len(false_alerts) / max(len(non_falls), 1),
            "median_first_alert_seconds": float(np.median(seconds)) if seconds else None,
        },
    }


def train_lstm(
    dataset_root: str | Path,
    split_manifest: str | Path,
    output_dir: str | Path,
    config: LstmConfig | None = None,
    *,
    causal_window_dataset: str | Path | None = None,
) -> LstmRun:
    """Train, select, evaluate, and persist a causal LSTM from raw CSI."""

    config = config or LstmConfig()
    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_split_manifest(split_manifest)
    split_records = load_split_records(root, manifest)
    validate_participant_splits(split_records)
    if not split_records["train"] or not split_records["validation"]:
        raise DatasetFormatError("LSTM requires train and validation recordings")
    audit = audit_wifall(root)
    audit_path = output / "dataset_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    persisted_windows: dict[str, np.ndarray] = {}
    persisted_window_path = None
    if causal_window_dataset is not None:
        persisted = load_causal_window_dataset(causal_window_dataset)
        persisted_config = persisted.metadata["config"]
        persisted_seconds = persisted_config.get("window_seconds")
        if (
            int(persisted_config["window_samples"]) != config.feature.window_samples
            or int(persisted_config["stride_samples"]) != config.feature.window_stride
            or (float(persisted_seconds) if persisted_seconds is not None else None) != config.feature.window_seconds
        ):
            raise DatasetFormatError("Persisted causal window configuration does not match LSTM feature configuration")
        for recording_id in np.unique(persisted.recording_ids):
            persisted_windows[str(recording_id)] = _normalize_windows(
                persisted.amplitude[persisted.recording_ids == recording_id]
            )
        persisted_window_path = str(Path(causal_window_dataset).expanduser())

    def windows_for(record: Recording) -> np.ndarray:
        if persisted_windows:
            try:
                return persisted_windows[record.reference.recording_id]
            except KeyError as error:
                raise DatasetFormatError(f"Persisted causal windows missing recording: {record.reference.recording_id}") from error
        return _recording_windows(record, config.feature)[0]

    selection_records = split_records["train"] + split_records["validation"]
    selection_windows = [windows_for(record) for _, _, record in selection_records]
    selection_labels = np.asarray([record.target for _, _, record in selection_records], dtype=object)
    selection_groups = [participant for participant, _, _ in selection_records]
    candidates: list[dict[str, Any]] = []
    for hidden_size in config.hidden_sizes:
        for layer_count in config.layer_counts:
            for dropout in config.dropout_values:
                for optimizer_name in config.optimizer_names:
                    scores, stopping_epochs = _grouped_oof_scores(
                        selection_windows, selection_labels, selection_groups, config, hidden_size, layer_count, dropout, optimizer_name
                    )
                    selected_threshold, threshold_table = select_lstm_threshold(scores= scores, labels=selection_labels, groups=selection_groups, minimum_recall=config.minimum_recall)
                    candidates.append({
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
                    })
    selected = max(candidates, key=lambda c: (c["threshold_metrics"]["eligible"], c["threshold_metrics"]["minimum_group_recall"], c["threshold_metrics"]["group_specificity"], c["threshold_metrics"]["f1"], -c["threshold"]))
    final_windows, final_labels = _window_dataset(selection_windows, selection_labels)
    network, _ = _fit_network(final_windows, final_labels, config, selected["hidden_size"], selected["layer_count"], selected["dropout"], selected["optimizer"], epochs_override=selected["effective_epochs"])
    pipeline = WiFallLstmPipeline(network, config.feature, selected["threshold"])
    model_path = output / "model.joblib"
    joblib.dump(pipeline, model_path)

    model_metadata = {
        "family": "lstm",
        "name": "small_causal_lstm_raw_windows",
        "training_code": "wifall.lstm",
        "artifact": model_path.name,
        "architecture": {"type": "unidirectional_lstm_rolling_window_classifier", "input_size": config.feature.subcarrier_count, "hidden_size": selected["hidden_size"], "layer_count": selected["layer_count"], "dropout": selected["dropout"], "bidirectional": False},
        "parameters": {"epochs": config.epochs, "effective_epochs": selected["effective_epochs"], "optimizer": selected["optimizer"], "batch_size": config.batch_size, "learning_rate": config.learning_rate, "weight_decay": config.weight_decay, "positive_weight": config.positive_weight, "random_state": config.random_state},
        "real_time": {"causal": True, "max_lookback_samples": config.feature.window_samples, "alert_cadence_samples": config.feature.window_stride, "state_reset": "clear rolling window at recording or stream session boundary", "input_contract": "raw timestamped CSI rows; no full-recording resampling"},
        "decision_threshold": selected["threshold"],
        "threshold_policy": "minimum_grouped_validation_recall_then_specificity",
        "minimum_recall": config.minimum_recall,
        "fit_splits": ["train", "validation"],
        "fit_participants": sorted(set(selection_groups)),
        "score_orientation": {"positive_label": "fall", "score": "fall_probability"},
    }
    training_metadata = {"early_stopping": {"enabled": True, "patience": config.early_stopping_patience, "selection": "inner_grouped_non_test_validation", "effective_epochs": selected["effective_epochs"]}, "scheduler": {"method": "ReduceLROnPlateau", "patience": config.scheduler_patience, "factor": config.scheduler_factor}, "window_training": {"weak_recording_labels": True, "window_stride": config.feature.window_stride}}
    feature_metadata = config.feature.as_dict()
    preprocessing_metadata = {"steps": ["raw chronological CSI rows", "per-window median/MAD amplitude normalization", "causal rolling-window inference", "participant-grouped non-test selection"], "real_time_contract": model_metadata["real_time"], "labeling": {"weak_recording_labels": True, "event_timestamps": False}}
    split_metadata = {"strategy": "participant-held-out", "config": manifest["split_config"], "participants": manifest["participants"], "counts": manifest["counts"], "selection_participants": sorted(set(selection_groups)), "test_participants": manifest["participants"]["test"]}
    metadata = {"run_id": config.run_id, "dataset": manifest["dataset"], "task": {"name": "binary_fall_detection", "signal": "WiFi CSI"}, "model": model_metadata, "features": feature_metadata, "preprocessing": preprocessing_metadata, "split": split_metadata, "data_source": {"type": "raw_timestamped_csi", "causal_window_dataset": persisted_window_path}}

    evaluations: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    for split, records in split_records.items():
        if not records:
            continue
        record_windows = [windows_for(record) for _, _, record in records]
        labels = np.asarray([record.target for _, _, record in records], dtype=object)
        ids = [record.reference.recording_id for _, _, record in records]
        groups = [participant for participant, _, _ in records]
        evaluation_result = evaluate_classifier(pipeline, record_windows, labels, label_order=LABELS, positive_label="fall", run_id=config.run_id, split_name=split, dataset_metadata=manifest["dataset"], task_metadata=metadata["task"], model_metadata=model_metadata, feature_metadata=feature_metadata, preprocessing_metadata=preprocessing_metadata, split_metadata=split_metadata, observation_ids=ids, group_ids=groups)
        evaluation = evaluation_result["evaluations"][split]
        evaluation["fit_status"] = "fit_data" if split in {"train", "validation"} else "held_out"
        evaluation["threshold_metrics"] = selected["threshold_table"] if split == "validation" else []
        evaluation["alerting"] = _alerting_diagnostics(pipeline, [(participant, activity, record) for participant, activity, record in records])
        evaluation["hard_example_analysis"] = analyze_hard_examples(evaluation, recording_observations(records), run_id=config.run_id, split_name=split, threshold=selected["threshold"])
        evaluations[split] = evaluation
        artifacts[split] = evaluation_result
    primary_split = "test" if "test" in evaluations else "validation"
    primary = evaluations[primary_split]
    artifact = artifacts[primary_split] | {"evaluation_split": primary_split, "data_source": metadata["data_source"], "training": training_metadata, "evaluations": evaluations, "metrics": primary["metrics"], "confusion_matrix": primary["confusion_matrix"], "data_audit": audit, "selection": {"candidates": candidates, "selected": selected, "selection_split": "grouped_validation", "protocol": {"method": "participant_grouped_oof_with_inner_stopping_validation", "participants": sorted(set(selection_groups)), "test_participants": manifest["participants"]["test"], "fit_participants": sorted(set(selection_groups))}}}
    artifact_path = output / "run.json"
    save_evaluation_result(artifact, artifact_path)
    report_path = output / "report.html"
    report_path.write_text(render_html_report(artifact), encoding="utf-8")
    return LstmRun(artifact_path, report_path, model_path, audit_path, artifact)


def load_lstm(path: str | Path) -> WiFallLstmPipeline:
    pipeline = joblib.load(Path(path).expanduser())
    if not isinstance(pipeline, WiFallLstmPipeline):
        raise TypeError("Model artifact is not a WiFallLstmPipeline")
    return pipeline
