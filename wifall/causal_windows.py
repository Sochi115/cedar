"""Causal consecutive-window views of raw WiFall recordings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .loader import DatasetFormatError, load_split_manifest
from .training import load_split_records, validate_participant_splits

_SPLITS = ("train", "validation", "test")
_SUBCARRIER_COUNT = 52


@dataclass(frozen=True)
class CausalWindowConfig:
    """Consecutive raw-row windows used by causal models."""

    window_samples: int = 32
    stride_samples: int = 4
    window_seconds: float | None = None
    pad_short_recordings: bool = True

    def __post_init__(self) -> None:
        if self.window_samples < 2:
            raise ValueError("window_samples must be at least 2")
        if self.stride_samples < 1:
            raise ValueError("stride_samples must be positive")
        if self.window_seconds is not None and self.window_seconds <= 0.0:
            raise ValueError("window_seconds must be positive when provided")

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_samples": self.window_samples,
            "stride_samples": self.stride_samples,
            "window_seconds": self.window_seconds,
            "pad_short_recordings": self.pad_short_recordings,
            "unit": "elapsed seconds plus raw CSI rows per model tensor" if self.window_seconds is not None else "raw CSI rows",
            "causal": True,
        }


@dataclass(frozen=True)
class CausalWindowRun:
    output_dir: Path
    data_path: Path
    metadata_path: Path
    config: CausalWindowConfig
    counts: dict[str, int]


@dataclass(frozen=True)
class CausalWindowDataset:
    amplitude: np.ndarray
    recording_ids: np.ndarray
    relative_paths: np.ndarray
    participants: np.ndarray
    activities: np.ndarray
    targets: np.ndarray
    splits: np.ndarray
    start_sequences: np.ndarray
    end_sequences: np.ndarray
    start_timestamps: np.ndarray
    end_timestamps: np.ndarray
    metadata: dict[str, Any]


def _fixed_row_ranges(row_count: int, config: CausalWindowConfig) -> list[tuple[int, int]]:
    if row_count < config.window_samples:
        return [(0, row_count - 1)] if config.pad_short_recordings else []
    starts = list(range(0, row_count - config.window_samples + 1, config.stride_samples))
    final_start = row_count - config.window_samples
    if starts[-1] != final_start:
        starts.append(final_start)
    return [(start, start + config.window_samples - 1) for start in starts]


def _window_ranges(recording: Any, config: CausalWindowConfig) -> list[tuple[int, int]]:
    row_count = recording.row_count
    if config.window_seconds is None:
        return _fixed_row_ranges(row_count, config)

    try:
        timestamps = [datetime.fromisoformat(value) for value in recording.timestamps]
    except (TypeError, ValueError):
        return _fixed_row_ranges(row_count, config)
    end_indexes = list(range(0, row_count, config.stride_samples))
    if end_indexes[-1] != row_count - 1:
        end_indexes.append(row_count - 1)
    ranges = []
    duration = timedelta(seconds=config.window_seconds)
    first = 0
    for end in end_indexes:
        earliest = timestamps[end] - duration
        while first < end and timestamps[first] < earliest:
            first += 1
        start = max(first, end - config.window_samples + 1)
        if end - start + 1 < config.window_samples and not config.pad_short_recordings:
            continue
        ranges.append((start, end))
    return ranges


def extract_causal_windows(
    recording: Any,
    config: CausalWindowConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract raw-amplitude windows and their observed ending sequence numbers."""

    config = config or CausalWindowConfig()
    if recording.csi.ndim != 2 or recording.csi.shape[1] != _SUBCARRIER_COUNT:
        raise DatasetFormatError("Causal windows require 52 CSI subcarriers")
    amplitude = np.abs(recording.csi).astype(np.float32)
    windows: list[np.ndarray] = []
    ends: list[int] = []
    for start, end in _window_ranges(recording, config):
        window = amplitude[start : end + 1]
        if window.shape[0] < config.window_samples:
            window = np.pad(
                window,
                ((0, config.window_samples - window.shape[0]), (0, 0)),
                mode="edge",
            )
        windows.append(window)
        ends.append(int(recording.sequences[end]))
    if not windows:
        return np.empty((0, config.window_samples, _SUBCARRIER_COUNT), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.stack(windows), np.asarray(ends, dtype=np.int64)


def build_causal_window_dataset(
    dataset_root: str | Path,
    split_manifest: str | Path,
    output_dir: str | Path,
    config: CausalWindowConfig | None = None,
) -> CausalWindowRun:
    """Persist chronological windows after participant split assignment."""

    config = config or CausalWindowConfig()
    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_split_manifest(split_manifest)
    split_records = load_split_records(root, manifest)
    validate_participant_splits(split_records)

    amplitude_windows: list[np.ndarray] = []
    recording_ids: list[str] = []
    relative_paths: list[str] = []
    participants: list[str] = []
    activities: list[str] = []
    targets: list[str] = []
    splits: list[str] = []
    start_sequences: list[int] = []
    end_sequences: list[int] = []
    start_timestamps: list[str] = []
    end_timestamps: list[str] = []

    for split in _SPLITS:
        for participant, activity, recording in split_records[split]:
            if recording.csi.ndim != 2 or recording.csi.shape[1] != _SUBCARRIER_COUNT:
                raise DatasetFormatError("Causal windows require 52 CSI subcarriers")
            amplitude = np.abs(recording.csi).astype(np.float32)
            for start, end in _window_ranges(recording, config):
                window = amplitude[start : end + 1]
                if window.shape[0] < config.window_samples:
                    if not config.pad_short_recordings:
                        continue
                    window = np.pad(
                        window,
                        ((0, config.window_samples - window.shape[0]), (0, 0)),
                        mode="edge",
                    )
                amplitude_windows.append(window)
                recording_ids.append(recording.reference.recording_id)
                relative_paths.append(recording.reference.relative_path)
                participants.append(participant)
                activities.append(activity)
                targets.append(recording.target)
                splits.append(split)
                start_sequences.append(int(recording.sequences[start]))
                end_sequences.append(int(recording.sequences[end]))
                start_timestamps.append(recording.timestamps[start])
                end_timestamps.append(recording.timestamps[end])

    if not amplitude_windows:
        raise DatasetFormatError("Cannot create causal windows from an empty dataset")
    amplitude_array = np.stack(amplitude_windows)
    data_path = output / "data.npz"
    np.savez_compressed(
        data_path,
        amplitude=amplitude_array,
        recording_ids=np.asarray(recording_ids),
        relative_paths=np.asarray(relative_paths),
        participants=np.asarray(participants),
        activities=np.asarray(activities),
        targets=np.asarray(targets),
        splits=np.asarray(splits),
        start_sequences=np.asarray(start_sequences, dtype=np.int64),
        end_sequences=np.asarray(end_sequences, dtype=np.int64),
        start_timestamps=np.asarray(start_timestamps),
        end_timestamps=np.asarray(end_timestamps),
    )
    counts = {split: int(np.sum(np.asarray(splits) == split)) for split in _SPLITS}
    metadata = {
        "schema_version": 1,
        "config": config.as_dict(),
        "shape": list(amplitude_array.shape),
        "source": {
            "type": "raw_timestamped_csi",
            "dataset": manifest["dataset"],
            "manifest_split_config": manifest["split_config"],
        },
        "counts": counts,
        "windowing": {
            "causal": True,
            "boundary": "recording",
            "timestamp_policy": "elapsed-duration lookback when timestamps parse; fixed-row fallback otherwise",
        },
        "labeling": {
            "weak_recording_labels": True,
            "warning": "WiFall activity labels do not mark exact fall onset or recovery",
        },
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return CausalWindowRun(output, data_path, metadata_path, config, counts)


def load_causal_window_dataset(path: str | Path) -> CausalWindowDataset:
    """Load persisted causal windows and validate their fixed shape."""

    output = Path(path).expanduser()
    data_path = output / "data.npz" if output.is_dir() else output
    metadata_path = data_path.with_name("metadata.json")
    if not data_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Causal window dataset requires {data_path} and {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config_data = metadata["config"]
    config = CausalWindowConfig(
        window_samples=int(config_data["window_samples"]),
        stride_samples=int(config_data["stride_samples"]),
        window_seconds=(float(config_data["window_seconds"]) if config_data.get("window_seconds") is not None else None),
        pad_short_recordings=bool(config_data["pad_short_recordings"]),
    )
    with np.load(data_path, allow_pickle=False) as arrays:
        amplitude = np.asarray(arrays["amplitude"], dtype=np.float32)
        values = {
            key: np.asarray(arrays[key]).astype(str)
            for key in (
                "recording_ids",
                "relative_paths",
                "participants",
                "activities",
                "targets",
                "splits",
                "start_timestamps",
                "end_timestamps",
            )
        }
        start_sequences = np.asarray(arrays["start_sequences"], dtype=np.int64)
        end_sequences = np.asarray(arrays["end_sequences"], dtype=np.int64)
    expected = (len(values["recording_ids"]), config.window_samples, _SUBCARRIER_COUNT)
    if amplitude.shape != expected:
        raise DatasetFormatError(f"Causal window amplitude shape {amplitude.shape} does not match {expected}")
    for name, value in values.items():
        if value.shape != (expected[0],):
            raise DatasetFormatError(f"Causal window metadata {name} has invalid shape {value.shape}")
    if start_sequences.shape != (expected[0],) or end_sequences.shape != (expected[0],):
        raise DatasetFormatError("Causal window sequence metadata has invalid shape")
    return CausalWindowDataset(
        amplitude=amplitude,
        recording_ids=values["recording_ids"],
        relative_paths=values["relative_paths"],
        participants=values["participants"],
        activities=values["activities"],
        targets=values["targets"],
        splits=values["splits"],
        start_sequences=start_sequences,
        end_sequences=end_sequences,
        start_timestamps=values["start_timestamps"],
        end_timestamps=values["end_timestamps"],
        metadata=metadata,
    )
