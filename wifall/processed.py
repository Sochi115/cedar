"""Persisted, duration-normalized CSI data for model training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .loader import (
    DatasetFormatError,
    Recording,
    RecordingReference,
    load_split_manifest,
)
from .training import load_split_records, validate_participant_splits

_SPLITS = ("train", "validation", "test")
_SUBCARRIER_COUNT = 52


@dataclass(frozen=True)
class DurationNormalizedConfig:
    """Fixed temporal grid used to normalize every recording by elapsed time."""

    sample_count: int = 128
    section_count: int = 8

    def __post_init__(self) -> None:
        if self.sample_count < 2:
            raise ValueError("sample_count must be at least 2")
        if self.section_count < 1:
            raise ValueError("section_count must be at least 1")
        if self.sample_count % self.section_count:
            raise ValueError("sample_count must be divisible by section_count")

    @property
    def samples_per_section(self) -> int:
        return self.sample_count // self.section_count

    def as_dict(self) -> dict[str, int]:
        return {
            "sample_count": self.sample_count,
            "section_count": self.section_count,
            "samples_per_section": self.samples_per_section,
        }


@dataclass(frozen=True)
class DurationNormalizedDatasetRun:
    """Persisted files and metadata produced by preprocessing."""

    data_path: Path
    metadata_path: Path
    config: DurationNormalizedConfig
    counts: dict[str, int]


@dataclass(frozen=True)
class DurationNormalizedDataset:
    """Loaded duration-normalized data and model-compatible recording views."""

    csi: np.ndarray
    rssi: np.ndarray
    config: DurationNormalizedConfig
    splits: dict[str, list[tuple[str, str, Recording]]]
    metadata: dict[str, Any]


def _elapsed_seconds(recording: Recording) -> np.ndarray:
    try:
        timestamps = [
            datetime.fromisoformat(value)
            for value in recording.timestamps
        ]
    except ValueError as error:
        raise DatasetFormatError(
            f"Recording has invalid timestamps: {recording.reference.recording_id}"
        ) from error
    first = timestamps[0]
    try:
        elapsed = np.asarray(
            [(timestamp - first).total_seconds() for timestamp in timestamps],
            dtype=np.float64,
        )
    except TypeError as error:
        raise DatasetFormatError(
            f"Recording mixes timezone-aware and naive timestamps: "
            f"{recording.reference.recording_id}"
        ) from error
    if not np.isfinite(elapsed).all() or np.any(np.diff(elapsed) < 0):
        raise DatasetFormatError(
            f"Recording timestamps must be finite and ordered: "
            f"{recording.reference.recording_id}"
        )
    if elapsed[-1] <= 0:
        raise DatasetFormatError(
            f"Recording duration must be positive: {recording.reference.recording_id}"
        )
    return elapsed


def _unique_time_samples(
    elapsed: np.ndarray, csi: np.ndarray, rssi: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique, inverse = np.unique(elapsed, return_inverse=True)
    if unique.size == elapsed.size:
        return elapsed, csi, rssi
    unique_csi = np.zeros((unique.size, csi.shape[1]), dtype=np.complex128)
    unique_rssi = np.zeros(unique.size, dtype=np.float64)
    for index in range(unique.size):
        rows = inverse == index
        unique_csi[index] = csi[rows].mean(axis=0)
        unique_rssi[index] = rssi[rows].mean()
    return unique, unique_csi, unique_rssi


def _interpolate_columns(
    source_positions: np.ndarray,
    values: np.ndarray,
    target_positions: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            np.interp(target_positions, source_positions, values[:, column])
            for column in range(values.shape[1])
        ]
    )


def duration_normalize_recording(
    recording: Recording,
    config: DurationNormalizedConfig | None = None,
) -> Recording:
    """Resample one recording onto an equal-elapsed-time grid."""

    config = config or DurationNormalizedConfig()
    if recording.csi.ndim != 2 or recording.csi.shape[1] != _SUBCARRIER_COUNT:
        raise DatasetFormatError("Duration normalization requires 52 CSI subcarriers")
    elapsed = _elapsed_seconds(recording)
    elapsed, csi, rssi = _unique_time_samples(
        elapsed,
        np.asarray(recording.csi, dtype=np.complex128),
        np.asarray(recording.rssi, dtype=np.float64),
    )
    target = np.linspace(0.0, elapsed[-1], config.sample_count)
    normalized_csi = _interpolate_columns(elapsed, csi.real, target)
    normalized_csi = normalized_csi + 1j * _interpolate_columns(
        elapsed, csi.imag, target
    )
    normalized_rssi = np.interp(target, elapsed, rssi)
    start = datetime.fromisoformat(recording.timestamps[0])
    normalized_timestamps = tuple(
        (start + timedelta(seconds=float(seconds))).isoformat()
        for seconds in target
    )
    return Recording(
        reference=recording.reference,
        csi=normalized_csi.astype(recording.csi.dtype, copy=False),
        rssi=normalized_rssi,
        timestamps=normalized_timestamps,
        sequences=np.arange(config.sample_count, dtype=np.int64),
        source_activities=(recording.reference.activity,) * config.sample_count,
    )


def build_duration_normalized_dataset(
    dataset_root: str | Path,
    split_manifest: str | Path,
    output_dir: str | Path,
    config: DurationNormalizedConfig | None = None,
) -> DurationNormalizedDatasetRun:
    """Create and persist one fixed-shape, duration-normalized dataset."""

    config = config or DurationNormalizedConfig()
    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_split_manifest(split_manifest)
    split_records = load_split_records(root, manifest)
    validate_participant_splits(split_records)

    entries: list[tuple[str, str, Recording, str]] = []
    for split in _SPLITS:
        entries.extend(
            (participant, activity, recording, split)
            for participant, activity, recording in split_records[split]
        )
    if not entries:
        raise DatasetFormatError("Cannot process an empty dataset")

    normalized = [duration_normalize_recording(recording, config) for _, _, recording, _ in entries]
    csi = np.stack(
        [
            recording.csi.reshape(
                config.section_count, config.samples_per_section, _SUBCARRIER_COUNT
            )
            for recording in normalized
        ]
    )
    rssi = np.stack([recording.rssi for recording in normalized])
    recording_ids = np.asarray([recording.reference.recording_id for *_, recording, _ in entries])
    relative_paths = np.asarray([recording.reference.relative_path for *_, recording, _ in entries])
    participants = np.asarray([participant for participant, _, _, _ in entries])
    activities = np.asarray([activity for _, activity, _, _ in entries])
    targets = np.asarray([recording.target for *_, recording, _ in entries])
    splits = np.asarray([split for *_, split in entries])
    durations = np.asarray([_elapsed_seconds(recording)[-1] for *_, recording, _ in entries])
    row_counts = np.asarray([recording.row_count for *_, recording, _ in entries], dtype=np.int64)

    data_path = output / "data.npz"
    np.savez_compressed(
        data_path,
        csi=csi,
        rssi=rssi,
        recording_ids=recording_ids,
        relative_paths=relative_paths,
        participants=participants,
        activities=activities,
        targets=targets,
        splits=splits,
        durations=durations,
        source_row_counts=row_counts,
    )
    counts = {split: int(np.sum(splits == split)) for split in _SPLITS}
    metadata = {
        "schema_version": 1,
        "config": config.as_dict(),
        "shape": list(csi.shape),
        "source": {
            "type": "participant-held-out",
            "dataset": manifest["dataset"],
            "manifest_split_config": manifest["split_config"],
        },
        "counts": counts,
        "duration_seconds": {
            "min": float(durations.min()),
            "median": float(np.median(durations)),
            "max": float(durations.max()),
        },
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return DurationNormalizedDatasetRun(data_path, metadata_path, config, counts)


def load_duration_normalized_dataset(
    path: str | Path,
) -> DurationNormalizedDataset:
    """Load persisted data and expose it as model-compatible recordings."""

    location = Path(path).expanduser()
    data_path = location / "data.npz" if location.is_dir() else location
    metadata_path = data_path.with_name("metadata.json")
    if not data_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Processed dataset requires {data_path} and {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config_data = metadata["config"]
    config = DurationNormalizedConfig(
        sample_count=int(config_data["sample_count"]),
        section_count=int(config_data["section_count"]),
    )
    with np.load(data_path, allow_pickle=False) as arrays:
        csi = np.asarray(arrays["csi"])
        rssi = np.asarray(arrays["rssi"])
        recording_ids = np.asarray(arrays["recording_ids"]).astype(str)
        relative_paths = np.asarray(arrays["relative_paths"]).astype(str)
        participants = np.asarray(arrays["participants"]).astype(str)
        activities = np.asarray(arrays["activities"]).astype(str)
        targets = np.asarray(arrays["targets"]).astype(str)
        splits = np.asarray(arrays["splits"]).astype(str)
        durations = np.asarray(arrays["durations"], dtype=np.float64)
    expected_shape = (len(recording_ids), config.section_count, config.samples_per_section, _SUBCARRIER_COUNT)
    if csi.shape != expected_shape:
        raise DatasetFormatError(f"Processed CSI shape {csi.shape} does not match {expected_shape}")
    if rssi.shape != (len(recording_ids), config.sample_count):
        raise DatasetFormatError("Processed RSSI shape does not match processed CSI shape")

    records: dict[str, list[tuple[str, str, Recording]]] = {split: [] for split in _SPLITS}
    base = datetime(2000, 1, 1, tzinfo=UTC)
    for index, split in enumerate(splits):
        if split not in records:
            raise DatasetFormatError(f"Unknown processed split: {split!r}")
        flat_csi = csi[index].reshape(config.sample_count, _SUBCARRIER_COUNT)
        elapsed = np.linspace(0.0, durations[index], config.sample_count)
        timestamps = tuple(
            (base + timedelta(seconds=float(seconds))).isoformat()
            for seconds in elapsed
        )
        reference = RecordingReference(
            recording_id=recording_ids[index],
            relative_path=relative_paths[index],
            participant_id=participants[index],
            activity=activities[index],
            path=Path(relative_paths[index]),
        )
        recording = Recording(
            reference=reference,
            csi=flat_csi,
            rssi=rssi[index],
            timestamps=timestamps,
            sequences=np.arange(config.sample_count, dtype=np.int64),
            source_activities=(activities[index],) * config.sample_count,
        )
        if recording.target != targets[index]:
            raise DatasetFormatError(f"Processed target mismatch: {recording_ids[index]}")
        records[split].append((participants[index], activities[index], recording))
    return DurationNormalizedDataset(csi, rssi, config, records, metadata)
