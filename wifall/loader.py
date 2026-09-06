"""Load raw WiFall recordings and create reproducible participant splits."""

from __future__ import annotations

import ast
import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


class DatasetFormatError(ValueError):
    """Raised when a WiFall recording does not match the expected format."""


class CsiParseError(DatasetFormatError):
    """Raised when a CSI field cannot be parsed into 52 complex values."""


@dataclass(frozen=True)
class RecordingReference:
    """Stable metadata identifying one source CSV recording."""

    recording_id: str
    relative_path: str
    participant_id: str
    activity: str
    path: Path

    @property
    def target(self) -> str:
        """Return the binary fall-detection target for this recording."""

        return "fall" if self.activity.casefold() == "fall" else "non-fall"


@dataclass(frozen=True)
class Recording:
    """One recording and its parsed CSI samples."""

    reference: RecordingReference
    csi: np.ndarray
    rssi: np.ndarray
    timestamps: tuple[str, ...]
    sequences: np.ndarray
    source_activities: tuple[str, ...]

    @property
    def target(self) -> str:
        return self.reference.target

    @property
    def row_count(self) -> int:
        return int(self.csi.shape[0])


@dataclass(frozen=True)
class RecordingSummary:
    """Metadata needed for splitting and downstream reproducibility."""

    recording_id: str
    relative_path: str
    participant_id: str
    activity: str
    target: str
    row_count: int
    first_timestamp: str
    last_timestamp: str


@dataclass(frozen=True)
class DatasetSummary:
    """Counts and labels observed while validating a WiFall dataset."""

    participant_count: int
    participants: tuple[str, ...]
    recording_count: int
    row_count: int
    activity_counts: Mapping[str, int]
    target_counts: Mapping[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "participant_count": self.participant_count,
            "participants": list(self.participants),
            "recording_count": self.recording_count,
            "row_count": self.row_count,
            "activity_counts": dict(self.activity_counts),
            "target_counts": dict(self.target_counts),
        }


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for deterministic participant-held-out splits."""

    seed: int = 42
    test_participants: int = 2
    validation_participants: int = 1

    def __post_init__(self) -> None:
        if self.test_participants < 1:
            raise ValueError("test_participants must be at least 1")
        if self.validation_participants < 0:
            raise ValueError("validation_participants cannot be negative")

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


# ``taget`` is the misspelled activity column in the published WiFall CSV files.
_SOURCE_ACTIVITY_COLUMN = "taget"
_REQUIRED_COLUMNS = frozenset({"data", _SOURCE_ACTIVITY_COLUMN, "timestamp", "seq"})
_SPLIT_NAMES = ("train", "validation", "test")


def parse_csi_field(value: str) -> np.ndarray:
    """Parse one WiFall ``data`` field into 52 complex CSI subcarriers.

    WiFall stores real and imaginary values alternately:
    ``[real_0, imag_0, real_1, imag_1, ...]``.
    """

    if not isinstance(value, str) or not value.strip():
        raise CsiParseError("CSI field must be a non-empty string")

    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise CsiParseError("CSI field is not a valid list") from exc

    if not isinstance(parsed, (list, tuple)) or len(parsed) != 104:
        actual_length = len(parsed) if isinstance(parsed, (list, tuple)) else "unknown"
        raise CsiParseError(f"CSI field must contain 104 values, got {actual_length}")

    if any(isinstance(item, bool) or not isinstance(item, int) for item in parsed):
        raise CsiParseError("CSI values must all be integers")

    values = np.asarray(parsed, dtype=np.float64)
    return values[::2] + 1j * values[1::2]


def discover_recordings(dataset_root: str | Path) -> tuple[RecordingReference, ...]:
    """Discover CSV recordings beneath ``ID*/<activity>/`` directories."""

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"WiFall dataset directory does not exist: {root}")

    references: list[RecordingReference] = []
    participant_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("ID")
    )
    if not participant_dirs:
        raise DatasetFormatError(f"No participant directories found beneath {root}")

    for participant_dir in participant_dirs:
        participant_id = participant_dir.name
        activity_dirs = sorted(path for path in participant_dir.iterdir() if path.is_dir())
        for activity_dir in activity_dirs:
            activity = activity_dir.name
            for path in sorted(activity_dir.glob("*.csv")):
                relative_path = path.relative_to(root).as_posix()
                references.append(
                    RecordingReference(
                        recording_id=relative_path,
                        relative_path=relative_path,
                        participant_id=participant_id,
                        activity=activity,
                        path=path,
                    )
                )

    if not references:
        raise DatasetFormatError(f"No CSV recordings found beneath {root}")
    return tuple(references)


def _nonblank_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DatasetFormatError(f"Recording has no CSV header: {path}")
        missing = _REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise DatasetFormatError(f"Recording {path} is missing columns: {missing_columns}")

        for row in reader:
            if not row or not any((value or "").strip() for value in row.values()):
                continue
            yield {key: value or "" for key, value in row.items()}


def _parse_sequence(value: str, path: Path) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise DatasetFormatError(f"Invalid sequence number in {path}: {value!r}") from exc


def _parse_rssi(value: str, path: Path) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise DatasetFormatError(f"Invalid RSSI value in {path}: {value!r}") from exc


def _read_rows(reference: RecordingReference, collect_csi: bool) -> tuple[Any, ...]:
    csi_rows: list[np.ndarray] = []
    rssi: list[float] = []
    timestamps: list[str] = []
    sequences: list[int] = []
    source_activities: list[str] = []

    for row in _nonblank_rows(reference.path):
        activity = row[_SOURCE_ACTIVITY_COLUMN].strip()
        if not activity:
            raise DatasetFormatError(f"Missing activity label in {reference.path}")
        if activity.casefold() != reference.activity.casefold():
            raise DatasetFormatError(
                f"Activity label {activity!r} does not match directory "
                f"{reference.activity!r} in {reference.path}"
            )

        csi = parse_csi_field(row["data"])
        sequences.append(_parse_sequence(row["seq"], reference.path))
        timestamps.append(row["timestamp"].strip())
        source_activities.append(activity)
        if "rssi" in row:
            rssi.append(_parse_rssi(row["rssi"], reference.path))
        if collect_csi:
            csi_rows.append(csi)

    if not sequences:
        raise DatasetFormatError(f"Recording contains no nonblank data rows: {reference.path}")

    if collect_csi:
        return (
            np.stack(csi_rows),
            np.asarray(rssi, dtype=np.float64),
            tuple(timestamps),
            np.asarray(sequences, dtype=np.int64),
            tuple(source_activities),
        )
    return (len(sequences), timestamps[0], timestamps[-1])


def load_recording(reference: RecordingReference) -> Recording:
    """Load and parse one recording into a recording-level example."""

    csi, rssi, timestamps, sequences, source_activities = _read_rows(
        reference, collect_csi=True
    )
    return Recording(
        reference=reference,
        csi=csi,
        rssi=rssi,
        timestamps=timestamps,
        sequences=sequences,
        source_activities=source_activities,
    )


def _summarize_recording(reference: RecordingReference) -> RecordingSummary:
    row_count, first_timestamp, last_timestamp = _read_rows(reference, collect_csi=False)
    return RecordingSummary(
        recording_id=reference.recording_id,
        relative_path=reference.relative_path,
        participant_id=reference.participant_id,
        activity=reference.activity,
        target=reference.target,
        row_count=row_count,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
    )


def summarize_dataset(dataset_root: str | Path) -> DatasetSummary:
    """Validate all recordings and return dataset-level counts."""

    references = discover_recordings(dataset_root)
    summaries = tuple(_summarize_recording(reference) for reference in references)
    participants = tuple(sorted({summary.participant_id for summary in summaries}))
    activity_counts = Counter(summary.activity for summary in summaries)
    target_counts = Counter(summary.target for summary in summaries)
    return DatasetSummary(
        participant_count=len(participants),
        participants=participants,
        recording_count=len(summaries),
        row_count=sum(summary.row_count for summary in summaries),
        activity_counts=dict(sorted(activity_counts.items())),
        target_counts=dict(sorted(target_counts.items())),
    )


def _split_participants(
    participants: tuple[str, ...], config: SplitConfig
) -> dict[str, tuple[str, ...]]:
    required = config.test_participants + config.validation_participants
    if required >= len(participants):
        raise ValueError(
            "test_participants + validation_participants must leave at least one "
            "training participant"
        )

    shuffled = list(participants)
    np.random.default_rng(config.seed).shuffle(shuffled)
    test = tuple(sorted(shuffled[: config.test_participants]))
    validation_start = config.test_participants
    validation_end = validation_start + config.validation_participants
    validation = tuple(sorted(shuffled[validation_start:validation_end]))
    held_out = set(test).union(validation)
    train = tuple(sorted(participant for participant in participants if participant not in held_out))
    return {"train": train, "validation": validation, "test": test}


def _counts_for_split(
    summaries: tuple[RecordingSummary, ...], assignments: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    by_split: dict[str, list[RecordingSummary]] = defaultdict(list)
    for summary in summaries:
        by_split[assignments[summary.recording_id]].append(summary)

    counts: dict[str, dict[str, Any]] = {}
    for split in _SPLIT_NAMES:
        split_summaries = by_split[split]
        counts[split] = {
            "recordings": len(split_summaries),
            "rows": sum(summary.row_count for summary in split_summaries),
            "targets": dict(
                sorted(Counter(summary.target for summary in split_summaries).items())
            ),
            "activities": dict(
                sorted(Counter(summary.activity for summary in split_summaries).items())
            ),
        }
    return counts


def generate_split_manifest(
    dataset_root: str | Path, config: SplitConfig | None = None
) -> dict[str, Any]:
    """Validate WiFall and create a JSON-serializable participant split manifest."""

    config = config or SplitConfig()
    root = Path(dataset_root).expanduser().resolve()
    references = discover_recordings(root)
    summaries = tuple(_summarize_recording(reference) for reference in references)
    participants = tuple(sorted({summary.participant_id for summary in summaries}))
    participant_splits = _split_participants(participants, config)
    participant_to_split = {
        participant: split
        for split, split_participants in participant_splits.items()
        for participant in split_participants
    }
    assignments = {
        summary.recording_id: participant_to_split[summary.participant_id]
        for summary in summaries
    }

    return {
        "schema_version": 1,
        "dataset": {
            "name": "WiFall",
            "root_name": root.name,
            "participants": len(participants),
            "recordings": len(summaries),
            "rows": sum(summary.row_count for summary in summaries),
        },
        "split_config": config.as_dict(),
        "participants": {
            split: list(split_participants)
            for split, split_participants in participant_splits.items()
        },
        "recordings": [
            {
                **asdict(summary),
                "split": assignments[summary.recording_id],
            }
            for summary in summaries
        ],
        "counts": _counts_for_split(summaries, assignments),
    }


def save_split_manifest(manifest: Mapping[str, Any], output_path: str | Path) -> Path:
    """Persist a split manifest as stable, human-readable JSON."""

    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return destination


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a persisted split manifest."""

    source = Path(path).expanduser()
    with source.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise DatasetFormatError("Unsupported split manifest schema version")
    for key in ("dataset", "split_config", "participants", "recordings", "counts"):
        if key not in manifest:
            raise DatasetFormatError(f"Split manifest is missing {key!r}")
    return manifest
