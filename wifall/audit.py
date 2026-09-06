"""Audit WiFall recording structure before improved model training."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .loader import DatasetFormatError, discover_recordings, load_recording


def _duration_seconds(timestamps: tuple[str, ...]) -> float | None:
    try:
        start = datetime.fromisoformat(timestamps[0])
        end = datetime.fromisoformat(timestamps[-1])
    except (IndexError, ValueError):
        return None
    return (end - start).total_seconds()


def audit_wifall(dataset_root: str | Path) -> dict[str, Any]:
    """Validate all recordings and return a JSON-serializable audit summary."""

    references = discover_recordings(dataset_root)
    activity_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    participant_counts: dict[str, Counter[str]] = defaultdict(Counter)
    durations: list[float] = []
    row_counts: list[int] = []
    malformed: list[dict[str, str]] = []
    empty: list[str] = []
    label_mismatches: list[dict[str, str]] = []

    for reference in references:
        try:
            recording = load_recording(reference)
        except DatasetFormatError as error:
            message = str(error)
            entry = {"recording_id": reference.recording_id, "error": message}
            if "no nonblank data rows" in message:
                empty.append(reference.recording_id)
            else:
                malformed.append(entry)
            if "does not match directory" in message:
                label_mismatches.append(entry)
            continue

        activity_counts[recording.reference.activity] += 1
        target_counts[recording.target] += 1
        participant_counts[recording.reference.participant_id][recording.target] += 1
        row_counts.append(recording.row_count)
        duration = _duration_seconds(recording.timestamps)
        if duration is not None:
            durations.append(duration)

    def distribution(values: list[float] | list[int]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "min": None, "max": None, "mean": None}
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    participants = {
        participant: dict(sorted(counts.items()))
        for participant, counts in sorted(participant_counts.items())
    }
    return {
        "dataset": {"name": "WiFall", "root_name": Path(dataset_root).name},
        "discovered_recordings": len(references),
        "valid_recordings": sum(activity_counts.values()),
        "activity_counts": dict(sorted(activity_counts.items())),
        "target_counts": dict(sorted(target_counts.items())),
        "participant_target_counts": participants,
        "duration_seconds": distribution(durations),
        "row_counts": distribution(row_counts),
        "malformed_recordings": malformed,
        "empty_recordings": empty,
        "label_mismatches": label_mismatches,
    }
