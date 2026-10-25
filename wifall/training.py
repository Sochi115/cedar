"""Shared participant-held-out training data utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .loader import DatasetFormatError, Recording, discover_recordings, load_recording

SPLITS = ("train", "validation", "test")


def load_split_records(
    dataset_root: Path,
    manifest: dict[str, Any],
) -> dict[str, list[tuple[str, str, Recording]]]:
    """Load raw recordings by participant-held-out split."""

    references = {reference.recording_id: reference for reference in discover_recordings(dataset_root)}
    split_records: dict[str, list[tuple[str, str, Recording]]] = {split: [] for split in SPLITS}
    for entry in manifest["recordings"]:
        recording_id = entry["recording_id"]
        split = entry["split"]
        if split not in split_records:
            raise DatasetFormatError(f"Unknown split in manifest: {split!r}")
        reference = references.get(recording_id)
        if reference is None:
            raise DatasetFormatError(f"Manifest recording is missing from dataset: {recording_id}")
        recording = load_recording(reference)
        if recording.target != entry["target"] or recording.row_count != entry["row_count"]:
            raise DatasetFormatError(f"Manifest metadata mismatch: {recording_id}")
        split_records[split].append((reference.participant_id, reference.activity, recording))
    return split_records


def validate_participant_splits(
    split_records: dict[str, list[tuple[str, str, Recording]]]
) -> None:
    """Reject manifests that place one participant in multiple splits."""

    participants = {
        split: {participant for participant, _, _ in records}
        for split, records in split_records.items()
    }
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            if participants[left].intersection(participants[right]):
                raise DatasetFormatError("Participant appears in multiple evaluation splits")
