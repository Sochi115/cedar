from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from wifall import (
    CsiParseError,
    DatasetFormatError,
    SplitConfig,
    discover_recordings,
    generate_split_manifest,
    load_recording,
    load_split_manifest,
    parse_csi_field,
    save_split_manifest,
    summarize_dataset,
)

HEADER = ["type", "seq", "timestamp", "taget", "rssi", "data"]


def _csi_values(start: int = 1) -> list[int]:
    return list(range(start, start + 104))


def _write_recording(
    root: Path,
    participant: str,
    activity: str,
    recording_name: str,
    row_count: int = 2,
) -> Path:
    directory = root / participant / activity
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / recording_name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerow([])
        for index in range(row_count):
            writer.writerow(
                [
                    "CSI_DATA",
                    str(100 + index),
                    f"2023-01-01 00:00:0{index}.000",
                    activity,
                    "-31",
                    str(_csi_values(index)),
                ]
            )
    return path


def test_parse_csi_field_returns_52_complex_subcarriers() -> None:
    parsed = parse_csi_field(str(_csi_values()))

    assert parsed.shape == (52,)
    assert parsed.dtype == np.complex128
    assert parsed[0] == 1 + 2j
    assert parsed[-1] == 103 + 104j


@pytest.mark.parametrize(
    "value",
    ["", "not a list", str([1, 2, 3]), str([1.0] * 104), str([True] * 104)],
)
def test_parse_csi_field_rejects_malformed_values(value: str) -> None:
    with pytest.raises(CsiParseError):
        parse_csi_field(value)


def test_loader_skips_blank_rows_and_preserves_activity_label(tmp_path: Path) -> None:
    root = tmp_path / "WiFall"
    path = _write_recording(root, "ID0", "Jump", "jump.csv", row_count=2)
    reference = discover_recordings(root)[0]

    recording = load_recording(reference)

    assert path == reference.path
    assert recording.row_count == 2
    assert recording.reference.activity == "Jump"
    assert recording.target == "non-fall"
    repeated = load_recording(reference)
    assert recording.source_activities == ("Jump", "Jump")
    assert recording.csi.shape == (2, 52)
    assert recording.rssi.tolist() == [-31.0, -31.0]
    np.testing.assert_array_equal(recording.csi, repeated.csi)
    np.testing.assert_array_equal(recording.sequences, repeated.sequences)
    assert recording.timestamps == repeated.timestamps


def test_loader_rejects_malformed_recording_data(tmp_path: Path) -> None:
    root = tmp_path / "WiFall"
    path = _write_recording(root, "ID0", "fall", "fall.csv", row_count=1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerow(["CSI_DATA", "1", "2023-01-01 00:00:00.000", "fall", "-31", "[1, 2]"])

    with pytest.raises(CsiParseError):
        load_recording(discover_recordings(root)[0])


def test_loader_rejects_activity_label_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "WiFall"
    path = _write_recording(root, "ID0", "fall", "fall.csv", row_count=1)
    text = path.read_text(encoding="utf-8").replace(",fall,-31,", ",sit,-31,")
    path.write_text(text, encoding="utf-8")

    with pytest.raises(DatasetFormatError, match="does not match directory"):
        load_recording(discover_recordings(root)[0])


def test_summary_counts_recordings_rows_and_targets(tmp_path: Path) -> None:
    root = tmp_path / "WiFall"
    _write_recording(root, "ID0", "fall", "fall.csv", row_count=2)
    _write_recording(root, "ID0", "sit", "sit.csv", row_count=3)
    _write_recording(root, "ID1", "walk", "walk.csv", row_count=4)

    summary = summarize_dataset(root)

    assert summary.participant_count == 2
    assert summary.recording_count == 3
    assert summary.row_count == 9
    assert summary.activity_counts == {"fall": 1, "sit": 1, "walk": 1}
    assert summary.target_counts == {"fall": 1, "non-fall": 2}


def test_split_is_deterministic_and_has_no_participant_leakage(tmp_path: Path) -> None:
    root = tmp_path / "WiFall"
    for participant_number in range(5):
        participant = f"ID{participant_number}"
        _write_recording(root, participant, "fall", "fall.csv")
        _write_recording(root, participant, "sit", "sit.csv")

    config = SplitConfig(seed=7, test_participants=1, validation_participants=1)
    first = generate_split_manifest(root, config)
    second = generate_split_manifest(root, config)

    assert first == second
    participant_splits = first["participants"]
    assert set(participant_splits["train"]).isdisjoint(participant_splits["validation"])
    assert set(participant_splits["train"]).isdisjoint(participant_splits["test"])
    assert set(participant_splits["validation"]).isdisjoint(participant_splits["test"])

    assignments = {entry["participant_id"]: entry["split"] for entry in first["recordings"]}
    for split, participants in participant_splits.items():
        assert {assignments[participant] for participant in participants} == {split}


def test_split_manifest_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "WiFall"
    _write_recording(root, "ID0", "fall", "fall.csv")
    _write_recording(root, "ID1", "sit", "sit.csv")
    _write_recording(root, "ID2", "walk", "walk.csv")
    manifest = generate_split_manifest(
        root, SplitConfig(seed=1, test_participants=1, validation_participants=1)
    )

    output = save_split_manifest(manifest, tmp_path / "manifest.json")

    assert load_split_manifest(output) == manifest
