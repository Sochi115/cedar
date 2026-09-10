from __future__ import annotations

import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from wifall import (
    DurationNormalizedConfig,
    build_duration_normalized_dataset,
    duration_normalize_recording,
    generate_split_manifest,
    load_duration_normalized_dataset,
    save_split_manifest,
    train_improved_svm,
)
from wifall.loader import Recording, RecordingReference

HEADER = ["type", "seq", "timestamp", "taget", "rssi", "data"]


def _recording() -> Recording:
    timestamps = (
        "2023-01-01T00:00:00.000000",
        "2023-01-01T00:00:01.000000",
        "2023-01-01T00:00:03.000000",
    )
    values = np.asarray([0.0, 10.0, 30.0])
    csi = np.repeat(values[:, None], 52, axis=1).astype(np.complex128)
    reference = RecordingReference(
        recording_id="ID0/fall/recording.csv",
        relative_path="ID0/fall/recording.csv",
        participant_id="ID0",
        activity="fall",
        path=Path("recording.csv"),
    )
    return Recording(
        reference=reference,
        csi=csi,
        rssi=np.asarray([-40.0, -35.0, -30.0]),
        timestamps=timestamps,
        sequences=np.asarray([0, 1, 2]),
        source_activities=("fall",) * 3,
    )


def _write_recording(root: Path, participant: str, activity: str) -> None:
    directory = root / participant / activity
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{activity}.csv"
    values = list(range(104))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for row_number in range(8):
            timestamp = datetime(2023, 1, 1, tzinfo=UTC) + timedelta(seconds=row_number / 2)
            writer.writerow(
                [
                    "CSI_DATA",
                    str(row_number),
                    timestamp.isoformat(),
                    activity,
                    "-31",
                    str(values),
                ]
            )


def _fixture_dataset(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "WiFall"
    for participant_number in range(4):
        participant = f"ID{participant_number}"
        _write_recording(root, participant, "fall")
        _write_recording(root, participant, "sit")
    manifest = generate_split_manifest(root)
    manifest_path = tmp_path / "split.json"
    save_split_manifest(manifest, manifest_path)
    return root, manifest_path


def test_duration_normalization_uses_elapsed_time_not_row_positions() -> None:
    normalized = duration_normalize_recording(
        _recording(),
        DurationNormalizedConfig(sample_count=7, section_count=1),
    )

    # Elapsed-time targets are [0, .5, 1, 1.5, 2, 2.5, 3] seconds.
    np.testing.assert_allclose(normalized.csi[:, 0].real, [0, 5, 10, 15, 20, 25, 30])
    assert normalized.csi.shape == (7, 52)
    assert normalized.row_count == 7
    assert normalized.target == "fall"
    assert normalized.reference.recording_id == "ID0/fall/recording.csv"


def test_processed_dataset_persists_sectioned_csi_and_metadata(tmp_path: Path) -> None:
    root, manifest_path = _fixture_dataset(tmp_path)
    output = tmp_path / "duration-normalized"
    config = DurationNormalizedConfig(sample_count=8, section_count=4)

    run = build_duration_normalized_dataset(root, manifest_path, output, config)
    loaded = load_duration_normalized_dataset(output)

    assert run.data_path == output / "data.npz"
    assert run.metadata_path == output / "metadata.json"
    assert loaded.csi.shape == (8, 4, 2, 52)
    assert loaded.config == config
    assert set(loaded.splits) == {"train", "validation", "test"}
    assert sum(len(records) for records in loaded.splits.values()) == 8
    assert all(record[2].row_count == 8 for record in loaded.splits["train"])
    assert loaded.splits["test"][0][2].target in {"fall", "non-fall"}

    metadata = json.loads(run.metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["config"] == config.as_dict()
    assert metadata["shape"] == [8, 4, 2, 52]
    assert metadata["source"]["type"] == "participant-held-out"


def test_processed_dataset_can_train_improved_svm(tmp_path: Path) -> None:
    root, manifest_path = _fixture_dataset(tmp_path)
    processed = build_duration_normalized_dataset(
        root,
        manifest_path,
        tmp_path / "duration-normalized",
        DurationNormalizedConfig(sample_count=8, section_count=4),
    )

    run = train_improved_svm(
        root,
        manifest_path,
        tmp_path / "processed-svm",
        processed_dataset=processed.data_path,
    )

    assert run.artifact["data_source"]["type"] == "duration_normalized_dataset"
    assert run.artifact["data_source"]["path"].endswith("data.npz")


def test_duration_normalized_config_rejects_incompatible_shapes() -> None:
    try:
        DurationNormalizedConfig(sample_count=7, section_count=4)
    except ValueError as error:
        assert str(error) == "sample_count must be divisible by section_count"
    else:
        raise AssertionError("Expected incompatible section shape to be rejected")
