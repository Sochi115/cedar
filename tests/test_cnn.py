from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from evaluation import load_evaluation_result
from wifall import (
    CnnConfig,
    CnnFeatureConfig,
    SplitConfig,
    extract_cnn_tensor,
    generate_split_manifest,
    load_cnn,
    save_split_manifest,
    select_cnn_threshold,
    train_cnn,
)
from wifall.loader import discover_recordings, load_recording

HEADER = ["type", "seq", "timestamp", "taget", "rssi", "data"]


def _write_recording(root: Path, participant: str, activity: str) -> None:
    directory = root / participant / activity
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{activity}.csv"
    participant_number = int(participant.removeprefix("ID"))
    activity_offset = 1000 if activity == "fall" else 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for row_number in range(8):
            values = list(
                range(
                    participant_number * 10 + activity_offset + row_number,
                    participant_number * 10 + activity_offset + row_number + 104,
                )
            )
            writer.writerow(
                [
                    "CSI_DATA",
                    str(row_number),
                    f"2023-01-01 00:00:0{row_number}.000",
                    activity,
                    "-31",
                    str(values),
                ]
            )


def _fixture_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "WiFall"
    for participant_number in range(5):
        participant = f"ID{participant_number}"
        _write_recording(root, participant, "fall")
        _write_recording(root, participant, "sit")
    return root


def _fixture_manifest(root: Path, tmp_path: Path) -> Path:
    manifest = generate_split_manifest(
        root, SplitConfig(seed=7, test_participants=1, validation_participants=1)
    )
    path = tmp_path / "split.json"
    save_split_manifest(manifest, path)
    return path


def _config(run_id: str = "test-cnn") -> CnnConfig:
    return CnnConfig(
        run_id=run_id,
        channel_variants=((2, 4),),
        dropout_values=(0.0,),
        feature=CnnFeatureConfig(sample_count=16),
        epochs=2,
        batch_size=4,
        learning_rate=0.01,
        minimum_recall=1.0,
        random_state=19,
    )


def test_cnn_tensor_shape_and_extraction_are_deterministic(tmp_path: Path) -> None:
    root = _fixture_dataset(tmp_path)
    recording = load_recording(discover_recordings(root)[0])
    config = CnnFeatureConfig(sample_count=16)

    first = extract_cnn_tensor(recording, config)
    second = extract_cnn_tensor(recording, config)

    assert first.shape == (1, 16, 52)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_cnn_threshold_requires_grouped_recall() -> None:
    labels = np.asarray(["fall", "non-fall", "fall", "non-fall"], dtype=object)
    scores = np.asarray([0.7, 0.6, 0.4, 0.3])
    groups = ["ID1", "ID1", "ID2", "ID2"]

    selected, table = select_cnn_threshold(labels, scores, groups, minimum_recall=1.0)

    assert selected["minimum_group_recall"] == 1.0
    assert selected["threshold"] <= 0.4
    assert table


def test_cnn_training_persists_artifact_report_and_isolation(tmp_path: Path) -> None:
    root = _fixture_dataset(tmp_path)
    manifest_path = _fixture_manifest(root, tmp_path)
    run = train_cnn(root, manifest_path, tmp_path / "cnn-run", _config())

    assert run.model_path.exists()
    assert run.artifact_path.exists()
    assert run.report_path.exists()
    artifact = load_evaluation_result(run.artifact_path)
    assert artifact["model"]["family"] == "cnn"
    assert artifact["model"]["architecture"]["channels"] == [2, 4]
    assert artifact["features"]["shape"] == [1, 16, 52]
    assert artifact["selection"]["selection_split"] == "grouped_validation"
    assert artifact["selection"]["protocol"]["test_participants"]
    assert set(artifact["selection"]["protocol"]["participants"]).isdisjoint(
        artifact["selection"]["protocol"]["test_participants"]
    )
    assert artifact["evaluations"]["test"]["hard_example_analysis"]["selection_excluded"] is True
    assert "Hard-example diagnostics" in run.report_path.read_text(encoding="utf-8")


def test_cnn_inference_reproduces_test_predictions(tmp_path: Path) -> None:
    root = _fixture_dataset(tmp_path)
    manifest_path = _fixture_manifest(root, tmp_path)
    run = train_cnn(root, manifest_path, tmp_path / "cnn-run", _config())
    artifact = run.artifact
    model = load_cnn(run.model_path)
    references = {reference.recording_id: reference for reference in discover_recordings(root)}
    test_ids = artifact["evaluations"]["test"]["observation_ids"]
    recordings = [load_recording(references[recording_id]) for recording_id in test_ids]

    predictions = model.predict_recordings(recordings)

    assert predictions.tolist() == artifact["evaluations"]["test"]["y_pred"]
