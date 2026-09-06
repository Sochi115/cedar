from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from wifall import (
    FeatureConfig,
    SplitConfig,
    SvmConfig,
    extract_recording_features,
    generate_split_manifest,
    load_run_artifact,
    load_trained_svm,
    save_split_manifest,
    train_svm,
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
        for row_number in range(3):
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


def test_recording_features_are_fixed_size_and_deterministic(tmp_path: Path) -> None:
    root = _fixture_dataset(tmp_path)
    recording = load_recording(discover_recordings(root)[0])

    first = extract_recording_features(recording, FeatureConfig())
    second = extract_recording_features(recording, FeatureConfig())

    assert first.shape == (260,)
    np.testing.assert_array_equal(first, second)


def test_svm_training_persists_generic_artifact_and_model(tmp_path: Path) -> None:
    root = _fixture_dataset(tmp_path)
    manifest_path = _fixture_manifest(root, tmp_path)
    output = tmp_path / "run"

    run = train_svm(
        root,
        manifest_path,
        output,
        SvmConfig(run_id="test-run"),
    )

    assert run.artifact_path.exists()
    assert run.model_path.exists()
    artifact = load_run_artifact(run.artifact_path)
    assert artifact["run_id"] == "test-run"
    assert artifact["task"]["labels"] == ["non-fall", "fall"]
    assert artifact["model"]["family"] == "svm"
    assert artifact["features"]["count"] == 260
    assert artifact["preprocessing"]["fit_split"] == "train"
    assert set(artifact["evaluations"]) == {"train", "validation", "test"}
    assert artifact["confusion_matrix"] == artifact["evaluations"]["test"]["confusion_matrix"]
    for metric in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1",
        "false_positive_rate",
        "support",
    ):
        assert metric in artifact["metrics"]


def test_svm_scaler_fits_train_recordings_only_and_splits_have_no_leakage(
    tmp_path: Path,
) -> None:
    root = _fixture_dataset(tmp_path)
    manifest_path = _fixture_manifest(root, tmp_path)
    run = train_svm(root, manifest_path, tmp_path / "run")
    artifact = run.artifact

    model = load_trained_svm(run.model_path)
    train_count = len(artifact["evaluations"]["train"]["recording_ids"])
    assert model.estimator.named_steps["scaler"].n_samples_seen_ == train_count

    references = {reference.recording_id: reference for reference in discover_recordings(root)}
    test_ids = artifact["evaluations"]["test"]["recording_ids"]
    test_recordings = [load_recording(references[recording_id]) for recording_id in test_ids]
    predictions = model.predict_recordings(test_recordings)
    assert [model_label for model_label in predictions] == [
        1 if label == "fall" else 0 for label in artifact["evaluations"]["test"]["y_pred"]
    ]

    participant_sets = {
        split: set(evaluation["participant_ids"])
        for split, evaluation in artifact["evaluations"].items()
    }
    assert participant_sets["train"].isdisjoint(participant_sets["validation"])
    assert participant_sets["train"].isdisjoint(participant_sets["test"])
    assert participant_sets["validation"].isdisjoint(participant_sets["test"])


def test_svm_artifact_is_reproducible_for_same_inputs(tmp_path: Path) -> None:
    root = _fixture_dataset(tmp_path)
    manifest_path = _fixture_manifest(root, tmp_path)
    config = SvmConfig(run_id="reproducible-run")

    first = train_svm(root, manifest_path, tmp_path / "first", config)
    second = train_svm(root, manifest_path, tmp_path / "second", config)

    first_json = json.loads(first.artifact_path.read_text(encoding="utf-8"))
    second_json = json.loads(second.artifact_path.read_text(encoding="utf-8"))
    assert first_json == second_json
