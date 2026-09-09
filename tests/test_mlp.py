from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from evaluation import load_evaluation_result
from wifall import (
    MlpConfig,
    MlpFeatureConfig,
    SplitConfig,
    extract_mlp_features,
    generate_split_manifest,
    load_mlp,
    save_split_manifest,
    select_mlp_threshold,
    train_mlp,
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


def _config(run_id: str = "test-mlp") -> MlpConfig:
    return MlpConfig(
        run_id=run_id,
        hidden_layer_sizes=((8,),),
        alpha_values=(0.01,),
        solver="lbfgs",
        max_iter=100,
        minimum_recall=1.0,
        random_state=19,
    )


def test_mlp_feature_extraction_is_fixed_size_and_deterministic(tmp_path: Path) -> None:
    root = _fixture_dataset(tmp_path)
    recording = load_recording(discover_recordings(root)[0])
    config = MlpFeatureConfig(window_count=4)

    first = extract_mlp_features(recording, config)
    second = extract_mlp_features(recording, config)

    assert first.shape == (1040,)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_mlp_threshold_requires_recall_for_each_validation_group() -> None:
    labels = np.asarray(["fall", "non-fall", "fall", "non-fall"], dtype=object)
    scores = np.asarray([0.7, 0.6, 0.4, 0.3])
    groups = ["ID1", "ID1", "ID2", "ID2"]

    selected, table = select_mlp_threshold(labels, scores, groups, minimum_recall=1.0)

    assert selected["minimum_group_recall"] == 1.0
    assert selected["threshold"] <= 0.4
    assert table


def test_mlp_training_persists_artifacts_and_safety_selection(tmp_path: Path) -> None:
    root = _fixture_dataset(tmp_path)
    manifest_path = _fixture_manifest(root, tmp_path)
    run = train_mlp(root, manifest_path, tmp_path / "mlp-run", _config())

    assert run.model_path.exists()
    assert run.artifact_path.exists()
    assert run.report_path.exists()
    artifact = load_evaluation_result(run.artifact_path)
    assert artifact["model"]["family"] == "mlp"
    assert artifact["model"]["architecture"] == [8]
    assert artifact["model"]["balancing"]["method"] == "oversample"
    assert artifact["model"]["balancing"]["target_ratio"] == 0.5
    assert artifact["model"]["dimensionality_reduction"]["method"] == "pca"
    assert artifact["model"]["calibration"]["method"] == "sigmoid_logistic_oof"
    assert artifact["model"]["fit_splits"] == ["train", "validation"]
    assert artifact["preprocessing"]["fit_split"] == "train+validation_after_selection"
    assert artifact["selection"]["selection_split"] == "grouped_validation"
    assert artifact["selection"]["selected"]["threshold_metrics"]["minimum_group_recall"] >= 1.0
    assert set(artifact["evaluations"]) == {"train", "validation", "test"}
    assert artifact["evaluations"]["test"]["hard_example_analysis"]["selection_excluded"] is True
    assert "Hard-example diagnostics" in run.report_path.read_text(encoding="utf-8")
    assert set(artifact["selection"]["protocol"]["participants"]).isdisjoint(
        artifact["selection"]["protocol"]["test_participants"]
    )


def test_mlp_inference_reproduces_test_predictions_and_training_is_reproducible(
    tmp_path: Path,
) -> None:
    root = _fixture_dataset(tmp_path)
    manifest_path = _fixture_manifest(root, tmp_path)
    config = _config("reproducible-mlp")
    first = train_mlp(root, manifest_path, tmp_path / "first", config)
    second = train_mlp(root, manifest_path, tmp_path / "second", config)

    assert json.loads(first.artifact_path.read_text()) == json.loads(second.artifact_path.read_text())
    artifact = first.artifact
    model = load_mlp(first.model_path)
    references = {reference.recording_id: reference for reference in discover_recordings(root)}
    test_ids = artifact["evaluations"]["test"]["observation_ids"]
    recordings = [load_recording(references[recording_id]) for recording_id in test_ids]

    predictions = model.predict_recordings(recordings)

    assert predictions.tolist() == artifact["evaluations"]["test"]["y_pred"]
    assert model.feature_config.as_dict() == {
        key: artifact["features"][key]
        for key in ("name", "signal", "window_count", "statistics")
    }
