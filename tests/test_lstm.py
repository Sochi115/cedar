from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from evaluation import load_evaluation_result
from wifall import (
    CausalWindowConfig,
    LstmConfig,
    LstmFeatureConfig,
    SplitConfig,
    build_causal_window_dataset,
    extract_causal_windows,
    generate_split_manifest,
    load_lstm,
    load_recording,
    save_split_manifest,
    select_lstm_threshold,
    train_lstm,
)
from wifall.loader import discover_recordings

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
            values = list(range(participant_number * 10 + activity_offset + row_number, participant_number * 10 + activity_offset + row_number + 104))
            writer.writerow(["CSI_DATA", str(row_number), f"2023-01-01 00:00:0{row_number}.000", activity, "-31", str(values)])


def _fixture_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "WiFall"
    for participant_number in range(5):
        participant = f"ID{participant_number}"
        _write_recording(root, participant, "fall")
        _write_recording(root, participant, "sit")
    return root


def _fixture_manifest(root: Path, tmp_path: Path) -> Path:
    manifest = generate_split_manifest(root, SplitConfig(seed=7, test_participants=1, validation_participants=1))
    path = tmp_path / "split.json"
    save_split_manifest(manifest, path)
    return path


def _config(run_id: str = "test-lstm") -> LstmConfig:
    return LstmConfig(
        run_id=run_id,
        hidden_sizes=(4,),
        layer_counts=(1,),
        dropout_values=(0.0,),
        feature=LstmFeatureConfig(window_samples=4, window_stride=2),
        epochs=2,
        batch_size=4,
        learning_rate=0.01,
        positive_weight=1.0,
        minimum_recall=1.0,
        random_state=19,
    )


def test_causal_windows_use_current_and_past_rows(tmp_path: Path) -> None:
    root = _fixture_dataset(tmp_path)
    recording = load_recording(discover_recordings(root)[0])
    config = CausalWindowConfig(window_samples=4, stride_samples=1)
    first, ends = extract_causal_windows(recording, config)

    altered = replace(recording, csi=recording.csi.copy())
    altered.csi[4:] *= 1000.0
    second, altered_ends = extract_causal_windows(altered, config)

    assert first.shape == (5, 4, 52)
    assert ends.tolist() == altered_ends.tolist()
    np.testing.assert_array_equal(first[0], second[0])


def test_lstm_threshold_requires_grouped_recall() -> None:
    labels = np.asarray(["fall", "non-fall", "fall", "non-fall"], dtype=object)
    scores = np.asarray([0.7, 0.6, 0.4, 0.3])
    groups = ["ID1", "ID1", "ID2", "ID2"]

    selected, table = select_lstm_threshold(labels, scores, groups, minimum_recall=1.0)

    assert selected["minimum_group_recall"] == 1.0
    assert selected["threshold"] <= 0.4
    assert table


def test_lstm_training_persists_raw_causal_artifacts(tmp_path: Path) -> None:
    root = _fixture_dataset(tmp_path)
    manifest_path = _fixture_manifest(root, tmp_path)
    run = train_lstm(root, manifest_path, tmp_path / "lstm-run", _config())

    assert run.model_path.exists()
    assert run.artifact_path.exists()
    assert run.report_path.exists()
    artifact = load_evaluation_result(run.artifact_path)
    assert artifact["model"]["family"] == "lstm"
    assert artifact["model"]["architecture"]["hidden_size"] == 4
    assert artifact["model"]["real_time"]["causal"] is True
    assert artifact["data_source"]["type"] == "raw_timestamped_csi"
    assert artifact["features"]["window_samples"] == 4
    assert artifact["preprocessing"]["labeling"]["weak_recording_labels"] is True
    assert artifact["training"]["early_stopping"]["enabled"] is True
    assert set(artifact["evaluations"]) == {"train", "validation", "test"}
    assert artifact["evaluations"]["test"]["alerting"]["causal"] is True
    assert artifact["evaluations"]["test"]["hard_example_analysis"]["selection_excluded"] is True
    assert "Real-time alert behavior" in run.report_path.read_text(encoding="utf-8")


def test_lstm_can_consume_persisted_causal_windows(tmp_path: Path) -> None:
    root = _fixture_dataset(tmp_path)
    manifest_path = _fixture_manifest(root, tmp_path)
    window_run = build_causal_window_dataset(
        root,
        manifest_path,
        tmp_path / "causal-windows",
        CausalWindowConfig(window_samples=4, stride_samples=2),
    )

    run = train_lstm(
        root,
        manifest_path,
        tmp_path / "lstm-persisted",
        _config(),
        causal_window_dataset=window_run.output_dir,
    )

    assert run.artifact["data_source"]["causal_window_dataset"] == str(window_run.output_dir)


def test_lstm_training_is_reproducible_and_persisted_model_predicts(tmp_path: Path) -> None:
    root = _fixture_dataset(tmp_path)
    manifest_path = _fixture_manifest(root, tmp_path)
    config = _config("reproducible-lstm")
    first = train_lstm(root, manifest_path, tmp_path / "first", config)
    second = train_lstm(root, manifest_path, tmp_path / "second", config)

    assert json.loads(first.artifact_path.read_text()) == json.loads(second.artifact_path.read_text())
    model = load_lstm(first.model_path)
    references = {reference.recording_id: reference for reference in discover_recordings(root)}
    test_ids = first.artifact["evaluations"]["test"]["observation_ids"]
    recordings = [load_recording(references[recording_id]) for recording_id in test_ids]

    predictions = model.predict_recordings(recordings)

    assert predictions.tolist() == first.artifact["evaluations"]["test"]["y_pred"]
