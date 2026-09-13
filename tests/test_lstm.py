from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from evaluation import load_evaluation_result
from wifall import (
    DurationNormalizedConfig,
    LstmConfig,
    LstmFeatureConfig,
    SplitConfig,
    WiFallLstmPipeline,
    build_duration_normalized_dataset,
    extract_lstm_sequence,
    generate_split_manifest,
    load_duration_normalized_dataset,
    load_lstm,
    load_recording,
    save_split_manifest,
    select_lstm_threshold,
    train_lstm,
)
from wifall.loader import discover_recordings
from wifall.lstm import SmallLstm

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


def _processed_dataset(root: Path, manifest_path: Path, tmp_path: Path) -> Path:
    output = tmp_path / "duration-normalized"
    build_duration_normalized_dataset(
        root,
        manifest_path,
        output,
        DurationNormalizedConfig(sample_count=16, section_count=4),
    )
    return output


def _config(run_id: str = "test-lstm") -> LstmConfig:
    return LstmConfig(
        run_id=run_id,
        hidden_sizes=(4,),
        layer_counts=(1,),
        dropout_values=(0.0,),
        feature=LstmFeatureConfig(sample_count=16, causal_window=8, window_stride=2),
        epochs=2,
        batch_size=4,
        learning_rate=0.01,
        positive_weight=1.0,
        minimum_recall=1.0,
        random_state=19,
    )


def test_lstm_sequence_shape_and_extraction_are_deterministic(tmp_path: Path) -> None:
    root = _fixture_dataset(tmp_path)
    recording = load_recording(discover_recordings(root)[0])
    config = LstmFeatureConfig(sample_count=16, causal_window=8)

    first = extract_lstm_sequence(recording, config)
    second = extract_lstm_sequence(recording, config)

    assert first.shape == (16, 52)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)

    altered = replace(recording, csi=recording.csi.copy())
    altered.csi[4:] *= 1000.0
    altered_sequence = extract_lstm_sequence(altered, config)
    np.testing.assert_array_equal(first[:4], altered_sequence[:4])


def test_lstm_causal_scores_ignore_future_samples() -> None:
    config = LstmFeatureConfig(sample_count=16, causal_window=4)
    network = SmallLstm(config.subcarrier_count, hidden_size=4, layer_count=1, dropout=0.0)
    pipeline = WiFallLstmPipeline(network, config, threshold=0.5)
    sequence = np.random.default_rng(3).normal(size=(16, 52)).astype(np.float32)
    altered = sequence.copy()
    altered[8:] *= 1000.0

    first = pipeline.causal_scores(sequence)
    second = pipeline.causal_scores(altered)

    np.testing.assert_array_equal(first[:8], second[:8])
    assert np.isnan(first[:3]).all()
    assert np.isfinite(first[3:]).all()


def test_lstm_threshold_requires_grouped_recall() -> None:
    labels = np.asarray(["fall", "non-fall", "fall", "non-fall"], dtype=object)
    scores = np.asarray([0.7, 0.6, 0.4, 0.3])
    groups = ["ID1", "ID1", "ID2", "ID2"]

    selected, table = select_lstm_threshold(labels, scores, groups, minimum_recall=1.0)

    assert selected["minimum_group_recall"] == 1.0
    assert selected["threshold"] <= 0.4
    assert table


def test_lstm_training_persists_artifacts_and_isolates_test(tmp_path: Path) -> None:
    root = _fixture_dataset(tmp_path)
    manifest_path = _fixture_manifest(root, tmp_path)
    processed = _processed_dataset(root, manifest_path, tmp_path)
    run = train_lstm(root, manifest_path, tmp_path / "lstm-run", _config(), processed_dataset=processed)

    assert run.model_path.exists()
    assert run.artifact_path.exists()
    assert run.report_path.exists()
    artifact = load_evaluation_result(run.artifact_path)
    assert artifact["model"]["family"] == "lstm"
    assert artifact["model"]["architecture"]["hidden_size"] == 4
    assert artifact["model"]["real_time"]["causal"] is True
    assert artifact["model"]["real_time"]["max_lookback_samples"] == 8
    assert artifact["training"]["early_stopping"]["enabled"] is True
    assert artifact["selection"]["selected"]["optimizer"] == "adam"
    assert artifact["selection"]["selected"]["stopping_epochs"]
    assert artifact["features"]["shape"] == [16, 52]
    assert artifact["data_source"]["type"] == "duration_normalized_dataset"
    assert artifact["selection"]["selection_split"] == "grouped_validation"
    assert set(artifact["evaluations"]) == {"train", "validation", "test"}
    assert artifact["evaluations"]["test"]["hard_example_analysis"]["selection_excluded"] is True
    assert artifact["evaluations"]["test"]["alerting"]["alert_cadence_samples"] == 1
    assert set(artifact["selection"]["protocol"]["participants"]).isdisjoint(
        artifact["selection"]["protocol"]["test_participants"]
    )
    report = run.report_path.read_text(encoding="utf-8")
    assert "Hard-example diagnostics" in report
    assert "Real-time alert behavior" in report


def test_lstm_training_is_reproducible_and_persisted_model_predicts(tmp_path: Path) -> None:
    root = _fixture_dataset(tmp_path)
    manifest_path = _fixture_manifest(root, tmp_path)
    processed = _processed_dataset(root, manifest_path, tmp_path)
    config = _config("reproducible-lstm")
    first = train_lstm(root, manifest_path, tmp_path / "first", config, processed_dataset=processed)
    second = train_lstm(root, manifest_path, tmp_path / "second", config, processed_dataset=processed)

    assert json.loads(first.artifact_path.read_text()) == json.loads(second.artifact_path.read_text())
    model = load_lstm(first.model_path)
    dataset = load_duration_normalized_dataset(processed)
    test_records = dataset.splits["test"]
    predictions = model.predict_recordings([recording for _, _, recording in test_records])

    assert predictions.tolist() == first.artifact["evaluations"]["test"]["y_pred"]

    raw_recording = load_recording(discover_recordings(root)[0])
    with pytest.raises(ValueError, match="fixed-rate duration-normalized"):
        model.predict_recordings([raw_recording])
