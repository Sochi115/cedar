from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from wifall import (
    ImprovedFeatureConfig,
    ImprovedSvmConfig,
    SplitConfig,
    audit_wifall,
    extract_improved_features,
    generate_split_manifest,
    load_improved_svm,
    save_split_manifest,
    select_threshold,
    train_improved_svm,
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


def test_audit_reports_valid_recordings_and_balance(tmp_path: Path) -> None:
    root = _fixture_dataset(tmp_path)

    audit = audit_wifall(root)

    assert audit["discovered_recordings"] == 10
    assert audit["valid_recordings"] == 10
    assert audit["target_counts"] == {"fall": 5, "non-fall": 5}
    assert audit["malformed_recordings"] == []
    assert audit["empty_recordings"] == []
    assert audit["label_mismatches"] == []


def test_audit_reports_empty_and_malformed_recordings(tmp_path: Path) -> None:
    root = _fixture_dataset(tmp_path)
    empty = root / "ID0" / "stand" / "empty.csv"
    empty.parent.mkdir(parents=True, exist_ok=True)
    empty.write_text(",".join(HEADER) + "\n\n", encoding="utf-8")
    malformed = root / "ID0" / "walk" / "malformed.csv"
    malformed.parent.mkdir(parents=True, exist_ok=True)
    with malformed.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerow(["CSI_DATA", "1", "2023-01-01 00:00:00.000", "walk", "-31", "[1, 2]"])

    audit = audit_wifall(root)

    assert audit["discovered_recordings"] == 12
    assert audit["valid_recordings"] == 10
    assert "ID0/stand/empty.csv" in audit["empty_recordings"]
    assert audit["malformed_recordings"][0]["recording_id"] == "ID0/walk/malformed.csv"


def test_improved_features_are_temporal_normalized_and_fixed_size(tmp_path: Path) -> None:
    root = _fixture_dataset(tmp_path)
    recording = load_recording(discover_recordings(root)[0])

    features = extract_improved_features(recording, ImprovedFeatureConfig())

    assert features.shape == (1040,)
    assert np.isfinite(features).all()
    assert np.abs(features).max() < 1000


def test_threshold_selection_prefers_f1_then_sensitivity() -> None:
    labels = np.asarray(["fall", "fall", "non-fall", "non-fall"], dtype=object)
    scores = np.asarray([0.9, 0.7, 0.3, 0.1])

    selected, table = select_threshold(labels, scores)

    assert selected.f1 == 1.0
    assert selected.sensitivity == 1.0
    assert selected.specificity == 1.0
    assert table


def test_improved_training_persists_model_run_report_and_audit(tmp_path: Path) -> None:
    root = _fixture_dataset(tmp_path)
    manifest_path = _fixture_manifest(root, tmp_path)
    run = train_improved_svm(
        root,
        manifest_path,
        tmp_path / "improved-run",
        ImprovedSvmConfig(c_values=(1.0,), gamma_values=("scale",)),
    )

    assert run.model_path.exists()
    assert run.artifact_path.exists()
    assert run.report_path.exists()
    assert run.audit_path.exists()
    report = run.report_path.read_text(encoding="utf-8")
    assert "Group metrics" in report
    assert "Threshold metrics" in report
    assert "Data audit" in report
    assert run.artifact["model"]["name"] == "improved_thresholded_rbf_svm"
    assert run.artifact["selection"]["selection_split"] == "validation"
    assert run.artifact["model"]["decision_threshold"] == run.artifact["selection"]["selected"]["threshold"]
    assert run.artifact["evaluations"]["validation"]["threshold_metrics"]
    assert run.artifact["evaluations"]["test"]["participant_metrics"]
    assert run.artifact["evaluations"]["test"]["score_distribution"]
    assert run.artifact["evaluations"]["test"]["hard_example_analysis"]["selection_excluded"] is True
    assert "Hard-example diagnostics" in report
    assert "data_audit" in run.artifact

    model = load_improved_svm(run.model_path)
    references = {reference.recording_id: reference for reference in discover_recordings(root)}
    test_ids = run.artifact["evaluations"]["test"]["observation_ids"]
    predictions = model.predict_recordings(
        [load_recording(references[recording_id]) for recording_id in test_ids]
    )
    assert len(predictions) == len(test_ids)
