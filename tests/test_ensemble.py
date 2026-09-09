from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from evaluation import load_evaluation_result, save_evaluation_result
from wifall.ensemble import train_ensemble
from wifall.loader import (
    SplitConfig,
    generate_split_manifest,
    load_split_manifest,
    save_split_manifest,
)
from wifall.training import load_split_records

HEADER = ["type", "seq", "timestamp", "taget", "rssi", "data"]


def _write_recording(root: Path, participant: str, activity: str) -> None:
    directory = root / participant / activity
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{activity}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        offset = 1000 if activity == "fall" else 0
        for row_number in range(8):
            values = list(range(offset + row_number, offset + row_number + 104))
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


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    root = tmp_path / "WiFall"
    for participant_number in range(5):
        participant = f"ID{participant_number}"
        _write_recording(root, participant, "fall")
        _write_recording(root, participant, "sit")
    manifest = generate_split_manifest(
        root, SplitConfig(seed=7, test_participants=1, validation_participants=1)
    )
    manifest_path = tmp_path / "split.json"
    save_split_manifest(manifest, manifest_path)
    return root, manifest_path, manifest


def _source_artifact(
    root: Path,
    manifest_path: Path,
    output: Path,
    run_id: str,
    score_offset: float,
) -> Path:
    manifest = load_split_manifest(manifest_path)
    records = load_split_records(root, manifest)
    evaluations = {}
    for split, split_records in records.items():
        labels = [record.target for _, _, record in split_records]
        scores = [
            (0.9 if label == "fall" else 0.1) + score_offset
            for label in labels
        ]
        evaluations[split] = {
            "labels": ["non-fall", "fall"],
            "observation_ids": [record.reference.recording_id for _, _, record in split_records],
            "group_ids": [participant for participant, _, _ in split_records],
            "y_true": labels,
            "y_pred": ["fall" if score >= 0.5 else "non-fall" for score in scores],
            "scores": scores,
            "positive_scores": scores,
            "score_type": "decision_function",
            "metrics": {},
            "confusion_matrix": [],
        }
    artifact = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset": manifest["dataset"],
        "task": {
            "name": "binary_fall_detection",
            "signal": "WiFi CSI",
            "labels": ["non-fall", "fall"],
            "positive_label": "fall",
        },
        "model": {"family": run_id},
        "features": {"name": run_id},
        "preprocessing": {},
        "split": {"strategy": "participant-held-out", "participants": manifest["participants"]},
        "evaluation_split": "test",
        "evaluations": evaluations,
        "metrics": {},
        "confusion_matrix": [],
    }
    save_evaluation_result(artifact, output / "run.json")
    return output / "run.json"


def test_ensemble_aligns_scores_selects_on_validation_and_is_reproducible(
    tmp_path: Path,
) -> None:
    root, manifest_path, manifest = _fixture(tmp_path)
    svm_path = _source_artifact(root, manifest_path, tmp_path / "svm", "improved-svm", 0.0)
    rf_path = _source_artifact(root, manifest_path, tmp_path / "rf", "random-forest", 0.0)

    first = train_ensemble(root, manifest_path, svm_path, rf_path, tmp_path / "first")
    second = train_ensemble(root, manifest_path, svm_path, rf_path, tmp_path / "second")

    assert json.loads(first.artifact_path.read_text()) == json.loads(second.artifact_path.read_text())
    artifact = load_evaluation_result(first.artifact_path)
    assert artifact["model"]["family"] == "score_ensemble"
    assert artifact["model"]["calibration"]["fit_splits"] == ["train", "validation"]
    assert artifact["model"]["blend_weight"]["improved_svm"] + artifact["model"]["blend_weight"]["random_forest"] == 1.0
    assert artifact["selection"]["selection_split"] == "grouped_validation"
    assert artifact["selection"]["protocol"]["test_participants"] == manifest["participants"]["test"]
    assert not any(
        participant in artifact["selection"]
        for participant in manifest["participants"]["test"]
    )
    assert first.report_path.exists()
    assert "calibrated_svm_random_forest_blend" in first.report_path.read_text()

    selected = artifact["selection"]["selected"]
    for source_path in (svm_path, rf_path):
        source = load_evaluation_result(source_path)
        test_scores = [0.99 if score < 0.5 else 0.01 for score in source["evaluations"]["test"]["scores"]]
        source["evaluations"]["test"]["scores"] = test_scores
        source["evaluations"]["test"]["positive_scores"] = test_scores
        save_evaluation_result(source, source_path)
    changed_test = train_ensemble(root, manifest_path, svm_path, rf_path, tmp_path / "changed-test")
    assert changed_test.artifact["selection"]["selected"] == selected


def test_ensemble_rejects_misaligned_source_runs(tmp_path: Path) -> None:
    root, manifest_path, _ = _fixture(tmp_path)
    svm_path = _source_artifact(root, manifest_path, tmp_path / "svm", "improved-svm", 0.0)
    rf_path = _source_artifact(root, manifest_path, tmp_path / "rf", "random-forest", 0.0)
    artifact = json.loads(rf_path.read_text())
    artifact["evaluations"]["test"]["observation_ids"].reverse()
    save_evaluation_result(artifact, rf_path)

    with pytest.raises(ValueError, match="Source observation IDs differ for test"):
        train_ensemble(root, manifest_path, svm_path, rf_path, tmp_path / "ensemble")
