from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from evaluation import load_evaluation_result
from wifall import (
    ImprovedFeatureConfig,
    RandomForestConfig,
    SplitConfig,
    extract_improved_features,
    extract_random_forest_features,
    generate_split_manifest,
    load_random_forest,
    load_split_manifest,
    save_split_manifest,
    train_random_forest,
)
from wifall.loader import discover_recordings, load_recording
from wifall.random_forest import select_random_forest_threshold

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


def _config(run_id: str = "test-random-forest") -> RandomForestConfig:
    return RandomForestConfig(
        run_id=run_id,
        n_estimators=(5,),
        max_depth=(None,),
        min_samples_leaf=(1,),
        max_features=("sqrt",),
        max_samples=(None,),
        criterion=("gini",),
        calibration_method="sigmoid",
        random_state=19,
    )


def test_random_forest_uses_improved_feature_representation(tmp_path: Path) -> None:
    root = _fixture_dataset(tmp_path)
    recording = load_recording(discover_recordings(root)[0])

    np.testing.assert_array_equal(
        extract_random_forest_features(recording, ImprovedFeatureConfig()),
        extract_improved_features(recording, ImprovedFeatureConfig()),
    )


def test_random_forest_threshold_policies_are_explicit() -> None:
    labels = np.asarray(["fall", "fall", "non-fall", "non-fall"], dtype=object)
    scores = np.asarray([0.8, 0.6, 0.4, 0.2])

    for policy in (
        "maximize_grouped_validation_f1_then_sensitivity",
        "maximize_grouped_validation_balanced_accuracy_then_sensitivity",
        "maximize_grouped_validation_recall_constrained_accuracy",
    ):
        selected, table = select_random_forest_threshold(labels, scores, policy, 1.0)

        assert np.isfinite(selected["threshold"])
        assert table


def test_random_forest_training_persists_separate_generic_artifacts_and_report(
    tmp_path: Path,
) -> None:
    root = _fixture_dataset(tmp_path)
    manifest_path = _fixture_manifest(root, tmp_path)
    improved_output = tmp_path / "improved-svm-run"
    improved_output.mkdir()
    sentinel = improved_output / "model.joblib"
    sentinel.write_bytes(b"improved-model")
    run = train_random_forest(root, manifest_path, tmp_path / "random-forest-run", _config())

    assert run.model_path.exists()
    assert sentinel.read_bytes() == b"improved-model"
    assert run.artifact_path.exists()
    assert run.report_path.exists()
    artifact = load_evaluation_result(run.artifact_path)
    assert artifact["model"]["family"] == "random_forest"
    assert artifact["model"]["training_code"] == "wifall.random_forest"
    assert artifact["model"]["decision_threshold"] == artifact["selection"]["selected"]["threshold"]
    assert artifact["model"]["threshold_policy"] == "maximize_grouped_validation_f1_then_sensitivity"
    assert artifact["model"]["calibration"] == {
        "method": "sigmoid",
        "fit_split": "grouped_validation_oof",
        "positive_label": "fall",
    }
    assert {"max_samples", "criterion"} <= artifact["model"]["parameters"].keys()
    assert artifact["selection"]["selected"]["selection_metric"] == "f1"
    assert artifact["features"]["name"] == "csi_normalized_temporal_v1"
    assert artifact["preprocessing"]["fit_split"] == "train+validation"
    assert artifact["split"]["strategy"] == "participant-held-out"
    assert artifact["split"]["participants"] == load_split_manifest(manifest_path)["participants"]
    assert artifact["selection"]["selection_split"] == "grouped_validation"
    protocol = artifact["selection"]["protocol"]
    assert protocol["method"] == "leave_one_participant_out"
    assert protocol["participant_weight_policy"] == "equal_participant_contribution"
    assert len(protocol["folds"]) == 4
    assert {
        tuple(fold["validation_participants"]) for fold in protocol["folds"]
    } == {(participant,) for participant in protocol["participants"]}
    assert all(
        set(fold["validation_participants"]).isdisjoint(protocol["test_participants"])
        for fold in protocol["folds"]
    )
    assert set(protocol["participants"]).isdisjoint(protocol["test_participants"])
    for fold in protocol["folds"]:
        contributions = {
            round(values["recordings"] * values["weight_per_recording"], 8)
            for values in fold["participant_weights"].values()
        }
        assert len(contributions) == 1
    assert set(artifact["evaluations"]) == {"train", "validation", "test"}
    assert artifact["evaluations"]["test"]["participant_metrics"]
    assert "random_forest" in run.report_path.read_text(encoding="utf-8")


def test_random_forest_model_reproduces_test_predictions_and_has_explicit_score_orientation(
    tmp_path: Path,
) -> None:
    root = _fixture_dataset(tmp_path)
    manifest_path = _fixture_manifest(root, tmp_path)
    run = train_random_forest(root, manifest_path, tmp_path / "random-forest-run", _config())
    artifact = run.artifact

    model = load_random_forest(run.model_path)
    references = {reference.recording_id: reference for reference in discover_recordings(root)}
    test_ids = artifact["evaluations"]["test"]["observation_ids"]
    test_recordings = [load_recording(references[recording_id]) for recording_id in test_ids]
    predictions = model.predict_recordings(test_recordings)

    assert predictions.tolist() == artifact["evaluations"]["test"]["y_pred"]
    assert model.estimator.positive_label == "fall"
    assert model.estimator.classes_.tolist() == ["non-fall", "fall"]
    assert model.estimator.calibrator is not None
    assert model.estimator.calibrator.method == "sigmoid"
    assert artifact["evaluations"]["test"]["score_type"] == "decision_function"
    assert artifact["evaluations"]["test"]["scores"]


def test_tuned_random_forest_remains_comparable_to_uncalibrated_baseline(
    tmp_path: Path,
) -> None:
    root = _fixture_dataset(tmp_path)
    manifest_path = _fixture_manifest(root, tmp_path)
    baseline = train_random_forest(
        root,
        manifest_path,
        tmp_path / "baseline",
        replace(_config("baseline"), calibration_method="none"),
    )
    tuned = train_random_forest(
        root,
        manifest_path,
        tmp_path / "tuned",
        replace(_config("tuned"), max_depth=(None, 4)),
    )

    assert baseline.artifact["model"]["calibration"]["method"] == "none"
    assert tuned.artifact["model"]["calibration"]["method"] == "sigmoid"
    assert baseline.artifact["split"]["participants"] == tuned.artifact["split"]["participants"]
    assert baseline.artifact_path != tuned.artifact_path
    assert len(tuned.artifact["selection"]["candidates"]) == 2


def test_random_forest_selection_and_artifact_are_reproducible(tmp_path: Path) -> None:
    root = _fixture_dataset(tmp_path)
    manifest_path = _fixture_manifest(root, tmp_path)
    config = _config("reproducible-random-forest")

    first = train_random_forest(root, manifest_path, tmp_path / "first", config)
    second = train_random_forest(root, manifest_path, tmp_path / "second", config)

    first_json = json.loads(first.artifact_path.read_text(encoding="utf-8"))
    second_json = json.loads(second.artifact_path.read_text(encoding="utf-8"))
    assert first_json == second_json
    selected = first_json["selection"]["selected"]
    assert selected["selection_metrics"]["split"] == "grouped_validation"
    assert "test" not in first_json["selection"]["protocol"]["participants"]
