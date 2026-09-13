from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import save_evaluation_result
from wifall.lstm_analysis import analyze_lstm_runs


def _artifact(
    output: Path,
    run_id: str,
    validation_predictions: list[str],
    test_predictions: list[str],
    validation_scores: list[float],
    test_scores: list[float],
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    ids = {
        "validation": ["ID0/fall/a.csv", "ID0/sit/b.csv", "ID1/fall/c.csv", "ID1/sit/d.csv"],
        "test": ["ID2/fall/e.csv", "ID2/sit/f.csv", "ID3/fall/g.csv", "ID3/sit/h.csv"],
    }
    labels = ["fall", "non-fall", "fall", "non-fall"]
    groups = {
        "validation": ["ID0", "ID0", "ID1", "ID1"],
        "test": ["ID2", "ID2", "ID3", "ID3"],
    }
    evaluations = {}
    for split, predictions, scores in (
        ("validation", validation_predictions, validation_scores),
        ("test", test_predictions, test_scores),
    ):
        evaluations[split] = {
            "observation_ids": ids[split],
            "group_ids": groups[split],
            "y_true": labels,
            "y_pred": predictions,
            "scores": scores,
            "positive_scores": scores,
            "score_type": "predict_proba",
            "metrics": {},
            "confusion_matrix": [],
        }
    artifact = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset": {"name": "WiFall"},
        "task": {
            "name": "binary_fall_detection",
            "labels": ["non-fall", "fall"],
            "positive_label": "fall",
            "signal": "WiFi CSI",
        },
        "model": {"family": run_id},
        "features": {"name": run_id},
        "preprocessing": {},
        "split": {"strategy": "participant-held-out"},
        "data_source": {"type": "duration_normalized_dataset", "path": "duration-normalized"},
        "evaluation_split": "test",
        "evaluations": evaluations,
        "metrics": {},
        "confusion_matrix": [],
    }
    path = output / "run.json"
    save_evaluation_result(artifact, path)
    return path


def _runs(tmp_path: Path) -> dict[str, Path]:
    predictions = {
        "lstm": {
            "validation": ["fall", "fall", "fall", "non-fall"],
            "test": ["fall", "fall", "non-fall", "non-fall"],
        },
        "cnn": {
            "validation": ["fall", "non-fall", "fall", "non-fall"],
            "test": ["fall", "non-fall", "fall", "non-fall"],
        },
        "svm": {
            "validation": ["fall", "fall", "fall", "non-fall"],
            "test": ["non-fall", "fall", "fall", "non-fall"],
        },
        "rf": {
            "validation": ["fall", "non-fall", "fall", "non-fall"],
            "test": ["fall", "non-fall", "fall", "non-fall"],
        },
    }
    scores = {
        name: {
            "validation": [0.9, 0.8, 0.7, 0.1],
            "test": [0.9, 0.8, 0.2, 0.1],
        }
        for name in predictions
    }
    return {
        name: _artifact(
            tmp_path / name,
            name,
            values["validation"],
            values["test"],
            scores[name]["validation"],
            scores[name]["test"],
        )
        for name, values in predictions.items()
    }


def test_lstm_analysis_reports_metrics_errors_and_complementarity(tmp_path: Path) -> None:
    runs = _runs(tmp_path)

    result = analyze_lstm_runs(
        runs["lstm"],
        (runs["cnn"], runs["svm"], runs["rf"]),
        tmp_path / "analysis",
    )

    assert result.artifact_path.exists()
    artifact = result.artifact
    assert artifact["model_order"] == ["lstm", "cnn", "svm", "rf"]
    assert artifact["sources"][0]["test_metrics"]["accuracy"] == 0.5
    assert artifact["sources"][0]["test_participant_metrics"]["ID2"]["false_positives"] == 1
    assert artifact["errors"]["lstm"]["test"]["false_negatives"] == ["ID3/fall/g.csv"]
    assert "ID3/fall/g.csv" in artifact["complementarity"]["by_comparator"]["cnn"]["comparator_catches_falls_missed_by_lstm"]
    assert artifact["threshold_analysis"]["uses_split"] == "validation"
    assert artifact["recommendation"]["test_used_for_decision"] is False
    assert "LSTM analysis" in result.report_path.read_text(encoding="utf-8")


def test_lstm_analysis_recommendation_uses_validation_only(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    first = analyze_lstm_runs(runs["lstm"], (runs["cnn"], runs["svm"], runs["rf"]), tmp_path / "first")

    changed = json.loads(runs["lstm"].read_text())
    changed["evaluations"]["test"]["y_pred"] = ["fall", "fall", "fall", "fall"]
    save_evaluation_result(changed, runs["lstm"])
    second = analyze_lstm_runs(runs["lstm"], (runs["cnn"], runs["svm"], runs["rf"]), tmp_path / "second")

    assert first.artifact["recommendation"] == second.artifact["recommendation"]


def test_lstm_analysis_is_deterministic_and_rejects_misalignment(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    first = analyze_lstm_runs(runs["lstm"], (runs["cnn"], runs["svm"], runs["rf"]), tmp_path / "first")
    second = analyze_lstm_runs(runs["lstm"], (runs["cnn"], runs["svm"], runs["rf"]), tmp_path / "second")

    assert json.loads(first.artifact_path.read_text()) == json.loads(second.artifact_path.read_text())

    broken = json.loads(runs["rf"].read_text())
    broken["evaluations"]["test"]["observation_ids"].reverse()
    save_evaluation_result(broken, runs["rf"])
    with pytest.raises(ValueError, match="observation IDs differ"):
        analyze_lstm_runs(runs["lstm"], (runs["cnn"], runs["svm"], runs["rf"]), tmp_path / "broken")
