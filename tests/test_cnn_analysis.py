from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import save_evaluation_result
from wifall.cnn_analysis import analyze_cnn_runs


def _artifact(
    output: Path,
    run_id: str,
    predictions_by_split: dict[str, list[str]],
    scores_by_split: dict[str, list[float]],
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    ids = {
        "validation": ["ID0/fall/a.csv", "ID0/sit/b.csv", "ID1/fall/c.csv", "ID1/sit/d.csv"],
        "test": ["ID2/fall/e.csv", "ID2/sit/f.csv", "ID3/fall/g.csv", "ID3/sit/h.csv"],
    }
    labels = ["fall", "non-fall", "fall", "non-fall"]
    evaluations = {}
    for split in ("validation", "test"):
        evaluations[split] = {
            "observation_ids": ids[split],
            "group_ids": ["ID0", "ID0", "ID1", "ID1"] if split == "validation" else ["ID2", "ID2", "ID3", "ID3"],
            "y_true": labels,
            "y_pred": predictions_by_split[split],
            "scores": scores_by_split[split],
            "positive_scores": scores_by_split[split],
            "score_type": "decision_function",
            "metrics": {"recall": 0.5, "specificity": 0.5},
            "confusion_matrix": [],
            "hard_example_analysis": {
                "error_counts": {"false_negatives": 1, "false_positives": 1},
                "missed_fall_score_range": {"min": 0.2, "max": 0.3},
                "false_negatives": {"duration_buckets": {}, "signal_quality": {}},
                "false_positives": {"duration_buckets": {}, "signal_quality": {}},
            },
        }
    artifact = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset": {"name": "WiFall"},
        "task": {"name": "binary_fall_detection", "labels": ["non-fall", "fall"]},
        "model": {"family": run_id},
        "features": {"name": run_id},
        "preprocessing": {},
        "split": {"strategy": "participant-held-out"},
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
        "cnn": {"validation": ["fall", "non-fall", "fall", "non-fall"], "test": ["fall", "fall", "non-fall", "non-fall"]},
        "svm": {"validation": ["fall", "fall", "fall", "non-fall"], "test": ["non-fall", "fall", "fall", "non-fall"]},
        "rf": {"validation": ["fall", "non-fall", "non-fall", "non-fall"], "test": ["fall", "non-fall", "fall", "non-fall"]},
        "ensemble": {"validation": ["fall", "non-fall", "fall", "non-fall"], "test": ["fall", "fall", "fall", "non-fall"]},
    }
    scores = {
        name: {
            "validation": [0.9, 0.1, 0.8, 0.2],
            "test": [0.9, 0.8, 0.2, 0.1],
        }
        for name in predictions
    }
    return {
        name: _artifact(tmp_path / name, name, values, scores[name])
        for name, values in predictions.items()
    }


def test_cnn_analysis_reports_complementary_and_shared_errors(tmp_path: Path) -> None:
    runs = _runs(tmp_path)

    result = analyze_cnn_runs(
        runs["cnn"],
        (runs["svm"], runs["rf"], runs["ensemble"]),
        tmp_path / "analysis",
    )

    assert result.artifact_path.exists()
    artifact = result.artifact
    assert artifact["model_order"] == ["cnn", "svm", "rf", "ensemble"]
    assert artifact["errors"]["cnn"]["test"]["false_negatives"] == ["ID3/fall/g.csv"]
    assert "ID2/fall/e.csv" in artifact["complementarity"]["cnn_catches_fall_missed_by"]["svm"]
    assert artifact["threshold_analysis"]["uses_split"] == "validation"
    assert artifact["cnn_diagnostics"]["test"]["temporal_pattern"]["available"] is False
    assert "CNN analysis" in result.report_path.read_text(encoding="utf-8")


def test_cnn_analysis_is_deterministic_and_rejects_misalignment(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    first = analyze_cnn_runs(runs["cnn"], (runs["svm"], runs["rf"], runs["ensemble"]), tmp_path / "first")
    second = analyze_cnn_runs(runs["cnn"], (runs["svm"], runs["rf"], runs["ensemble"]), tmp_path / "second")

    assert json.loads(first.artifact_path.read_text()) == json.loads(second.artifact_path.read_text())
    broken = json.loads(runs["rf"].read_text())
    broken["evaluations"]["test"]["observation_ids"].reverse()
    save_evaluation_result(broken, runs["rf"])
    with pytest.raises(ValueError, match="observation IDs differ"):
        analyze_cnn_runs(runs["cnn"], (runs["svm"], runs["rf"], runs["ensemble"]), tmp_path / "broken")
