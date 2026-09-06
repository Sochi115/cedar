from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evaluation import evaluate_classifier, save_evaluation_result
from evaluation.comparison import render_comparison_report_file


class FixtureClassifier:
    classes_ = np.asarray(["non-fall", "fall"])

    def __init__(self, predictions: list[str], scores: list[float]) -> None:
        self.predictions = predictions
        self.scores = scores

    def predict(self, _features: object) -> np.ndarray:
        return np.asarray(self.predictions)

    def decision_function(self, _features: object) -> np.ndarray:
        return np.asarray(self.scores)


class NoScoreClassifier:
    def predict(self, _features: object) -> np.ndarray:
        return np.asarray(["non-fall", "fall", "non-fall", "fall"])


def _result(
    *,
    model_family: str,
    dataset_name: str,
    task_name: str = "binary_fall_detection",
    strategy: str = "participant-held-out",
    with_scores: bool = True,
) -> dict:
    labels = ["non-fall", "fall", "non-fall", "fall"]
    model = (
        FixtureClassifier(labels, [-0.8, 0.8, -0.4, 0.4])
        if with_scores
        else NoScoreClassifier()
    )
    result = evaluate_classifier(
        model,
        np.zeros((4, 2)),
        labels,
        label_order=["non-fall", "fall"],
        positive_label="fall",
        run_id=f"{model_family}-run",
        dataset_metadata={"name": dataset_name, "version": "v1"},
        task_metadata={"name": task_name, "signal": "WiFi CSI"},
        model_metadata={
            "family": model_family,
            "training_code": f"training.{model_family}",
            "threshold_policy": "fixed",
            "parameters": {"seed": 7},
        },
        feature_metadata={"name": f"{model_family}-features", "count": 2, "window_count": 4},
        preprocessing_metadata={"steps": ["fixture"]},
        split_metadata={"strategy": strategy, "config": {"seed": 7}},
    )
    return result


def _write_run(root: Path, result: dict) -> Path:
    run_dir = root / result["run_id"]
    run_dir.mkdir()
    result_path = run_dir / "run.json"
    save_evaluation_result(result, result_path)
    (run_dir / "report.html").write_text("<html>run</html>", encoding="utf-8")
    return result_path


def test_comparison_report_indexes_runs_and_displays_metadata_and_metrics(
    tmp_path: Path,
) -> None:
    first = _write_run(tmp_path, _result(model_family="svm", dataset_name="WiFall"))
    second = _write_run(
        tmp_path,
        _result(model_family="random_forest", dataset_name="WiFall"),
    )
    output = tmp_path / "comparison" / "report.html"

    render_comparison_report_file([first, second], output)

    html = output.read_text(encoding="utf-8")
    assert "Evaluation comparison" in html
    assert "Analysis and summary" in html
    assert "Metric leaders" in html
    assert "Human interpretation" in html
    assert "Window-size analysis" in html
    assert "4 windows" in html
    assert "Highest" not in html
    assert "svm-run" in html
    assert "random_forest-run" in html
    assert "training.svm" in html
    assert "training.random_forest" in html
    assert "svm-features" in html
    assert "random_forest-features" in html
    assert "Accuracy" in html
    assert "ROC-AUC" in html
    assert "report.html" in html
    assert "WiFall" in html
    assert "participant-held-out" in html
    assert "<script" not in html


def test_comparison_warns_about_incompatible_runs(tmp_path: Path) -> None:
    compatible = _result(model_family="svm", dataset_name="WiFall")
    incompatible = _result(
        model_family="other",
        dataset_name="OtherSet",
        task_name="multiclass_activity",
        strategy="random-record-held-out",
    )
    first = _write_run(tmp_path, compatible)
    second = _write_run(tmp_path, incompatible)
    output = tmp_path / "comparison.html"

    render_comparison_report_file([first, second], output)

    html = output.read_text(encoding="utf-8")
    assert "Compatibility warnings" in html
    assert "multiple datasets" in html
    assert "different task definitions" in html
    assert "different evaluation protocols" in html


def test_comparison_handles_missing_optional_metrics_and_future_runs(tmp_path: Path) -> None:
    first = _write_run(
        tmp_path,
        _result(model_family="future_model", dataset_name="NewSet", with_scores=False),
    )
    output = tmp_path / "comparison.html"

    render_comparison_report_file([first], output)

    html = output.read_text(encoding="utf-8")
    assert "future_model-run" in html
    assert "Not available" in html
    assert "PR-AUC" in html
    assert "No comparable metrics available" not in html


def test_comparison_inputs_are_loaded_as_generic_result_artifacts(tmp_path: Path) -> None:
    result_path = _write_run(tmp_path, _result(model_family="svm", dataset_name="WiFall"))
    output = tmp_path / "comparison.html"

    render_comparison_report_file([result_path.parent], output)

    assert output.exists()
    loaded = json.loads(result_path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == 1
