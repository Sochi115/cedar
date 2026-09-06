from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluation import (
    evaluate_classifier,
    load_evaluation_result,
    render_html_report,
    save_evaluation_result,
)


class FixedClassifier:
    classes_ = np.asarray(["negative", "positive"])

    def __init__(self, predictions: list[str], scores: list[float] | None = None) -> None:
        self.predictions = predictions
        self.scores = scores
        self.fit_called = False

    def fit(self, _features: object, _labels: object) -> None:
        self.fit_called = True
        raise AssertionError("validation must not fit model")

    def predict(self, _features: object) -> np.ndarray:
        return np.asarray(self.predictions)

    def decision_function(self, _features: object) -> np.ndarray:
        if self.scores is None:
            raise AttributeError("scores unavailable")
        return np.asarray(self.scores)


class NoScoreClassifier:
    def predict(self, _features: object) -> np.ndarray:
        return np.asarray(["negative", "negative", "negative", "positive"])


def _result(model: object) -> dict:
    return evaluate_classifier(
        model,
        np.zeros((4, 2)),
        ["negative", "positive", "negative", "positive"],
        label_order=["negative", "positive"],
        positive_label="positive",
        run_id="generic-fixture",
        dataset_metadata={"name": "Example dataset"},
        task_metadata={"name": "Example classification"},
        model_metadata={"family": "Example model"},
        feature_metadata={"name": "Example features"},
        preprocessing_metadata={"steps": ["Example preprocessing"]},
        split_metadata={"strategy": "held-out"},
        observation_ids=["sample-0", "sample-1", "sample-2", "sample-3"],
        group_ids=["group-a", "group-a", "group-b", "group-b"],
    )


def test_generic_validator_does_not_fit_and_emits_metrics_and_scores() -> None:
    model = FixedClassifier(
        ["negative", "positive", "positive", "positive"],
        [-2.0, 2.0, 1.0, 3.0],
    )

    result = _result(model)

    assert model.fit_called is False
    evaluation = result["evaluations"]["validation"]
    assert evaluation["score_type"] == "decision_function"
    assert evaluation["scores"] == [-2.0, 2.0, 1.0, 3.0]
    assert evaluation["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["metrics"]["sensitivity"] == 1.0
    assert result["metrics"]["specificity"] == 0.5
    assert "macro_avg" in result["metrics"]
    assert "weighted_avg" in result["metrics"]
    assert result["task"]["labels"] == ["negative", "positive"]


def test_generic_validator_marks_score_metrics_unavailable_without_scores() -> None:
    result = _result(NoScoreClassifier())

    evaluation = result["evaluations"]["validation"]
    assert evaluation["scores"] is None
    assert evaluation["score_type"] is None
    assert result["metrics"]["roc_auc"] is None
    assert result["metrics"]["pr_auc"] is None


def test_generic_result_round_trip_and_html_rendering(tmp_path: Path) -> None:
    result = _result(NoScoreClassifier())
    result_path = tmp_path / "result.json"
    html_path = tmp_path / "report.html"

    save_evaluation_result(result, result_path)
    loaded = load_evaluation_result(result_path)
    html = render_html_report(loaded)
    html_path.write_text(html, encoding="utf-8")

    assert loaded == result
    assert "Example dataset" in html
    assert "Example model" in html
    assert "Confusion matrix" in html
    assert "50.0%" in html
    assert "Not available" in html
    assert "Example classification" in html
    assert "<script" not in html
    assert html_path.read_text(encoding="utf-8") == html
