from __future__ import annotations

from pathlib import Path

from csi_bench.cross_dataset_report import average_balanced_accuracy, render_report


def _model(name: str, family: str, enetfall: float, wifall: float) -> dict:
    return {
        "name": name,
        "family": family,
        "evaluation_dir": Path("."),
        "training": {"config": {"lstm_hidden_size": 64}},
        "evaluations": {
            dataset: {
                "y_true": ["non-fall", "fall"],
                "labels": ["non-fall", "fall"],
                "confusion_matrix": [[1, 0], [0, 1]],
                "metrics": {
                    "balanced_accuracy": score,
                    "accuracy": score,
                    "precision": score,
                    "recall": score,
                    "f1": score,
                },
            }
            for dataset, score in (("ENetFall", enetfall), ("WiFall", wifall))
        },
    }


def test_report_ranks_with_equal_dataset_weight_and_embeds_neural_architectures(
    tmp_path,
) -> None:
    random_forest = _model("RF user", "random_forest", 0.75, 0.75)
    cnn = _model("CNN", "cnn", 0.9, 0.5)
    cnn_lstm = _model("CNN-LSTM", "cnn_lstm", 0.8, 0.4)

    assert average_balanced_accuracy(random_forest) == 0.75
    report = render_report([cnn, cnn_lstm, random_forest], tmp_path / "report.html")

    assert "Best performing model: RF user" in report
    assert "What data splits mean" in report
    assert "original train, validation, and test partitions" in report
    assert "75.0%" in report
    assert "Deep CNN architecture" in report
    assert "CNN-LSTM · user v3" in report
    assert report.count("True \\ Predicted") == 6
