from __future__ import annotations

import numpy as np
import torch

from enetfall.evaluate_cnn_wifall_zero_shot import render_report
from enetfall.evaluate_wifall_zero_shot import adapt_wifall_amplitude
from enetfall.train_cnn import CompactCsiCnn, normalize_event


def test_wifall_adapter_produces_variable_length_cnn_input() -> None:
    amplitude = np.random.default_rng(42).uniform(0, 20, size=(240, 52))

    adapted = adapt_wifall_amplitude(amplitude)
    normalized = normalize_event(adapted)
    model = CompactCsiCnn().eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(normalized).unsqueeze(0))

    assert adapted.shape[1:] == (3, 30)
    assert normalized.shape == (3, adapted.shape[0], 30)
    assert logits.shape == (1, 2)


def test_zero_shot_html_report_is_standalone() -> None:
    report = render_report(
        {
            "dataset": {
                "evaluation_split": "test",
                "participants": ["ID5", "ID6"],
                "source_counts": {"targets": {"fall": 32, "non-fall": 32}},
            },
            "metrics": {
                "support": 64,
                "threshold": 0.4,
                "accuracy": 0.7,
                "balanced_accuracy": 0.7,
                "roc_auc": 0.8,
                "fall_precision": 0.75,
                "fall_recall": 0.65,
                "confusion_matrix_nonfall_fall": [[22, 10], [9, 23]],
            },
        }
    )

    assert report.startswith("<!doctype html>")
    assert "No WiFall training, fine-tuning, or threshold selection" in report
    assert "ID5, ID6" in report
    assert "<td>23</td>" in report
