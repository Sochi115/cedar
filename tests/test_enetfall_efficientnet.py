from __future__ import annotations

import numpy as np
import torch

from enetfall.train_efficientnet import (
    _render_report,
    build_efficientnet,
    normalize_event,
)


def test_normalize_event_preserves_csi_tensor_shape_and_finite_values() -> None:
    amplitude = np.random.default_rng(7).uniform(0, 10, size=(625, 3, 30)).astype(np.float32)

    normalized = normalize_event(amplitude)

    assert normalized.shape == (3, 625, 30)
    assert np.isfinite(normalized).all()
    assert normalized.min() >= -8.0
    assert normalized.max() <= 8.0


def test_report_includes_validation_loss_and_confusion_matrices() -> None:
    result = {
        "training": {
            "device": "cpu",
            "epochs_completed": 1,
            "best_epoch": 1,
            "best_validation_loss": 0.5,
            "batch_size": 60,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "validation_loss_history": [{"epoch": 1, "validation_loss": 0.5}],
        },
        "threshold": 0.4,
        "model": {"artifact": "model.pt"},
        "validation": {
            "support": 2,
            "balanced_accuracy": 0.5,
            "confusion_matrix_nonfall_fall": [[1, 0], [1, 0]],
        },
        "test": {
            "support": 2,
            "balanced_accuracy": 0.5,
            "confusion_matrix_nonfall_fall": [[1, 0], [1, 0]],
        },
    }

    report = _render_report(result)

    assert "Best validation loss: **0.500000**" in report
    assert "| 1 | 0.500000 |" in report
    assert "| Fall | 1 | 0 |" in report
    assert "only checkpoint retained" in report


def test_efficientnet_has_three_input_channels_and_two_classes() -> None:
    model = build_efficientnet()

    assert model.features[0][0].in_channels == 3
    assert model.classifier[1].out_features == 2
    model.eval()
    with torch.inference_mode():
        assert model(torch.zeros(1, 3, 625, 30)).shape == (1, 2)
