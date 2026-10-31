from __future__ import annotations

import numpy as np
import torch

import enetfall.train_cnn as cnn
from enetfall.train_cnn import CompactCsiCnn, normalize_event, render_report


def test_compact_cnn_accepts_native_enetfall_tensor() -> None:
    model = CompactCsiCnn().eval()
    with torch.inference_mode():
        logits = model(torch.zeros(2, 3, 625, 30))
    assert logits.shape == (2, 2)
    assert sum(parameter.numel() for parameter in model.parameters()) < 100_000


def test_load_splits_uses_stratified_random_80_20_with_inner_validation(
    monkeypatch,
) -> None:
    amplitude = np.ones((625, 3, 30), dtype=np.float32)
    events = [
        {
            "event_id": f"event-{index}",
            "room": "room_a",
            "label": label,
            "amplitude": amplitude,
        }
        for index, label in enumerate(["fall", "non-fall"] * 10)
    ]
    monkeypatch.setattr(cnn, "iter_enetfall_events", lambda _: iter(events))
    monkeypatch.setattr(cnn, "EXPECTED_ROOM_COUNTS", {"room_a": {"fall": 10, "non-fall": 10}})

    splits = cnn.load_splits("unused-root", seed=42)

    assert {name: len(split["records"]) for name, split in splits.items()} == {
        "train": 12,
        "validation": 4,
        "test": 4,
    }
    assert all(set(split["labels"].tolist()) == {0, 1} for split in splits.values())


def test_zscore_normalization_has_native_input_shape_and_finite_values() -> None:
    amplitude = np.random.default_rng(7).uniform(0, 10, size=(625, 3, 30)).astype(np.float32)

    normalized = normalize_event(amplitude)

    assert normalized.shape == (3, 625, 30)
    assert np.isfinite(normalized).all()


def test_report_is_standalone_and_contains_losses_confusion_and_scores() -> None:
    metrics = {
        "support": 2,
        "balanced_accuracy": 0.5,
        "roc_auc": 0.5,
        "confusion_matrix_nonfall_fall": [[1, 0], [1, 0]],
        "score_distribution_by_actual_label": {
            "non-fall": {"count": 1, "mean": 0.2, "median": 0.2, "min": 0.2, "max": 0.2},
            "fall": {"count": 1, "mean": 0.3, "median": 0.3, "min": 0.3, "max": 0.3},
        },
    }
    result = {
        "training": {
            "device": "cpu",
            "epochs_completed": 1,
            "best_epoch": 1,
            "best_validation_loss": 0.5,
            "batch_size": 8,
            "learning_rate": 1e-4,
            "weight_decay": 1e-3,
            "loss_history": [{"epoch": 1, "train_loss": 0.6, "validation_loss": 0.5}],
        },
        "threshold": 0.25,
        "batch_comparison": [
            {
                "batch_size": 8,
                "best_epoch": 1,
                "best_validation_loss": 0.5,
                "validation": {"balanced_accuracy": 0.5, "roc_auc": 0.5},
            }
        ],
        "split": {
            "counts": {
                "train": {"size": 1, "fall": 1, "non_fall": 0, "rooms": {"room_a": 1}},
                "validation": {
                    "size": 1,
                    "fall": 0,
                    "non_fall": 1,
                    "rooms": {"room_b": 1},
                },
                "test": {"size": 1, "fall": 1, "non_fall": 0, "rooms": {"room_c": 1}},
            }
        },
        "validation": metrics,
        "test": metrics,
    }

    report = render_report(result)

    assert report.startswith("<!doctype html>")
    assert "0.500000" in report
    assert "Confusion matrix" in report
    assert "Fall-score distribution" in report
    assert "<table>" in report
