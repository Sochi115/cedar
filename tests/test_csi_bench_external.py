from __future__ import annotations

import numpy as np
import pytest
import torch

import csi_bench.evaluate_external as external
from csi_bench.evaluate_external import (
    LABELS,
    _BinaryClassifier,
    _combined_report,
    _csi_features,
    _csi_image,
)
from csi_bench.evaluate_external_cnn import _CnnClassifier
from evaluation.core import evaluate_classifier


class _Estimator:
    classes_ = np.asarray([0, 1])

    def predict(self, features):
        return np.asarray([0, 1])[: len(features)]

    def predict_proba(self, features):
        return np.tile([0.8, 0.2], (len(features), 1))


def test_external_adapter_matches_csi_bench_feature_shape() -> None:
    features = _csi_features(np.arange(80 * 52, dtype=float).reshape(80, 52))

    assert features.shape == (192,)
    assert _csi_image(np.ones((8, 52))).shape == (500, 232)
    assert np.isfinite(features).all()
    assert _csi_features(np.ones((8, 52))).shape == (192,)
    with pytest.raises(ValueError, match="finite time-by-feature"):
        _csi_features(np.full((8, 52), np.nan))


class _FixtureCnn(torch.nn.Module):
    def forward(self, values):
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], device=values.device)
        return logits[: len(values)]


def test_cnn_classifier_reorders_class_probabilities() -> None:
    model = _CnnClassifier(_FixtureCnn(), ["Fall", "Nonfall"], torch.device("cpu"))
    predictions = model.predict(np.zeros((2, 500, 232), dtype=np.float32))

    np.testing.assert_array_equal(predictions, ["fall", "non-fall"])
    np.testing.assert_allclose(
        model.predict_proba(np.zeros((2, 500, 232), dtype=np.float32))[:, 1],
        torch.softmax(torch.tensor([[2.0, 0.0], [0.0, 2.0]]), dim=1)[:, 0].numpy(),
    )


def test_binary_classifier_reorders_csi_classes_to_fall_labels() -> None:
    model = _BinaryClassifier(
        {"estimator": _Estimator(), "class_names": ["Fall", "Nonfall"]}
    )

    np.testing.assert_array_equal(model.classes_, LABELS)
    np.testing.assert_array_equal(
        model.predict(np.zeros((2, 192))), ["fall", "non-fall"]
    )
    np.testing.assert_allclose(model.predict_proba(np.zeros((2, 192)))[0], [0.2, 0.8])


def test_full_dataset_scope_includes_train_and_test_events(monkeypatch) -> None:
    events = [
        {
            "event_id": room,
            "room": room,
            "label": "fall",
            "amplitude": np.zeros((2, 3, 30)),
        }
        for room in ("living_room_A", "lecture_room_D")
    ]
    monkeypatch.setattr(external, "iter_enetfall_events", lambda _: iter(events))
    monkeypatch.setattr(external, "_csi_features", lambda _: np.zeros(192))

    full_ids, _, _ = external._enetfall_test("unused", "all")
    test_ids, _, _ = external._enetfall_test("unused", "test")

    assert full_ids == ["living_room_A", "lecture_room_D"]
    assert test_ids == ["lecture_room_D"]


def test_combined_report_contains_both_confusion_matrices() -> None:
    model = _BinaryClassifier(
        {"estimator": _Estimator(), "class_names": ["Fall", "Nonfall"]}
    )
    results = [
        evaluate_classifier(
            model,
            np.zeros((2, 192)),
            ["fall", "non-fall"],
            label_order=LABELS,
            positive_label="fall",
            run_id=f"{dataset}-test",
            split_name="all",
            dataset_metadata={"name": dataset},
        )
        for dataset in ("ENetFall", "WiFall")
    ]

    report = _combined_report(results)

    assert "ENetFall full dataset" in report
    assert "WiFall full dataset" in report
    assert "including source train and validation partitions" in report
    assert report.count("True \\ Predicted") == 2
    assert "index-interpolated" in report
