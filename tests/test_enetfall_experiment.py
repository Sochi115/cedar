from __future__ import annotations

import numpy as np

from enetfall.compare_models import extract_windowed_features, select_threshold


def test_windowed_features_keep_temporal_order_and_have_compact_shape() -> None:
    first = np.zeros((8, 3, 30), dtype=np.float64)
    first[4:] = 10.0
    second = first[::-1].copy()

    first_features = extract_windowed_features(first)
    second_features = extract_windowed_features(second)

    assert first_features.shape == (216,)
    assert not np.array_equal(first_features, second_features)


def test_robust_normalization_removes_per_channel_offset_and_scale() -> None:
    rng = np.random.default_rng(42)
    amplitude = rng.uniform(1.0, 5.0, size=(32, 3, 30))
    scale = rng.uniform(0.5, 3.0, size=(3, 30))
    offset = rng.uniform(2.0, 20.0, size=(3, 30))

    original = extract_windowed_features(amplitude, robust_normalize=True)
    shifted = extract_windowed_features(
        amplitude * scale[None, :, :] + offset[None, :, :],
        robust_normalize=True,
    )

    np.testing.assert_allclose(original, shifted, atol=1e-10)


def test_threshold_is_chosen_using_labels_and_scores_only() -> None:
    threshold = select_threshold(
        np.array(["non-fall", "non-fall", "fall", "fall"]),
        np.array([-2.0, -1.0, 0.2, 0.8]),
    )

    predictions = np.where(np.array([-2.0, -1.0, 0.2, 0.8]) >= threshold, "fall", "non-fall")
    np.testing.assert_array_equal(predictions, ["non-fall", "non-fall", "fall", "fall"])
