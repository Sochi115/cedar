from __future__ import annotations

import numpy as np
import pytest
from scipy.io import savemat

from enetfall.train_svm import (
    MAT_FILES,
    ROOM_SPLITS,
    extract_event_features,
    iter_enetfall_events,
)


def test_event_features_cover_each_receiver_and_subcarrier() -> None:
    samples = np.arange(625, dtype=np.float64)[:, None, None]
    amplitude = np.broadcast_to(samples, (625, 3, 30)).copy()

    features = extract_event_features(amplitude)

    assert features.shape == (450,)
    np.testing.assert_allclose(features[:90], 312.0)
    np.testing.assert_allclose(features[90:180], np.std(np.arange(625)))
    np.testing.assert_allclose(features[360:], 0.0)


def test_feature_extraction_rejects_invalid_event_arrays() -> None:
    with pytest.raises(ValueError, match="shape"):
        extract_event_features(np.zeros((625, 2, 30)))
    with pytest.raises(ValueError, match="non-finite"):
        extract_event_features(np.full((625, 3, 30), np.nan))


def test_environment_splits_keep_home_lab_recordings_together() -> None:
    assert ROOM_SPLITS == {
        "living_room_A": "train",
        "meeting_room_B": "train",
        "home_lab_left": "validation",
        "home_lab_right": "validation",
        "lecture_room_D": "test",
    }


def test_mat_loader_deinterleaves_receiver_channels(tmp_path) -> None:
    for filename, _room in MAT_FILES:
        csi = np.zeros((1, 625, 90), dtype=np.float32)
        csi[..., 0::3] = 1
        csi[..., 1::3] = 2
        csi[..., 2::3] = 3
        savemat(tmp_path / filename, {"dataset_CSI_t": csi, "dataset_labels": [[1]]})

    events = list(iter_enetfall_events(tmp_path))

    assert len(events) == 5
    assert events[0]["label"] == "fall"
    np.testing.assert_array_equal(events[0]["amplitude"][0, :, 0], [1, 2, 3])
