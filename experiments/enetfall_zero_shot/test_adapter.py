from __future__ import annotations

import numpy as np
import pytest
from scipy.io import savemat

from experiments.enetfall_zero_shot.adapter import (
    _read_mat_file,
    interpolate_subcarrier_order,
    split_receiver_streams,
)


def test_split_receiver_streams_preserves_interleaved_receiver_order() -> None:
    source = np.arange(2 * 4 * 90).reshape(2, 4, 90)

    result = split_receiver_streams(source)

    assert result.shape == (2, 4, 3, 30)
    np.testing.assert_array_equal(result[:, :, 0, :], source[:, :, 0::3])
    np.testing.assert_array_equal(result[:, :, 1, :], source[:, :, 1::3])
    np.testing.assert_array_equal(result[:, :, 2, :], source[:, :, 2::3])


def test_interpolate_subcarrier_order_keeps_time_and_order() -> None:
    source = np.stack([np.linspace(0.0, 1.0, 30), np.linspace(1.0, 2.0, 30)])

    result = interpolate_subcarrier_order(source)

    assert result.shape == (2, 52)
    np.testing.assert_allclose(result[0], np.linspace(0.0, 1.0, 52), atol=1e-6)
    np.testing.assert_allclose(result[1], np.linspace(1.0, 2.0, 52), atol=1e-6)


def test_interpolation_rejects_invalid_shape_and_values() -> None:
    with pytest.raises(ValueError, match=r"shape \(samples, 30\)"):
        interpolate_subcarrier_order(np.zeros((4, 29)))
    with pytest.raises(ValueError, match="non-finite"):
        interpolate_subcarrier_order(np.full((4, 30), np.nan))


def test_mat_event_adapts_to_three_wifall_recordings(tmp_path) -> None:
    path = tmp_path / "dataset_meeting_room.mat"
    csi = np.ones((2, 625, 90), dtype=np.float64)
    csi[..., 0::3] *= 1.0
    csi[..., 1::3] *= 2.0
    csi[..., 2::3] *= 3.0
    savemat(path, {"dataset_CSI_t": csi, "dataset_labels": np.asarray([[0], [1]])})

    events = list(_read_mat_file(path, "meeting_room_B"))
    assert [event.label for event in events] == ["non-fall", "fall"]
    assert events[0].amplitude.shape == (625, 3, 30)

    with pytest.raises(ValueError, match="Opt in"):
        events[0].to_wifall_recordings()

    recordings = events[1].to_wifall_recordings(
        allow_approximate_subcarrier_mapping=True
    )
    assert len(recordings) == 3
    assert {recording.csi.shape for recording in recordings} == {(420, 52)}
    assert all(recording.target == "fall" for recording in recordings)
    assert {recording.reference.recording_id.rsplit(":", 1)[-1] for recording in recordings} == {
        "rx0",
        "rx1",
        "rx2",
    }
    assert all(np.iscomplexobj(recording.csi) for recording in recordings)
    assert recordings[0].timestamps[0] < recordings[0].timestamps[-1]
