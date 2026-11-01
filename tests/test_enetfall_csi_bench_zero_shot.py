import numpy as np
import pytest

from enetfall.evaluate_csi_bench_zero_shot import adapt_csi_bench_amplitude


def test_adapter_maps_subcarriers_and_replicates_pseudo_receivers():
    values = np.broadcast_to(np.arange(232, dtype=np.float64), (12, 232)).copy()

    adapted = adapt_csi_bench_amplitude(values)

    assert adapted.shape == (12, 3, 30)
    np.testing.assert_allclose(adapted[0, 0], np.linspace(0, 231, 30))
    np.testing.assert_array_equal(adapted[:, 0], adapted[:, 1])
    np.testing.assert_array_equal(adapted[:, 1], adapted[:, 2])
    assert adapt_csi_bench_amplitude(np.zeros((12, 64))).shape == (12, 3, 30)


def test_adapter_rejects_invalid_shape_and_non_finite_values():
    with pytest.raises(ValueError, match="subcarriers"):
        adapt_csi_bench_amplitude(np.zeros((12, 0)))
    with pytest.raises(ValueError, match="non-finite"):
        adapt_csi_bench_amplitude(np.full((12, 232), np.nan))
