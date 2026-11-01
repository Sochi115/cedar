from __future__ import annotations

import torch

from csi_bench.cnn_lstm import DeepCnnLstm


def test_cnn_lstm_keeps_downsampled_time_sequence() -> None:
    model = DeepCnnLstm(class_count=2, hidden_size=64).eval()
    sample = torch.randn(1, 1, 500, 232)

    with torch.inference_mode():
        sequence = model.encode_sequence(sample)
        logits = model(sample)

    assert sequence.shape == (1, 32, 256)
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()
