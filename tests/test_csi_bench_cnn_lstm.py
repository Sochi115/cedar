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
    cnn_dropouts = [
        layer for layer in model.cnn.modules() if isinstance(layer, torch.nn.Dropout2d)
    ]
    assert len(cnn_dropouts) == 8
    assert all(layer.p == 0.1 for layer in cnn_dropouts)
    assert model.sequence_dropout.p == 0.25
    assert model.head_dropout.p == 0.4
