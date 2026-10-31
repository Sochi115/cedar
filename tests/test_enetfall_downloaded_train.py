from __future__ import annotations

import torch

from enetfall.train_downloaded import build_model


def test_downloaded_model_head_accepts_two_classes() -> None:
    model = build_model(pretrained=False).eval()

    assert model.classifier[-1].out_features == 2
    with torch.inference_mode():
        assert model(torch.zeros(1, 3, 64, 30)).shape == (1, 2)
