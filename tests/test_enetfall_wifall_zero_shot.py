from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from enetfall.evaluate_wifall_zero_shot import (
    adapt_wifall_amplitude,
    load_wifall_records,
    parse_wifall_csi_field,
)


def test_wifall_adaptation_resamples_maps_and_replicates_pseudo_receivers() -> None:
    source = np.broadcast_to(np.arange(52, dtype=np.float64), (100, 52)).copy()

    adapted = adapt_wifall_amplitude(source)

    assert adapted.shape == (149, 3, 30)
    np.testing.assert_allclose(adapted[74, 0], np.linspace(0.0, 51.0, 30), atol=0.01)
    np.testing.assert_array_equal(adapted[:, 0], adapted[:, 1])
    np.testing.assert_array_equal(adapted[:, 1], adapted[:, 2])
    assert (adapted >= 0).all()


def test_adapter_rejects_invalid_wifi_csi_shape_and_values() -> None:
    with pytest.raises(ValueError, match="52 subcarriers"):
        adapt_wifall_amplitude(np.zeros((8, 51)))
    with pytest.raises(ValueError, match="non-finite"):
        adapt_wifall_amplitude(np.full((8, 52), np.nan))


def test_wifall_csi_parser_validates_104_real_imag_values() -> None:
    np.testing.assert_allclose(
        parse_wifall_csi_field(str([3, 4] * 52)),
        np.full(52, 5.0),
    )
    with pytest.raises(ValueError, match="104"):
        parse_wifall_csi_field("[1, 2]")


def test_wifall_loader_can_select_test_or_all_manifest_recordings(tmp_path) -> None:
    recordings = []
    for split, participant, activity, label in (
        ("train", "ID0", "fall", "fall"),
        ("train", "ID0", "walk", "non-fall"),
        ("test", "ID1", "fall", "fall"),
        ("test", "ID1", "walk", "non-fall"),
    ):
        relative = f"{participant}/{activity}/clip.csv"
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        with source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("data", "taget", "seq"))
            writer.writeheader()
            for sequence in range(4):
                writer.writerow({"data": str([3, 4] * 52), "taget": activity, "seq": sequence})
        recordings.append(
            {
                "split": split,
                "participant_id": participant,
                "recording_id": relative,
                "relative_path": relative,
                "target": label,
                "row_count": 4,
            }
        )
    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": {"name": "WiFall", "participants": 2, "recordings": 4},
                "split_config": {},
                "participants": {"train": ["ID0"], "test": ["ID1"]},
                "recordings": recordings,
            }
        ),
        encoding="utf-8",
    )

    all_records, all_features, _ = load_wifall_records(
        tmp_path, manifest, evaluation_scope="all"
    )
    test_records, test_features, _ = load_wifall_records(
        tmp_path, manifest, evaluation_scope="test"
    )

    assert len(all_records) == 4
    assert all_features.shape == (4, 216)
    assert len(test_records) == 2
    assert test_features.shape == (2, 216)
