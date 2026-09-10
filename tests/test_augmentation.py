from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from wifall.augmentation import (
    SUPPORTED_AUGMENTATIONS,
    CsiAugmentationConfig,
    augment_recording,
    augment_split_records,
)
from wifall.loader import Recording, RecordingReference


def _recording(participant: str = "ID0", activity: str = "fall") -> Recording:
    rows = 24
    subcarriers = 6
    values = np.arange(rows * subcarriers, dtype=np.float64).reshape(rows, subcarriers)
    csi = values**2 + 1j * (values + 100.0) ** 2
    reference = RecordingReference(
        recording_id=f"{participant}/{activity}/recording.csv",
        relative_path=f"{participant}/{activity}/recording.csv",
        participant_id=participant,
        activity=activity,
        path=Path("recording.csv"),
    )
    return Recording(
        reference=reference,
        csi=csi,
        rssi=np.linspace(-40.0, -30.0, rows),
        timestamps=tuple(f"t{index}" for index in range(rows)),
        sequences=np.arange(rows),
        source_activities=(activity,) * rows,
    )


def _config(*families: str, seed: int = 7) -> CsiAugmentationConfig:
    return CsiAugmentationConfig(
        families=families,
        copies_per_recording=1,
        seed=seed,
        window_count=4,
    )


@pytest.mark.parametrize("family", sorted(SUPPORTED_AUGMENTATIONS))
def test_each_augmentation_is_deterministic_and_preserves_recording_metadata(
    family: str,
) -> None:
    source = _recording()
    original_csi = source.csi.copy()
    config = _config(family)

    first = augment_recording(source, config)
    second = augment_recording(source, config)

    np.testing.assert_array_equal(first.csi, second.csi)
    assert first.csi.shape == source.csi.shape
    assert first.csi.dtype == source.csi.dtype
    assert first.target == source.target
    assert first.reference == source.reference
    assert first.timestamps == source.timestamps
    np.testing.assert_array_equal(source.csi, original_csi)
    assert not np.array_equal(first.csi, source.csi)


def test_combined_augmentation_is_reproducible_and_changes_complex_csi() -> None:
    source = _recording(activity="sit")
    config = _config(*SUPPORTED_AUGMENTATIONS, seed=19)

    augmented = augment_recording(source, config)

    assert augmented.target == "non-fall"
    assert np.iscomplexobj(augmented.csi)
    assert np.isfinite(augmented.csi.real).all()
    assert np.isfinite(augmented.csi.imag).all()
    np.testing.assert_array_equal(augmented.sequences, source.sequences)
    np.testing.assert_allclose(augmented.rssi, source.rssi)
    assert not np.array_equal(augmented.csi, source.csi)
    np.testing.assert_array_equal(
        augmented.csi,
        augment_recording(source, config).csi,
    )


def test_split_augmentation_only_generates_training_examples() -> None:
    train = [("ID0", "fall", _recording("ID0", "fall"))]
    validation = [("ID1", "sit", _recording("ID1", "sit"))]
    test = [("ID2", "fall", _recording("ID2", "fall"))]

    result = augment_split_records(
        {"train": train, "validation": validation, "test": test},
        _config("csi_noise", seed=31),
    )

    assert len(result.splits["train"]) == 2
    assert len(result.splits["validation"]) == 1
    assert len(result.splits["test"]) == 1
    assert result.metadata["generated_examples"] == {
        "train": 1,
        "validation": 0,
        "test": 0,
    }
    assert result.metadata["generated_by_family"] == {"csi_noise": 1}
    assert result.metadata["source_split"] == "train_only"
    assert result.splits["validation"][0][2] is validation[0][2]
    assert result.splits["test"][0][2] is test[0][2]


def test_multiple_copies_have_unique_deterministic_data_and_counts() -> None:
    source = [("ID0", "fall", _recording())]
    config = replace(_config("temporal_jitter"), copies_per_recording=3)

    first = augment_split_records({"train": source}, config)
    second = augment_split_records({"train": source}, config)

    assert len(first.splits["train"]) == 4
    assert first.metadata["generated_examples"]["train"] == 3
    np.testing.assert_array_equal(
        np.stack([item[2].csi for item in first.splits["train"][1:]]),
        np.stack([item[2].csi for item in second.splits["train"][1:]]),
    )
    assert len(
        {
            item[2].csi.tobytes()
            for item in first.splits["train"][1:]
        }
    ) == 3


def test_result_metadata_is_json_compatible() -> None:
    result = augment_split_records(
        {"train": [("ID0", "fall", _recording())]},
        _config("csi_noise"),
    )

    json.dumps(result.metadata)
    assert result.metadata["parameters"]["noise_scale"] == 0.01


def test_disabled_augmentation_returns_original_split_and_zero_counts() -> None:
    records = {"train": [("ID0", "fall", _recording())], "test": []}

    result = augment_split_records(records, CsiAugmentationConfig())

    assert result.splits["train"][0][2] is records["train"][0][2]
    assert result.metadata["enabled"] is False
    assert result.metadata["generated_examples"] == {"train": 0, "test": 0}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"families": ("unknown",)}, "Unsupported augmentation family"),
        ({"copies_per_recording": -1}, "copies_per_recording"),
        ({"noise_scale": -0.1}, "noise_scale"),
        ({"temporal_jitter": 1.1}, "temporal_jitter"),
        ({"dropout_fraction": 1.0}, "dropout_fraction"),
        ({"window_count": 0}, "window_count"),
    ],
)
def test_augmentation_config_rejects_invalid_values(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CsiAugmentationConfig(**kwargs)


def test_split_augmentation_rejects_unknown_split() -> None:
    with pytest.raises(ValueError, match="unknown split"):
        augment_split_records({"holdout": []}, _config("csi_noise"))
