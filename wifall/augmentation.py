"""Deterministic, participant-safe augmentation of raw WiFall CSI recordings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import blake2b
from typing import Any, Literal

import numpy as np

from .loader import Recording

SUPPORTED_AUGMENTATIONS = frozenset(
    {
        "csi_noise",
        "temporal_jitter",
        "subcarrier_dropout",
        "window_perturbation",
    }
)
_SPLITS = frozenset({"train", "validation", "test"})
RecordEntry = tuple[str, str, Recording]
AugmentationFamily = Literal[
    "csi_noise",
    "temporal_jitter",
    "subcarrier_dropout",
    "window_perturbation",
]


@dataclass(frozen=True)
class CsiAugmentationConfig:
    """Configuration for deterministic raw-CSI augmentation.

    ``families`` are applied in listed order. Augmentation operates only on
    copies of recordings; labels and recording metadata remain unchanged.
    """

    families: tuple[str, ...] = ()
    copies_per_recording: int = 1
    seed: int = 42
    noise_scale: float = 0.01
    temporal_jitter: float = 0.05
    dropout_fraction: float = 0.05
    window_jitter: float = 0.05
    window_count: int = 4

    def __post_init__(self) -> None:
        families = tuple(self.families)
        object.__setattr__(self, "families", families)
        unsupported = set(families).difference(SUPPORTED_AUGMENTATIONS)
        if unsupported:
            raise ValueError(f"Unsupported augmentation family: {sorted(unsupported)}")
        if len(set(families)) != len(families):
            raise ValueError("families must not contain duplicates")
        if self.copies_per_recording < 0:
            raise ValueError("copies_per_recording must be non-negative")
        if self.noise_scale < 0:
            raise ValueError("noise_scale must be non-negative")
        if not 0 <= self.temporal_jitter < 1:
            raise ValueError("temporal_jitter must be in [0, 1)")
        if not 0 <= self.dropout_fraction < 1:
            raise ValueError("dropout_fraction must be in [0, 1)")
        if not 0 <= self.window_jitter < 1:
            raise ValueError("window_jitter must be in [0, 1)")
        if self.window_count < 1:
            raise ValueError("window_count must be at least 1")

    @property
    def enabled(self) -> bool:
        """Whether this configuration creates augmented examples."""

        return bool(self.families) and self.copies_per_recording > 0

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible configuration metadata."""

        return {
            "enabled": self.enabled,
            "families": list(self.families),
            "copies_per_recording": self.copies_per_recording,
            "seed": self.seed,
            "parameters": {
                "noise_scale": self.noise_scale,
                "temporal_jitter": self.temporal_jitter,
                "dropout_fraction": self.dropout_fraction,
                "window_jitter": self.window_jitter,
                "window_count": self.window_count,
            },
        }


@dataclass(frozen=True)
class AugmentationResult:
    """Augmented split records and metadata for downstream training artifacts."""

    splits: dict[str, tuple[RecordEntry, ...]]
    metadata: dict[str, Any]


def _recording_seed(recording: Recording, config: CsiAugmentationConfig, copy_index: int) -> int:
    digest = blake2b(
        recording.reference.recording_id.encode("utf-8"), digest_size=8
    ).digest()
    recording_number = int.from_bytes(digest, byteorder="little")
    return int(np.random.SeedSequence([config.seed, recording_number, copy_index]).generate_state(1)[0])


def _family_rng(
    recording: Recording,
    config: CsiAugmentationConfig,
    copy_index: int,
    family_index: int,
) -> np.random.Generator:
    seed = _recording_seed(recording, config, copy_index)
    sequence = np.random.SeedSequence([seed, family_index])
    return np.random.default_rng(sequence)


def _resample_rows(values: np.ndarray, source_positions: np.ndarray) -> np.ndarray:
    """Interpolate complex rows at bounded, monotonic source positions."""

    if values.shape[0] <= 1:
        return np.array(values, copy=True)
    indexes = np.arange(values.shape[0], dtype=np.float64)
    real = np.column_stack(
        [np.interp(source_positions, indexes, values[:, column].real) for column in range(values.shape[1])]
    )
    imaginary = np.column_stack(
        [np.interp(source_positions, indexes, values[:, column].imag) for column in range(values.shape[1])]
    )
    return real + 1j * imaginary


def _add_csi_noise(
    csi: np.ndarray, config: CsiAugmentationConfig, rng: np.random.Generator
) -> np.ndarray:
    if config.noise_scale == 0:
        return csi
    scale = np.median(np.abs(csi), axis=0)
    scale = np.maximum(scale, 1.0)
    noise = (
        rng.normal(size=csi.shape) + 1j * rng.normal(size=csi.shape)
    ) * (scale * config.noise_scale / np.sqrt(2.0))
    return csi + noise


def _apply_temporal_jitter(
    csi: np.ndarray, config: CsiAugmentationConfig, rng: np.random.Generator
) -> np.ndarray:
    if config.temporal_jitter == 0 or csi.shape[0] <= 1:
        return csi
    row_count = csi.shape[0]
    base = np.arange(row_count, dtype=np.float64)
    maximum_offset = config.temporal_jitter * (row_count - 1)
    positions = base + rng.uniform(-maximum_offset, maximum_offset, row_count)
    positions[0] = 0.0
    positions[-1] = float(row_count - 1)
    positions = np.maximum.accumulate(np.clip(positions, 0.0, row_count - 1))
    positions[-1] = float(row_count - 1)
    return _resample_rows(csi, positions)


def _drop_subcarriers(
    csi: np.ndarray, config: CsiAugmentationConfig, rng: np.random.Generator
) -> np.ndarray:
    if config.dropout_fraction == 0 or csi.shape[1] <= 1:
        return csi
    count = min(
        csi.shape[1] - 1,
        max(1, int(np.ceil(csi.shape[1] * config.dropout_fraction))),
    )
    dropped = rng.choice(csi.shape[1], size=count, replace=False)
    keep = np.ones(csi.shape[1], dtype=bool)
    keep[dropped] = False
    kept_indexes = np.flatnonzero(keep)
    output = np.array(csi, copy=True)
    for row in range(csi.shape[0]):
        output[row, dropped] = np.interp(
            dropped,
            kept_indexes,
            csi[row, keep].real,
        ) + 1j * np.interp(
            dropped,
            kept_indexes,
            csi[row, keep].imag,
        )
    return output


def _perturb_windows(
    csi: np.ndarray, config: CsiAugmentationConfig, rng: np.random.Generator
) -> np.ndarray:
    if config.window_jitter == 0 or csi.shape[0] <= 1:
        return csi
    window_count = min(config.window_count, csi.shape[0] - 1)
    if window_count < 1:
        return csi
    boundaries = np.linspace(0.0, csi.shape[0] - 1, window_count + 1)
    width = (csi.shape[0] - 1) / window_count
    offsets = rng.uniform(
        -config.window_jitter * width,
        config.window_jitter * width,
        window_count + 1,
    )
    offsets[0] = 0.0
    offsets[-1] = 0.0
    source_boundaries = np.clip(boundaries + offsets, 0.0, csi.shape[0] - 1)
    source_boundaries = np.maximum.accumulate(source_boundaries)
    source_boundaries[-1] = float(csi.shape[0] - 1)
    target_positions = np.arange(csi.shape[0], dtype=np.float64)
    source_positions = np.interp(target_positions, boundaries, source_boundaries)
    return _resample_rows(csi, source_positions)


def _copy_with_csi(recording: Recording, csi: np.ndarray) -> Recording:
    return Recording(
        reference=recording.reference,
        csi=np.asarray(csi, dtype=recording.csi.dtype),
        rssi=np.array(recording.rssi, copy=True),
        timestamps=recording.timestamps,
        sequences=np.array(recording.sequences, copy=True),
        source_activities=recording.source_activities,
    )


def augment_recording(
    recording: Recording,
    config: CsiAugmentationConfig,
    *,
    copy_index: int = 0,
) -> Recording:
    """Create one deterministic augmented copy of ``recording``."""

    if copy_index < 0:
        raise ValueError("copy_index must be non-negative")
    csi = np.array(recording.csi, copy=True)
    operations = {
        "csi_noise": _add_csi_noise,
        "temporal_jitter": _apply_temporal_jitter,
        "subcarrier_dropout": _drop_subcarriers,
        "window_perturbation": _perturb_windows,
    }
    for family_index, family in enumerate(config.families):
        csi = operations[family](
            csi,
            config,
            _family_rng(recording, config, copy_index, family_index),
        )
    return _copy_with_csi(recording, csi)


def augment_split_records(
    split_records: Mapping[str, Sequence[RecordEntry]],
    config: CsiAugmentationConfig,
) -> AugmentationResult:
    """Augment only ``train`` records and preserve other split records.

    Original training records remain first in the returned training split,
    followed by generated copies. Validation and test records are returned by
    reference and are never transformed.
    """

    unknown_splits = set(split_records).difference(_SPLITS)
    if unknown_splits:
        raise ValueError(f"unknown split: {sorted(unknown_splits)}")

    result_splits: dict[str, tuple[RecordEntry, ...]] = {
        split: tuple(records) for split, records in split_records.items()
    }
    generated_examples = {split: 0 for split in split_records}
    generated_by_family = {family: 0 for family in config.families}
    if config.enabled and "train" in split_records:
        generated: list[RecordEntry] = []
        for participant, activity, recording in split_records["train"]:
            for copy_index in range(config.copies_per_recording):
                generated.append(
                    (
                        participant,
                        activity,
                        augment_recording(recording, config, copy_index=copy_index),
                    )
                )
        result_splits["train"] = tuple(split_records["train"]) + tuple(generated)
        generated_examples["train"] = len(generated)
        generated_by_family = {
            family: len(generated) for family in config.families
        }

    metadata = config.as_dict() | {
        "source_split": "train_only",
        "source_examples": {split: len(records) for split, records in split_records.items()},
        "generated_examples": generated_examples,
        "generated_by_family": generated_by_family,
    }
    return AugmentationResult(splits=result_splits, metadata=metadata)
