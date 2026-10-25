from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from wifall import (
    CausalWindowConfig,
    SplitConfig,
    build_causal_window_dataset,
    generate_split_manifest,
    load_causal_window_dataset,
    save_split_manifest,
)

HEADER = ["type", "seq", "timestamp", "taget", "rssi", "data"]


def _write_recording(root: Path, participant: str, activity: str) -> None:
    directory = root / participant / activity
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{activity}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        offset = 1000 if activity == "fall" else 0
        for row in range(8):
            writer.writerow(
                [
                    "CSI_DATA",
                    row,
                    f"2023-01-01 00:00:0{row}.000",
                    activity,
                    "-31",
                    str(list(range(offset + row, offset + row + 104))),
                ]
            )


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "WiFall"
    for participant in ("ID0", "ID1", "ID2"):
        _write_recording(root, participant, "fall")
        _write_recording(root, participant, "sit")
    manifest = generate_split_manifest(
        root, SplitConfig(seed=3, test_participants=1, validation_participants=1)
    )
    manifest_path = tmp_path / "split.json"
    save_split_manifest(manifest, manifest_path)
    return root, manifest_path


def test_causal_windows_stay_inside_recordings_and_splits(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    run = build_causal_window_dataset(
        root,
        manifest,
        tmp_path / "windows",
        CausalWindowConfig(window_samples=4, stride_samples=2),
    )
    loaded = load_causal_window_dataset(run.output_dir)

    assert loaded.amplitude.shape[1:] == (4, 52)
    assert loaded.metadata["windowing"]["causal"] is True
    assert loaded.metadata["labeling"]["weak_recording_labels"] is True
    assert set(loaded.metadata["counts"]) == {"train", "validation", "test"}
    assert all(start <= end for start, end in zip(loaded.start_sequences, loaded.end_sequences))
    assert set(loaded.participants).issubset({"ID0", "ID1", "ID2"})


def test_elapsed_window_and_row_fallback_are_persisted(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    run = build_causal_window_dataset(
        root,
        manifest,
        tmp_path / "elapsed-windows",
        CausalWindowConfig(window_samples=4, stride_samples=2, window_seconds=2.0),
    )
    loaded = load_causal_window_dataset(run.output_dir)

    assert loaded.metadata["config"]["window_seconds"] == 2.0
    assert "fallback" in loaded.metadata["windowing"]["timestamp_policy"]
    assert loaded.amplitude.shape[1:] == (4, 52)


def test_causal_window_dataset_is_deterministic(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    config = CausalWindowConfig(window_samples=4, stride_samples=2)
    first = build_causal_window_dataset(root, manifest, tmp_path / "first", config)
    second = build_causal_window_dataset(root, manifest, tmp_path / "second", config)

    with np.load(first.data_path) as left, np.load(second.data_path) as right:
        for key in left.files:
            np.testing.assert_array_equal(left[key], right[key])
