"""Adapt local ENetFall MAT clips to the WiFall model input contract.

The 30-to-52 subcarrier mapping is an explicit index-grid approximation, not
physical-frequency alignment. Callers must opt in before creating WiFall views.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

from wifall.loader import Recording, RecordingReference

MAT_VARIABLE = "dataset_CSI_t"
LABEL_VARIABLE = "dataset_labels"
MAT_FILES = (
    ("dataset_living_room.mat", "living_room_A"),
    ("dataset_meeting_room.mat", "meeting_room_B"),
    ("dataset_home_lab(L).mat", "home_lab_left"),
    ("dataset_home_lab(R).mat", "home_lab_right"),
    ("dataset_lecture_room.mat", "lecture_room_D"),
)
SOURCE_SAMPLE_RATE_HZ = 125
WIFALL_SAMPLE_RATE_HZ = 84
WIFALL_SUBCARRIER_COUNT = 52
RECEIVER_COUNT = 3
SOURCE_SUBCARRIER_COUNT = 30
EXPECTED_LABEL_COUNTS = {"non-fall": 436, "fall": 321}


@dataclass(frozen=True)
class EnetFallEvent:
    """One labeled five-second event with three receiver CSI streams."""

    event_id: str
    room: str
    label: str
    amplitude: np.ndarray  # samples x receiver x Intel subcarrier
    source_path: Path

    def to_wifall_recordings(
        self,
        *,
        allow_approximate_subcarrier_mapping: bool = False,
    ) -> tuple[Recording, ...]:
        """Create one WiFall-compatible recording per receiver antenna.

        ENetFall contains amplitudes rather than complex CSI. Real-only complex
        values preserve amplitude for WiFall extractors, which all apply abs();
        phase cannot be reconstructed. Samples are resampled from 125 Hz to
        approximately WiFall's measured 84 Hz row rate to preserve LSTM window
        duration and adjacent-difference statistics.
        """

        if not allow_approximate_subcarrier_mapping:
            raise ValueError(
                "ENetFall has 30 Intel 5300 subcarriers; WiFall models require 52. "
                "Opt in with allow_approximate_subcarrier_mapping=True to use "
                "linear interpolation by subcarrier order. Physical frequency "
                "alignment is not verified."
            )

        recordings = []
        for receiver in range(RECEIVER_COUNT):
            values = interpolate_subcarrier_order(self.amplitude[:, receiver, :])
            values = resample_poly(
                values,
                up=WIFALL_SAMPLE_RATE_HZ,
                down=SOURCE_SAMPLE_RATE_HZ,
                axis=0,
            )
            values = np.maximum(values, 0.0).astype(np.float32, copy=False)
            row_count = values.shape[0]
            activity = self.label
            recording_id = f"{self.event_id}:rx{receiver}"
            reference = RecordingReference(
                recording_id=recording_id,
                relative_path=recording_id,
                participant_id="unknown",
                activity=activity,
                path=self.source_path,
            )
            start = datetime(2000, 1, 1, tzinfo=UTC)
            timestamps = tuple(
                (start + timedelta(seconds=index / WIFALL_SAMPLE_RATE_HZ)).isoformat()
                for index in range(row_count)
            )
            recordings.append(
                Recording(
                    reference=reference,
                    csi=values.astype(np.complex64),
                    rssi=np.empty(0, dtype=np.float64),
                    timestamps=timestamps,
                    sequences=np.arange(row_count, dtype=np.int64),
                    source_activities=(activity,) * row_count,
                )
            )
        return tuple(recordings)


def split_receiver_streams(csi: np.ndarray) -> np.ndarray:
    """Convert ENetFall interleaved 90-stream columns to N x T x 3 x 30."""

    values = np.asarray(csi)
    if values.ndim != 3 or values.shape[-1] != RECEIVER_COUNT * SOURCE_SUBCARRIER_COUNT:
        raise ValueError("CSI must have shape (events, samples, 90)")
    return values.reshape(*values.shape[:2], SOURCE_SUBCARRIER_COUNT, RECEIVER_COUNT).transpose(
        0, 1, 3, 2
    )


def interpolate_subcarrier_order(amplitude: np.ndarray) -> np.ndarray:
    """Interpolate one receiver stream from 30 to 52 ordered bins.

    This approximate mapping preserves order and endpoints only. It does not
    claim Intel 5300 bins coincide with ESP32-S3 WiFall frequency bins.
    """

    values = np.asarray(amplitude, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != SOURCE_SUBCARRIER_COUNT:
        raise ValueError("Receiver stream must have shape (samples, 30)")
    if not np.isfinite(values).all():
        raise ValueError("Receiver stream contains non-finite amplitudes")
    source_grid = np.linspace(0.0, 1.0, SOURCE_SUBCARRIER_COUNT)
    target_grid = np.linspace(0.0, 1.0, WIFALL_SUBCARRIER_COUNT)
    return np.stack(
        [np.interp(target_grid, source_grid, row) for row in values], axis=0
    ).astype(np.float32, copy=False)


def _read_mat_file(path: Path, room: str) -> Iterator[EnetFallEvent]:
    if not path.is_file():
        raise FileNotFoundError(f"ENetFall MAT file not found: {path}")
    data = loadmat(path, variable_names=[MAT_VARIABLE, LABEL_VARIABLE])
    if MAT_VARIABLE not in data or LABEL_VARIABLE not in data:
        raise ValueError(f"{path.name} must contain {MAT_VARIABLE} and {LABEL_VARIABLE}")

    csi = np.asarray(data[MAT_VARIABLE])
    labels = np.asarray(data[LABEL_VARIABLE]).reshape(-1)
    expected_shape = (labels.size, 625, RECEIVER_COUNT * SOURCE_SUBCARRIER_COUNT)
    if csi.shape != expected_shape:
        raise ValueError(f"{path.name} has CSI shape {csi.shape}; expected {expected_shape}")
    if not np.issubdtype(csi.dtype, np.number) or not np.isfinite(csi).all():
        raise ValueError(f"{path.name} CSI must contain finite numeric amplitudes")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"{path.name} labels must be 0 (non-fall) or 1 (fall)")

    streams = split_receiver_streams(csi)
    for index, raw_label in enumerate(labels):
        label = "fall" if int(raw_label) == 1 else "non-fall"
        amplitude = np.maximum(streams[index], 0.0).astype(np.float32, copy=True)
        yield EnetFallEvent(
            event_id=f"{path.stem}:{index:04d}",
            room=room,
            label=label,
            amplitude=amplitude,
            source_path=path,
        )


def iter_enetfall_events(dataset_root: str | Path) -> Iterator[EnetFallEvent]:
    """Yield labeled events from the five local ENetFall MAT files."""

    root = Path(dataset_root).expanduser().resolve()
    for filename, room in MAT_FILES:
        yield from _read_mat_file(root / filename, room)


def validate_dataset(dataset_root: str | Path) -> dict[str, object]:
    """Check local MAT files against published total and per-room counts."""

    total_labels: Counter[str] = Counter()
    rooms: dict[str, Counter[str]] = {}
    event_count = 0
    for event in iter_enetfall_events(dataset_root):
        total_labels[event.label] += 1
        rooms.setdefault(event.room, Counter())[event.label] += 1
        event_count += 1

    counts = dict(sorted(total_labels.items()))
    expected_rooms = {
        "living_room_A": {"fall": 91, "non-fall": 157},
        "meeting_room_B": {"fall": 58, "non-fall": 76},
        "home_lab_left": {"fall": 40, "non-fall": 45},
        "home_lab_right": {"fall": 36, "non-fall": 49},
        "lecture_room_D": {"fall": 96, "non-fall": 109},
    }
    actual_rooms = {
        room: dict(sorted(room_counts.items())) for room, room_counts in sorted(rooms.items())
    }
    if counts != EXPECTED_LABEL_COUNTS:
        raise ValueError(f"Unexpected ENetFall label counts: {counts}")
    if actual_rooms != expected_rooms:
        raise ValueError(f"Unexpected ENetFall room counts: {actual_rooms}")

    return {
        "events": event_count,
        "labels": counts,
        "rooms": actual_rooms,
        "source_shape_per_event": [625, 3, 30],
        "source_sample_rate_hz": SOURCE_SAMPLE_RATE_HZ,
        "wifall_view_shape_per_receiver": [420, WIFALL_SUBCARRIER_COUNT],
        "wifall_view_sample_rate_hz": WIFALL_SAMPLE_RATE_HZ,
        "subcarrier_mapping": "approximate linear interpolation by index; not frequency-calibrated",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default="data/raw/ENetFall",
        help="directory containing the five ENetFall MAT files",
    )
    args = parser.parse_args()
    print(json.dumps(validate_dataset(args.dataset_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
