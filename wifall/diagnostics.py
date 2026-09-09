"""WiFall recording metadata adapter for generic evaluation diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from evaluation.diagnostics import duration_from_timestamps

from .loader import Recording

Record = tuple[str, str, Recording]


def _signal_quality(recording: Recording) -> dict[str, float | int] | None:
    values = np.asarray(recording.rssi, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return {
        "rssi_count": int(values.size),
        "rssi_min": float(values.min()),
        "rssi_max": float(values.max()),
        "rssi_mean": float(values.mean()),
        "rssi_std": float(values.std()),
    }


def recording_observations(records: Sequence[Record]) -> list[dict[str, Any]]:
    """Adapt WiFall records to the generic hard-example observation interface."""

    return [
        {
            "observation_id": recording.reference.recording_id,
            "group_id": participant,
            "activity": activity,
            "duration_seconds": duration_from_timestamps(recording.timestamps),
            "signal_quality": _signal_quality(recording),
        }
        for participant, activity, recording in records
    ]
