"""Backward-compatible Random Forest adapter for generic diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from evaluation.diagnostics import analyze_hard_examples

from .diagnostics import Record, recording_observations


def analyze_random_forest_errors(
    evaluation: Mapping[str, Any],
    records: Sequence[Record],
    *,
    run_id: str,
    split_name: str,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Analyze Random Forest errors through the generic diagnostics seam."""

    return analyze_hard_examples(
        evaluation,
        recording_observations(records),
        run_id=run_id,
        split_name=split_name,
        threshold=threshold,
    )
