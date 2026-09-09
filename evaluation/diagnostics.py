"""Model-independent diagnostics for classification evaluation artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, TypedDict

import numpy as np


class HardExampleObservation(TypedDict, total=False):
    """Metadata for one evaluation observation.

    ``observation_id`` is required. Other fields are optional so models and
    datasets can provide only metadata they actually have.
    """

    observation_id: str
    group_id: str
    activity: str
    duration_seconds: float | None
    signal_quality: Mapping[str, Any] | None


_DURATION_BUCKETS = (
    ("unknown", None, None),
    ("under_1s", 0.0, 1.0),
    ("1_to_5s", 1.0, 5.0),
    ("5_to_10s", 5.0, 10.0),
    ("10_to_30s", 10.0, 30.0),
    ("30s_or_more", 30.0, None),
)


def _number_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    numbers = np.asarray(values, dtype=float)
    return {
        "count": int(numbers.size),
        "min": float(numbers.min()),
        "max": float(numbers.max()),
        "mean": float(numbers.mean()),
    }


def _score(value: Any) -> float | None:
    if value is None:
        return None
    values = np.asarray(value, dtype=float)
    if values.size != 1:
        return None
    number = float(values.reshape(-1)[0])
    return number if np.isfinite(number) else None


def _duration_bucket(duration: Any) -> str:
    if duration is None:
        return "unknown"
    try:
        number = float(duration)
    except (TypeError, ValueError):
        return "unknown"
    for name, lower, upper in _DURATION_BUCKETS[1:]:
        if (lower is None or number >= lower) and (upper is None or number < upper):
            return name
    return "unknown"


def _group_summary(examples: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        grouped.setdefault(str(example[key]), []).append(example)
    return {
        group: {
            "count": len(group_examples),
            "observation_ids": [item["observation_id"] for item in group_examples],
            "examples": group_examples,
        }
        for group, group_examples in sorted(grouped.items())
    }


def _duration_patterns(examples: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in _DURATION_BUCKETS}
    for example in examples:
        buckets[example["duration_bucket"]].append(example)
    return {
        name: {
            "count": len(bucket_examples),
            "observation_ids": [item["observation_id"] for item in bucket_examples],
        }
        for name, bucket_examples in buckets.items()
    }


def _signal_quality_pattern(examples: list[dict[str, Any]]) -> dict[str, Any]:
    available = [
        example["signal_quality"]
        for example in examples
        if isinstance(example["signal_quality"], Mapping)
    ]
    field_values: dict[str, list[float]] = {}
    for quality in available:
        for field, value in quality.items():
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and np.isfinite(value)
            ):
                field_values.setdefault(str(field), []).append(float(value))
    return {
        "recordings": len(examples),
        "available_recordings": len(available),
        "missing_recordings": len(examples) - len(available),
        "fields": {
            field: _number_summary(values)
            for field, values in sorted(field_values.items())
        },
    }


def analyze_hard_examples(
    evaluation: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    split_name: str,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Analyze finalized classification errors without knowing model type.

    Callers provide one evaluation mapping from ``evaluate_classifier`` and
    aligned observation metadata. This function never fits a model or chooses
    features, parameters, calibration, or thresholds.
    """

    y_true = list(evaluation.get("y_true", []))
    y_pred = list(evaluation.get("y_pred", []))
    score_source = evaluation.get("positive_scores")
    if score_source is None:
        score_source = evaluation.get("scores")
    raw_scores = list(score_source) if score_source is not None else []
    if len(y_true) != len(y_pred) or len(y_true) != len(observations):
        raise ValueError("Evaluation labels and observations must have equal lengths")
    if raw_scores and len(raw_scores) != len(observations):
        raise ValueError("Evaluation scores and observations must have equal lengths")

    observation_ids = [str(observation.get("observation_id", "")) for observation in observations]
    if not all(observation_ids):
        raise ValueError("Every observation must have an observation_id")
    artifact_ids = evaluation.get("observation_ids")
    if artifact_ids is not None and list(artifact_ids) != observation_ids:
        raise ValueError("Evaluation observation IDs do not match observation order")

    false_negatives: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []
    for index, (true_label, predicted_label, observation) in enumerate(
        zip(y_true, y_pred, observations)
    ):
        if true_label == "fall" and predicted_label == "non-fall":
            error_type = "false_negative"
        elif true_label == "non-fall" and predicted_label == "fall":
            error_type = "false_positive"
        else:
            continue
        duration = observation.get("duration_seconds")
        try:
            duration = float(duration) if duration is not None else None
        except (TypeError, ValueError):
            duration = None
        signal_quality = observation.get("signal_quality")
        if not isinstance(signal_quality, Mapping):
            signal_quality = None
        example = {
            "observation_id": observation_ids[index],
            "group_id": str(observation.get("group_id", "ungrouped")),
            "activity": str(observation.get("activity", "unknown")),
            "target": true_label,
            "prediction": predicted_label,
            "score": _score(raw_scores[index]) if raw_scores else None,
            "duration_seconds": duration,
            "duration_bucket": _duration_bucket(duration),
            "signal_quality": dict(signal_quality) if signal_quality is not None else None,
        }
        (false_negatives if error_type == "false_negative" else false_positives).append(example)

    false_negatives.sort(key=lambda item: item["observation_id"])
    false_positives.sort(key=lambda item: item["observation_id"])
    all_errors = false_negatives + false_positives
    score_values = [item["score"] for item in all_errors if item["score"] is not None]
    missed_score_values = [item["score"] for item in false_negatives if item["score"] is not None]
    missed_examples = sorted(
        false_negatives,
        key=lambda item: (
            item["score"] is None,
            item["score"] if item["score"] is not None else 0.0,
            item["observation_id"],
        ),
    )

    def pattern(examples: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(examples),
            "by_group": _group_summary(examples, "group_id"),
            "by_activity": _group_summary(examples, "activity"),
            "duration_seconds": _number_summary(
                [item["duration_seconds"] for item in examples if item["duration_seconds"] is not None]
            ),
            "duration_buckets": _duration_patterns(examples),
            "signal_quality": _signal_quality_pattern(examples),
            "observation_examples": examples,
        }

    return {
        "purpose": "post_evaluation_diagnostic_only",
        "selection_excluded": True,
        "source_run_id": run_id,
        "source_evaluation_split": split_name,
        "source_observation_ids": observation_ids,
        "decision_threshold": threshold,
        "error_counts": {
            "false_negatives": len(false_negatives),
            "false_positives": len(false_positives),
        },
        "false_negatives": pattern(false_negatives),
        "false_positives": pattern(false_positives),
        "error_score_range": _number_summary(score_values),
        "missed_fall_score_range": _number_summary(missed_score_values),
        "missed_fall_examples": missed_examples,
        "all_error_observation_examples": all_errors,
    }


def duration_from_timestamps(timestamps: Sequence[str]) -> float | None:
    """Return timestamp duration for adapters with timestamp metadata."""

    try:
        start = datetime.fromisoformat(timestamps[0])
        end = datetime.fromisoformat(timestamps[-1])
    except (IndexError, TypeError, ValueError):
        return None
    duration = (end - start).total_seconds()
    return duration if duration >= 0 else None
