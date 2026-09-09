from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluation import analyze_hard_examples, render_html_report
from wifall.loader import Recording, RecordingReference
from wifall.random_forest_diagnostics import analyze_random_forest_errors


def _record(
    tmp_path: Path,
    participant: str,
    activity: str,
    name: str,
    *,
    rssi: list[float] | None = None,
    duration_seconds: int = 2,
) -> Recording:
    path = tmp_path / participant / activity / f"{name}.csv"
    reference = RecordingReference(
        recording_id=f"{participant}/{activity}/{name}.csv",
        relative_path=f"{participant}/{activity}/{name}.csv",
        participant_id=participant,
        activity=activity,
        path=path,
    )
    return Recording(
        reference=reference,
        csi=np.ones((3, 52), dtype=np.complex128),
        rssi=np.asarray([] if rssi is None else rssi, dtype=float),
        timestamps=(
            "2023-01-01T00:00:00",
            f"2023-01-01T00:00:{duration_seconds:02d}",
        ),
        sequences=np.arange(3),
        source_activities=(activity,) * 3,
    )


def _fixture(tmp_path: Path) -> tuple[dict[str, object], list[tuple[str, str, Recording]]]:
    recordings = [
        ("ID1", "fall", _record(tmp_path, "ID1", "fall", "missed", rssi=[-45, -44])),
        ("ID1", "sit", _record(tmp_path, "ID1", "sit", "alarm", rssi=[-50, -51])),
        ("ID2", "walk", _record(tmp_path, "ID2", "walk", "alarm", rssi=None, duration_seconds=8)),
        ("ID2", "fall", _record(tmp_path, "ID2", "fall", "detected", rssi=[-40, -41])),
    ]
    evaluation = {
        "y_true": ["fall", "non-fall", "non-fall", "fall"],
        "y_pred": ["non-fall", "fall", "fall", "fall"],
        "scores": [0.2, 0.8, 0.7, 0.9],
        "observation_ids": [recording.reference.recording_id for _, _, recording in recordings],
    }
    return evaluation, recordings


def test_diagnostics_group_errors_and_report_missed_falls(tmp_path: Path) -> None:
    evaluation, records = _fixture(tmp_path)

    result = analyze_random_forest_errors(
        evaluation,
        records,
        run_id="rf-run",
        split_name="test",
        threshold=0.5,
    )

    assert result["purpose"] == "post_evaluation_diagnostic_only"
    assert result["selection_excluded"] is True
    assert result["source_run_id"] == "rf-run"
    assert result["error_counts"] == {"false_negatives": 1, "false_positives": 2}
    assert result["false_negatives"]["by_group"]["ID1"]["count"] == 1
    assert result["false_negatives"]["by_activity"]["fall"]["count"] == 1
    assert result["false_positives"]["by_activity"]["sit"]["count"] == 1
    assert result["false_positives"]["by_activity"]["walk"]["count"] == 1
    assert result["missed_fall_score_range"]["min"] == 0.2
    assert result["missed_fall_examples"][0]["observation_id"].endswith("missed.csv")
    assert result["false_positives"]["signal_quality"]["missing_recordings"] == 1
    assert result["false_positives"]["duration_buckets"]["5_to_10s"]["count"] == 1


def test_diagnostics_are_deterministic_and_require_aligned_observations(tmp_path: Path) -> None:
    evaluation, records = _fixture(tmp_path)

    first = analyze_random_forest_errors(evaluation, records, run_id="rf", split_name="test")
    second = analyze_random_forest_errors(evaluation, records, run_id="rf", split_name="test")

    assert first == second
    mismatched = dict(evaluation, observation_ids=list(reversed(evaluation["observation_ids"])))
    try:
        analyze_random_forest_errors(mismatched, records, run_id="rf", split_name="test")
    except ValueError as error:
        assert str(error) == "Evaluation observation IDs do not match observation order"
    else:
        raise AssertionError("Expected observation-order validation")


def test_generic_diagnostics_accept_model_agnostic_observations() -> None:
    evaluation = {
        "y_true": ["fall", "non-fall"],
        "y_pred": ["non-fall", "fall"],
        "scores": [[0.7, 0.3], [0.2, 0.8]],
        "positive_scores": [0.3, 0.8],
        "observation_ids": ["sample-1", "sample-2"],
    }
    observations = [
        {"observation_id": "sample-1", "group_id": "subject-A"},
        {"observation_id": "sample-2", "group_id": "subject-B", "activity": "sit"},
    ]

    result = analyze_hard_examples(
        evaluation,
        observations,
        run_id="other-model-run",
        split_name="test",
    )

    assert result["false_negatives"]["by_group"]["subject-A"]["count"] == 1
    assert result["false_positives"]["by_activity"]["sit"]["count"] == 1
    assert result["false_negatives"]["signal_quality"]["missing_recordings"] == 1
    assert result["missed_fall_score_range"]["min"] == 0.3


def test_report_renders_diagnostic_section_and_recording_examples(tmp_path: Path) -> None:
    evaluation, records = _fixture(tmp_path)
    evaluation["hard_example_analysis"] = analyze_random_forest_errors(
        evaluation, records, run_id="rf-run", split_name="test", threshold=0.5
    )
    artifact = {
        "run_id": "rf-run",
        "evaluation_split": "test",
        "evaluations": {"test": evaluation},
    }

    report = render_html_report(artifact)

    assert "Hard-example diagnostics" in report
    assert "missed.csv" in report
    assert "Diagnostic-only analysis" in report
