"""Validation-only analysis of LSTM performance and model complementarity."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from evaluation import load_evaluation_result, save_evaluation_result


@dataclass(frozen=True)
class LstmAnalysisRun:
    """Paths and artifact produced by LSTM comparison analysis."""

    artifact_path: Path
    report_path: Path
    artifact: dict[str, Any]


def _load(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser()
    if source.is_dir():
        source = source / "run.json"
    return source.resolve(), load_evaluation_result(source)


def _check_compatible(runs: list[tuple[Path, dict[str, Any]]]) -> None:
    reference = runs[0][1]
    reference_source = reference.get("data_source", {}).get("type")
    if reference_source != "duration_normalized_dataset":
        raise ValueError("LSTM analysis requires duration-normalized source runs")
    for _, result in runs[1:]:
        if result.get("dataset") != reference.get("dataset"):
            raise ValueError("Source datasets differ")
        if result.get("task") != reference.get("task"):
            raise ValueError("Source task definitions differ")
        if result.get("data_source", {}).get("type") != reference_source:
            raise ValueError("Source data representations differ")
    for split in ("validation", "test"):
        expected = reference["evaluations"][split]
        for _, result in runs[1:]:
            evaluation = result["evaluations"].get(split)
            if evaluation is None:
                raise ValueError(f"Source run is missing {split} evaluation")
            for field, message in (
                ("observation_ids", "observation IDs"),
                ("y_true", "labels"),
                ("group_ids", "groups"),
            ):
                if evaluation.get(field) != expected.get(field):
                    raise ValueError(f"Source {message} differ for {split}")


def _metrics(labels: list[str], predictions: list[str]) -> dict[str, Any]:
    actual = np.asarray(labels, dtype=object)
    predicted = np.asarray(predictions, dtype=object)
    falls = actual == "fall"
    negatives = actual == "non-fall"
    true_positives = int(np.sum(falls & (predicted == "fall")))
    false_positives = int(np.sum(negatives & (predicted == "fall")))
    false_negatives = int(np.sum(falls & (predicted == "non-fall")))
    true_negatives = int(np.sum(negatives & (predicted == "non-fall")))
    recall = true_positives / max(true_positives + false_negatives, 1)
    specificity = true_negatives / max(true_negatives + false_positives, 1)
    precision = true_positives / max(true_positives + false_positives, 1)
    return {
        "accuracy": float(np.mean(actual == predicted)),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "false_positive_rate": float(1.0 - specificity),
        "f1": float(2.0 * precision * recall / (precision + recall))
        if precision + recall
        else 0.0,
        "support": int(actual.size),
        "false_negatives": false_negatives,
        "false_positives": false_positives,
    }


def _participant_metrics(evaluation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    labels = evaluation["y_true"]
    predictions = evaluation["y_pred"]
    groups = evaluation["group_ids"]
    metrics: dict[str, dict[str, Any]] = {}
    for group in sorted(set(groups)):
        indexes = [index for index, value in enumerate(groups) if value == group]
        group_metrics = _metrics(
            [labels[index] for index in indexes],
            [predictions[index] for index in indexes],
        )
        group_metrics["fall_support"] = int(
            sum(labels[index] == "fall" for index in indexes)
        )
        metrics[str(group)] = group_metrics
    return metrics


def _error_ids(evaluation: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "false_negatives": sorted(
            observation_id
            for observation_id, actual, prediction in zip(
                evaluation["observation_ids"], evaluation["y_true"], evaluation["y_pred"]
            )
            if actual == "fall" and prediction == "non-fall"
        ),
        "false_positives": sorted(
            observation_id
            for observation_id, actual, prediction in zip(
                evaluation["observation_ids"], evaluation["y_true"], evaluation["y_pred"]
            )
            if actual == "non-fall" and prediction == "fall"
        ),
    }


def _scores(evaluation: dict[str, Any]) -> np.ndarray:
    values = evaluation.get("positive_scores")
    if values is None:
        values = evaluation.get("scores")
    if values is None:
        raise ValueError("Evaluation has no positive-class scores")
    scores = np.asarray(values, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Evaluation positive-class scores must be finite one-dimensional values")
    return scores


def _specificity(labels: np.ndarray, predictions: np.ndarray) -> float:
    negatives = labels == "non-fall"
    return float(np.mean(predictions[negatives] == "non-fall")) if negatives.any() else 0.0


def _operating_points(evaluation: dict[str, Any]) -> dict[str, Any]:
    labels = np.asarray(evaluation["y_true"], dtype=object)
    groups = [str(value) for value in evaluation["group_ids"]]
    scores = _scores(evaluation)
    if scores.size != labels.size:
        raise ValueError("Scores and labels differ in length")
    unique = np.unique(scores)
    thresholds = {0.0, 0.5, 1.0, *unique.tolist()}
    thresholds.update(float((left + right) / 2.0) for left, right in pairwise(unique))
    points = []
    for threshold in sorted(thresholds):
        predictions = np.where(scores >= threshold, "fall", "non-fall")
        recalls = []
        group_specificities = []
        for group in sorted(set(groups)):
            indexes = np.asarray([index for index, value in enumerate(groups) if value == group])
            group_labels = labels[indexes]
            group_predictions = predictions[indexes]
            falls = group_labels == "fall"
            recalls.append(float(np.mean(group_predictions[falls] == "fall")) if falls.any() else 0.0)
            group_specificities.append(_specificity(group_labels, group_predictions))
        recall = float(np.mean(predictions[labels == "fall"] == "fall")) if np.any(labels == "fall") else 0.0
        specificity = _specificity(labels, predictions)
        points.append(
            {
                "threshold": float(threshold),
                "recall": recall,
                "sensitivity": recall,
                "specificity": specificity,
                "false_positive_rate": 1.0 - specificity,
                "minimum_group_recall": min(recalls) if recalls else 0.0,
                "group_specificity": float(np.mean(group_specificities)) if group_specificities else 0.0,
            }
        )
    selected: dict[str, dict[str, Any] | None] = {}
    for target in (0.9, 0.95, 1.0):
        eligible = [point for point in points if point["minimum_group_recall"] >= target]
        selected[str(target)] = (
            max(
                eligible,
                key=lambda point: (point["group_specificity"], point["specificity"], -point["threshold"]),
            )
            if eligible
            else None
        )
    return {"points": points, "targets": selected}


def _source_summary(name: str, path: Path, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "run_id": result["run_id"],
        "artifact_path": path.as_posix(),
        "validation_metrics": _metrics(
            result["evaluations"]["validation"]["y_true"],
            result["evaluations"]["validation"]["y_pred"],
        ),
        "test_metrics": _metrics(
            result["evaluations"]["test"]["y_true"],
            result["evaluations"]["test"]["y_pred"],
        ),
        "test_errors": _error_ids(result["evaluations"]["test"]),
        "test_participant_metrics": _participant_metrics(result["evaluations"]["test"]),
        "validation_participant_metrics": _participant_metrics(result["evaluations"]["validation"]),
    }


def _complementarity(
    names: list[str], evaluations: list[dict[str, Any]]
) -> dict[str, Any]:
    primary = evaluations[0]
    primary_ids = primary["observation_ids"]
    true_fall_ids = {
        observation_id
        for observation_id, label in zip(primary_ids, primary["y_true"])
        if label == "fall"
    }
    predictions = [dict(zip(evaluation["observation_ids"], evaluation["y_pred"])) for evaluation in evaluations]
    errors = [_error_ids(evaluation) for evaluation in evaluations]
    primary_fall_predictions = {
        observation_id for observation_id in primary_ids if predictions[0][observation_id] == "fall"
    }
    by_comparator: dict[str, Any] = {}
    for index, name in enumerate(names[1:], start=1):
        comparator_falls = {
            observation_id for observation_id in primary_ids if predictions[index][observation_id] == "fall"
        }
        by_comparator[name] = {
            "lstm_catches_falls_missed_by_comparator": sorted(
                true_fall_ids.intersection(primary_fall_predictions - comparator_falls)
            ),
            "comparator_catches_falls_missed_by_lstm": sorted(
                true_fall_ids.intersection(comparator_falls - primary_fall_predictions)
            ),
            "shared_false_negatives": sorted(
                set(errors[0]["false_negatives"]).intersection(errors[index]["false_negatives"])
            ),
            "shared_false_positives": sorted(
                set(errors[0]["false_positives"]).intersection(errors[index]["false_positives"])
            ),
            "fall_prediction_jaccard": float(
                len(primary_fall_predictions & comparator_falls)
                / max(len(primary_fall_predictions | comparator_falls), 1)
            ),
        }
    primary_fn = set(errors[0]["false_negatives"])
    primary_fp = set(errors[0]["false_positives"])
    comparator_fns = [set(error["false_negatives"]) for error in errors[1:]]
    comparator_fps = [set(error["false_positives"]) for error in errors[1:]]
    return {
        "by_comparator": by_comparator,
        "lstm_catches_fall_missed_by": {
            name: values["lstm_catches_falls_missed_by_comparator"]
            for name, values in by_comparator.items()
        },
        "lstm_only_false_negatives": sorted(primary_fn.difference(*comparator_fns)),
        "lstm_only_false_positives": sorted(primary_fp.difference(*comparator_fps)),
        "shared_false_negatives_all_models": sorted(set.intersection(primary_fn, *comparator_fns)),
        "shared_false_positives_all_models": sorted(set.intersection(primary_fp, *comparator_fps)),
    }


def _recommendation(threshold_models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    lstm_point = threshold_models["lstm"]["targets"]["0.95"]
    comparator_points = [
        model["targets"]["0.95"]
        for name, model in threshold_models.items()
        if name != "lstm" and model["targets"]["0.95"] is not None
    ]
    if lstm_point is None:
        decision = "reject"
        reason = "LSTM has no validation operating point meeting 95% minimum participant recall."
        best_comparator_specificity = None
    elif not comparator_points:
        decision = "investigate"
        reason = "No compatible comparator has a validation operating point at 95% minimum participant recall."
        best_comparator_specificity = None
    else:
        best_comparator_specificity = max(point["specificity"] for point in comparator_points)
        gap = best_comparator_specificity - lstm_point["specificity"]
        if gap <= 0.0:
            decision = "adopt"
            reason = "LSTM matches or exceeds comparator validation specificity at 95% minimum participant recall."
        elif gap <= 0.10:
            decision = "investigate"
            reason = "LSTM is close to the best comparator on validation specificity at 95% minimum participant recall."
        else:
            decision = "reject"
            reason = "LSTM trails comparator validation specificity by more than 0.10 at 95% minimum participant recall."
    return {
        "decision": decision,
        "reason": reason,
        "basis": "validation_only",
        "target_minimum_participant_recall": 0.95,
        "lstm_operating_point": lstm_point,
        "best_comparator_specificity": best_comparator_specificity,
        "test_used_for_decision": False,
        "limitations": [
            "Only held-out test participants provide final generalization evidence.",
            "The participant count is small, so per-participant metrics are uncertain.",
            "Recording-level metrics do not measure event-level alert latency or temporal persistence.",
        ],
    }


def _render(artifact: dict[str, Any]) -> str:
    def escape(value: Any) -> str:
        return html.escape(str(value), quote=True)

    def json_text(value: Any) -> str:
        return escape(json.dumps(value, sort_keys=True, default=str, indent=2))

    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "false_positive_rate",
        "f1",
    )
    aggregate_rows = []
    for source in artifact["sources"]:
        for split in ("validation", "test"):
            metrics = source[f"{split}_metrics"]
            aggregate_rows.append(
                "<tr>"
                f"<th>{escape(source['name'])}</th><th>{escape(split)}</th>"
                + "".join(f"<td>{escape(metrics[name])}</td>" for name in metric_names)
                + "</tr>"
            )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LSTM analysis</title><style>
body {{ font-family: system-ui, sans-serif; max-width: 1500px; margin: auto; padding: 2rem; background: #f8f9fa; }}
section {{ background: white; border: 1px solid #dadce0; border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
table {{ width: 100%; border-collapse: collapse; margin: .5rem 0 1rem; }}
th, td {{ border: 1px solid #dadce0; padding: .4rem; text-align: left; vertical-align: top; }}
th {{ background: #f1f3f4; }}
.note {{ color: #5f6368; }} pre {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
</style></head><body>
<h1>LSTM analysis</h1>
<p class="note">Architecture and threshold decisions use validation observations only. Test results are final diagnostics and are excluded from the recommendation.</p>
<section><h2>Aggregate metrics</h2><table><thead><tr><th>model</th><th>split</th>{''.join(f'<th>{escape(name)}</th>' for name in metric_names)}</tr></thead><tbody>{''.join(aggregate_rows)}</tbody></table></section>
<section><h2>Per-participant metrics</h2><pre>{json_text({source['name']: {'validation': source['validation_participant_metrics'], 'test': source['test_participant_metrics']} for source in artifact['sources']})}</pre></section>
<section><h2>Errors</h2><pre>{json_text(artifact['errors'])}</pre></section>
<section><h2>Validation operating points</h2><pre>{json_text(artifact['threshold_analysis'])}</pre></section>
<section><h2>Complementarity</h2><pre>{json_text(artifact['complementarity'])}</pre></section>
<section><h2>Recommendation and limitations</h2><pre>{json_text(artifact['recommendation'])}</pre></section>
</body></html>"""


def analyze_lstm_runs(
    lstm_run: str | Path,
    comparator_runs: tuple[str | Path, ...],
    output_dir: str | Path,
) -> LstmAnalysisRun:
    """Analyze LSTM against duration-normalized CNN, SVM, and RF runs."""

    paths_and_results = [_load(lstm_run)] + [_load(path) for path in comparator_runs]
    if len(paths_and_results) != 4:
        raise ValueError("LSTM analysis requires LSTM, CNN, SVM, and RF runs")
    _check_compatible(paths_and_results)
    names = ["lstm", "cnn", "svm", "rf"]
    evaluations = [result["evaluations"]["test"] for _, result in paths_and_results]
    summaries = [
        _source_summary(name, path, result)
        for name, (path, result) in zip(names, paths_and_results)
    ]
    threshold_models = {
        name: _operating_points(result["evaluations"]["validation"])
        for name, (_, result) in zip(names, paths_and_results)
    }
    artifact = {
        "schema_version": 1,
        "run_id": "wifall-lstm-analysis",
        "dataset": paths_and_results[0][1]["dataset"],
        "task": paths_and_results[0][1]["task"],
        "model": {"family": "model_analysis", "name": "lstm_complementarity_analysis"},
        "features": {},
        "preprocessing": {},
        "split": paths_and_results[0][1]["split"],
        "evaluation_split": "test",
        "evaluations": {"test": evaluations[0]},
        "metrics": summaries[0]["test_metrics"],
        "confusion_matrix": evaluations[0].get("confusion_matrix", []),
        "model_order": names,
        "sources": summaries,
        "errors": {
            name: {"validation": _error_ids(result["evaluations"]["validation"]), "test": _error_ids(result["evaluations"]["test"])}
            for name, (_, result) in zip(names, paths_and_results)
        },
        "complementarity": _complementarity(names, evaluations),
        "threshold_analysis": {
            "uses_split": "validation",
            "test_excluded": True,
            "models": threshold_models,
        },
        "recommendation": _recommendation(threshold_models),
    }
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = output / "lstm-analysis.json"
    save_evaluation_result(artifact, artifact_path)
    report_path = output / "report.html"
    report_path.write_text(_render(artifact), encoding="utf-8")
    return LstmAnalysisRun(artifact_path, report_path, artifact)
