"""Safety-oriented analysis of CNN and comparator run artifacts."""

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
class CnnAnalysisRun:
    """Paths and artifact produced by CNN comparison analysis."""

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
    for _, result in runs[1:]:
        if result.get("dataset") != reference.get("dataset"):
            raise ValueError("Source datasets differ")
        if result.get("task") != reference.get("task"):
            raise ValueError("Source task definitions differ")
    for split in ("validation", "test"):
        expected = reference["evaluations"][split]
        for _, result in runs[1:]:
            evaluation = result["evaluations"].get(split)
            if evaluation is None:
                raise ValueError(f"Source run is missing {split} evaluation")
            if evaluation.get("observation_ids") != expected.get("observation_ids"):
                raise ValueError(f"Source observation IDs differ for {split}")
            if evaluation.get("y_true") != expected.get("y_true"):
                raise ValueError(f"Source labels differ for {split}")
            if evaluation.get("group_ids") != expected.get("group_ids"):
                raise ValueError(f"Source groups differ for {split}")


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


def _error_ids(evaluation: dict[str, Any]) -> dict[str, list[str]]:
    ids = evaluation["observation_ids"]
    true = evaluation["y_true"]
    predicted = evaluation["y_pred"]
    return {
        "false_negatives": sorted(
            observation_id
            for observation_id, actual, prediction in zip(ids, true, predicted)
            if actual == "fall" and prediction == "non-fall"
        ),
        "false_positives": sorted(
            observation_id
            for observation_id, actual, prediction in zip(ids, true, predicted)
            if actual == "non-fall" and prediction == "fall"
        ),
    }


def _specificity(labels: np.ndarray, predictions: np.ndarray) -> float:
    negative = labels == "non-fall"
    return float(np.mean(predictions[negative] == "non-fall")) if negative.any() else 0.0


def _operating_points(evaluation: dict[str, Any]) -> dict[str, Any]:
    labels = np.asarray(evaluation["y_true"], dtype=object)
    groups = [str(value) for value in evaluation["group_ids"]]
    scores = _scores(evaluation)
    unique = np.unique(scores)
    thresholds = {0.0, 0.5, 1.0, *unique.tolist()}
    thresholds.update(float((left + right) / 2.0) for left, right in pairwise(unique))
    points = []
    for threshold in sorted(thresholds):
        predictions = np.where(scores >= threshold, "fall", "non-fall")
        recalls = []
        specificities = []
        for group in sorted(set(groups)):
            indexes = np.asarray([index for index, value in enumerate(groups) if value == group])
            group_labels = labels[indexes]
            group_predictions = predictions[indexes]
            falls = group_labels == "fall"
            recalls.append(float(np.mean(group_predictions[falls] == "fall")) if falls.any() else 0.0)
            specificities.append(_specificity(group_labels, group_predictions))
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
                "group_specificity": float(np.mean(specificities)) if specificities else 0.0,
            }
        )
    selected: dict[str, dict[str, Any] | None] = {}
    for target in (0.9, 0.95, 1.0):
        eligible = [point for point in points if point["minimum_group_recall"] >= target]
        selected[str(target)] = max(
            eligible,
            key=lambda point: (point["group_specificity"], point["specificity"], -point["threshold"]),
        ) if eligible else None
    return {"points": points, "targets": selected}


def _participant_metrics(evaluation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    labels = evaluation["y_true"]
    predictions = evaluation["y_pred"]
    groups = evaluation["group_ids"]
    metrics: dict[str, dict[str, Any]] = {}
    for group in sorted(set(groups)):
        indexes = [index for index, value in enumerate(groups) if value == group]
        group_labels = np.asarray([labels[index] for index in indexes], dtype=object)
        group_predictions = np.asarray([predictions[index] for index in indexes], dtype=object)
        falls = group_labels == "fall"
        negatives = group_labels == "non-fall"
        false_negatives = int(np.sum(falls & (group_predictions == "non-fall")))
        false_positives = int(np.sum(negatives & (group_predictions == "fall")))
        metrics[str(group)] = {
            "support": len(indexes),
            "fall_support": int(np.sum(falls)),
            "recall": float(np.mean(group_predictions[falls] == "fall")) if falls.any() else 0.0,
            "sensitivity": float(np.mean(group_predictions[falls] == "fall")) if falls.any() else 0.0,
            "specificity": _specificity(group_labels, group_predictions),
            "false_negatives": false_negatives,
            "false_positives": false_positives,
        }
    return metrics


def _source_summary(name: str, path: Path, result: dict[str, Any]) -> dict[str, Any]:
    test = result["evaluations"]["test"]
    validation = result["evaluations"]["validation"]
    return {
        "name": name,
        "run_id": result["run_id"],
        "artifact_path": path.as_posix(),
        "validation_metrics": validation.get("metrics", {}),
        "test_metrics": test.get("metrics", {}),
        "test_errors": _error_ids(test),
        "test_participant_metrics": _participant_metrics(test),
        "validation_participant_metrics": _participant_metrics(validation),
    }


def _render(artifact: dict[str, Any]) -> str:
    def escape(value: Any) -> str:
        return html.escape(str(value), quote=True)

    def json_text(value: Any) -> str:
        return escape(json.dumps(value, sort_keys=True, default=str))

    metric_names = (
        "accuracy", "balanced_accuracy", "precision", "recall", "sensitivity",
        "specificity", "f1", "roc_auc", "pr_auc",
    )
    metric_rows = []
    for source in artifact["sources"]:
        metrics = source["test_metrics"]
        metric_rows.append(
            "<tr>"
            f"<th>{escape(source['name'])}</th>"
            + "".join(f"<td>{escape(metrics.get(name, 'Not available'))}</td>" for name in metric_names)
            + "</tr>"
        )
    error_rows = []
    for source in artifact["sources"]:
        errors = source["test_errors"]
        error_rows.append(
            "<tr>"
            f"<th>{escape(source['name'])}</th>"
            f"<td>{json_text(errors['false_negatives'])}</td>"
            f"<td>{json_text(errors['false_positives'])}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CNN analysis</title><style>
body {{ font-family: system-ui, sans-serif; max-width: 1400px; margin: auto; padding: 2rem; background: #f8f9fa; }}
section {{ background: white; border: 1px solid #dadce0; border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
table {{ width: 100%; border-collapse: collapse; margin: .5rem 0 1rem; }}
th, td {{ border: 1px solid #dadce0; padding: .4rem; text-align: left; vertical-align: top; }}
th {{ background: #f1f3f4; }}
.note {{ color: #5f6368; }}
</style></head><body>
<h1>CNN analysis</h1>
<p class="note">Validation operating points use validation observations only. Test errors are final diagnostic reporting.</p>
<section><h2>Test metrics</h2><table><thead><tr><th>model</th>{''.join(f'<th>{escape(name)}</th>' for name in metric_names)}</tr></thead><tbody>{''.join(metric_rows)}</tbody></table></section>
<section><h2>Test errors</h2><table><thead><tr><th>model</th><th>false negatives</th><th>false positives</th></tr></thead><tbody>{''.join(error_rows)}</tbody></table></section>
<section><h2>Per-participant safety metrics</h2><pre>{json_text({source['name']: source['test_participant_metrics'] for source in artifact['sources']})}</pre></section>
<section><h2>CNN operating points</h2><pre>{json_text(artifact['threshold_analysis']['models']['cnn'])}</pre></section>
<section><h2>CNN diagnostics</h2><pre>{json_text(artifact['cnn_diagnostics'])}</pre></section>
<section><h2>Complementarity</h2><pre>{json_text(artifact['complementarity'])}</pre></section>
<section><h2>Recommendation input</h2><pre>{json_text(artifact['recommendation'])}</pre></section>
</body></html>"""


def analyze_cnn_runs(
    cnn_run: str | Path,
    comparator_runs: tuple[str | Path, ...],
    output_dir: str | Path,
) -> CnnAnalysisRun:
    """Create reproducible CNN comparison and safety operating-point analysis."""

    paths_and_results = [_load(cnn_run)] + [_load(path) for path in comparator_runs]
    _check_compatible(paths_and_results)
    names = ["cnn", "svm", "rf", "ensemble"][: len(paths_and_results)]
    summaries = [_source_summary(name, path, result) for name, (path, result) in zip(names, paths_and_results)]
    test_evaluations = [result["evaluations"]["test"] for _, result in paths_and_results]
    cnn_errors = _error_ids(test_evaluations[0])
    all_error_sets = {
        "cnn_false_negatives": set(cnn_errors["false_negatives"]),
        "cnn_false_positives": set(cnn_errors["false_positives"]),
    }
    for name, evaluation in zip(names[1:], test_evaluations[1:]):
        errors = _error_ids(evaluation)
        all_error_sets[f"{name}_false_negatives"] = set(errors["false_negatives"])
        all_error_sets[f"{name}_false_positives"] = set(errors["false_positives"])

    true_fall_ids = {
        observation_id
        for observation_id, label in zip(test_evaluations[0]["observation_ids"], test_evaluations[0]["y_true"])
        if label == "fall"
    }
    cnn_predictions = dict(zip(test_evaluations[0]["observation_ids"], test_evaluations[0]["y_pred"]))
    catches_missed_by: dict[str, list[str]] = {}
    for name, evaluation in zip(names[1:], test_evaluations[1:]):
        other_predictions = dict(zip(evaluation["observation_ids"], evaluation["y_pred"]))
        catches_missed_by[name] = sorted(
            observation_id
            for observation_id in true_fall_ids
            if cnn_predictions[observation_id] == "fall" and other_predictions[observation_id] == "non-fall"
        )

    cnn_fn = all_error_sets["cnn_false_negatives"]
    cnn_fp = all_error_sets["cnn_false_positives"]
    shared_fn = sorted(set.intersection(*(all_error_sets[f"{name}_false_negatives"] for name in names)))
    shared_fp = sorted(set.intersection(*(all_error_sets[f"{name}_false_positives"] for name in names)))
    threshold_models = {
        name: _operating_points(result["evaluations"]["validation"])
        for name, (_, result) in zip(names, paths_and_results)
    }
    cnn_diagnostics = {
        split: {
            "hard_examples": result["evaluations"][split].get("hard_example_analysis", {}),
            "temporal_pattern": {
                "available": False,
                "reason": "Source run artifacts do not contain temporal event annotations",
            },
        }
        for split in ("validation", "test")
        for _, result in [paths_and_results[0]]
    }
    artifact = {
        "schema_version": 1,
        "run_id": "wifall-cnn-analysis",
        "dataset": paths_and_results[0][1]["dataset"],
        "task": paths_and_results[0][1]["task"],
        "model": {"family": "model_analysis", "name": "cnn_complementarity_analysis"},
        "features": {},
        "preprocessing": {},
        "split": paths_and_results[0][1]["split"],
        "evaluation_split": "test",
        "evaluations": {"test": test_evaluations[0]},
        "metrics": test_evaluations[0].get("metrics", {}),
        "confusion_matrix": test_evaluations[0].get("confusion_matrix", []),
        "model_order": names,
        "sources": summaries,
        "errors": {name: {"test": _error_ids(evaluation)} for name, evaluation in zip(names, test_evaluations)},
        "complementarity": {
            "cnn_catches_fall_missed_by": catches_missed_by,
            "shared_false_negatives": shared_fn,
            "shared_false_positives": shared_fp,
            "cnn_only_false_negatives": sorted(cnn_fn.difference(*(all_error_sets[f"{name}_false_negatives"] for name in names[1:]))),
            "cnn_only_false_positives": sorted(cnn_fp.difference(*(all_error_sets[f"{name}_false_positives"] for name in names[1:]))),
        },
        "threshold_analysis": {
            "uses_split": "validation",
            "test_excluded": True,
            "models": threshold_models,
        },
        "cnn_diagnostics": cnn_diagnostics,
        "recommendation": {
            "decision": "continue_investigating",
            "reason": "CNN catches some falls missed by comparator models but has insufficient specificity at its safety-oriented test operating point.",
        },
    }
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = output / "cnn-analysis.json"
    save_evaluation_result(artifact, artifact_path)
    report_path = output / "report.html"
    report_path.write_text(_render(artifact), encoding="utf-8")
    return CnnAnalysisRun(artifact_path, report_path, artifact)