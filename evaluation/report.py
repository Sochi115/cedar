"""Render generic evaluation result artifacts as self-contained HTML."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .core import load_evaluation_result


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _json_text(value: Any) -> str:
    return _escape(json.dumps(value, sort_keys=True, default=str))


def _format_value(value: Any) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    return _escape(value)


def _metadata_table(title: str, metadata: Mapping[str, Any]) -> str:
    rows = []
    for key, value in metadata.items():
        rendered = _json_text(value) if isinstance(value, (dict, list)) else _format_value(value)
        rows.append(f"<tr><th>{_escape(key)}</th><td>{rendered}</td></tr>")
    return f"<section><h2>{_escape(title)}</h2><table class='metadata'>{''.join(rows)}</table></section>"


def _metric_table(metrics: Mapping[str, Any]) -> str:
    names = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1",
        "false_positive_rate",
        "roc_auc",
        "pr_auc",
        "support",
    )
    rows = [
        f"<tr><th>{_escape(name)}</th><td>{_format_value(metrics.get(name))}</td></tr>"
        for name in names
    ]
    return f"<h3>Overall metrics</h3><table class='metrics'>{''.join(rows)}</table>"


def _averages_table(metrics: Mapping[str, Any]) -> str:
    rows = []
    for average_name in ("macro_avg", "weighted_avg"):
        values = metrics.get(average_name)
        if not isinstance(values, Mapping):
            continue
        rows.append(
            "<tr>"
            f"<th>{_escape(average_name)}</th>"
            f"<td>{_format_value(values.get('precision'))}</td>"
            f"<td>{_format_value(values.get('recall'))}</td>"
            f"<td>{_format_value(values.get('f1'))}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return (
        "<h3>Macro and weighted averages</h3>"
        "<table class='metrics'><thead><tr><th>average</th>"
        "<th>precision</th><th>recall</th><th>F1</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _per_class_table(metrics: Mapping[str, Any]) -> str:
    per_class = metrics.get("per_class")
    if not isinstance(per_class, Mapping):
        return ""
    rows = []
    for label, values in per_class.items():
        rows.append(
            "<tr>"
            f"<th>{_escape(label)}</th>"
            f"<td>{_format_value(values.get('precision'))}</td>"
            f"<td>{_format_value(values.get('recall'))}</td>"
            f"<td>{_format_value(values.get('f1'))}</td>"
            f"<td>{_format_value(values.get('support'))}</td>"
            "</tr>"
        )
    return (
        "<h3>Per-class metrics</h3>"
        "<table class='metrics'><thead><tr><th>class</th>"
        "<th>precision</th><th>recall</th><th>F1</th><th>support</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _confusion_matrix_table(evaluation: Mapping[str, Any]) -> str:
    labels = evaluation.get("labels", [])
    matrix = evaluation.get("confusion_matrix", [])
    if not labels or not matrix:
        return "<p>Confusion matrix: Not available</p>"
    rows = []
    for label, values in zip(labels, matrix):
        total = sum(values)
        cells = []
        for value in values:
            percentage = (100 * value / total) if total else 0.0
            cells.append(f"<td>{_escape(value)} <small>({percentage:.1f}%)</small></td>")
        rows.append(f"<tr><th>{_escape(label)}</th>{''.join(cells)}</tr>")
    header = "".join(f"<th>{_escape(label)}</th>" for label in labels)
    return (
        "<h3>Confusion matrix</h3>"
        "<p>Rows are true labels. Percentages use each row total.</p>"
        "<table class='confusion'><thead><tr><th>True \\ Predicted</th>"
        f"{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _evaluation_section(name: str, evaluation: Mapping[str, Any], primary: bool) -> str:
    metrics = evaluation.get("metrics", {})
    title = f"Evaluation: {_escape(name)}"
    if primary:
        title += " <span class='badge'>primary</span>"
    score_type = evaluation.get("score_type") or "none"
    return (
        f"<section><h2>{title}</h2>"
        f"<p>Score source: {_escape(score_type)}</p>"
        f"{_confusion_matrix_table(evaluation)}"
        f"{_metric_table(metrics)}"
        f"{_per_class_table(metrics)}"
        f"{_averages_table(metrics)}</section>"
    )


def render_html_report(result: Mapping[str, Any]) -> str:
    """Return a self-contained HTML report for one generic result artifact."""

    evaluations = result.get("evaluations", {})
    primary = result.get("evaluation_split")
    sections = [
        _evaluation_section(name, evaluation, name == primary)
        for name, evaluation in evaluations.items()
    ]
    metadata = "".join(
        (
            _metadata_table("Dataset", result.get("dataset", {})),
            _metadata_table("Task", result.get("task", {})),
            _metadata_table("Model", result.get("model", {})),
            _metadata_table("Features", result.get("features", {})),
            _metadata_table("Preprocessing", result.get("preprocessing", {})),
            _metadata_table("Split", result.get("split", {})),
        )
    )
    title = _escape(result.get("run_id", "Evaluation report"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: light; font-family: system-ui, sans-serif; }}
body {{ margin: 0 auto; max-width: 1100px; padding: 2rem; color: #202124; background: #f8f9fa; }}
section {{ margin: 1.5rem 0; padding: 1rem 1.25rem; background: white; border: 1px solid #dadce0; border-radius: 8px; }}
h1 {{ margin-top: 0; }}
h2 {{ border-bottom: 1px solid #dadce0; padding-bottom: .4rem; }}
h3 {{ margin-bottom: .4rem; }}
table {{ width: 100%; border-collapse: collapse; margin: .5rem 0 1.2rem; }}
th, td {{ padding: .45rem .6rem; border: 1px solid #dadce0; text-align: left; vertical-align: top; }}
th {{ background: #f1f3f4; }}
td {{ white-space: pre-wrap; word-break: break-word; }}
.metadata th {{ width: 24%; }}
.confusion td, .confusion th {{ text-align: center; }}
small {{ color: #5f6368; }}
.badge {{ font-size: .7em; background: #d7f0da; color: #176b2c; padding: .2em .5em; border-radius: 1em; vertical-align: middle; }}
.note {{ color: #5f6368; }}
</style>
</head>
<body>
<h1>Evaluation report: {title}</h1>
<p class="note">This report is generated from a model-agnostic evaluation result artifact.</p>
{metadata}
{''.join(sections)}
</body>
</html>
"""


def render_html_report_file(
    result_path: str | Path, output_path: str | Path
) -> Path:
    """Load one result artifact and write a self-contained HTML report."""

    result = load_evaluation_result(result_path)
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_html_report(result), encoding="utf-8")
    return destination
