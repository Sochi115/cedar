"""Compare generic evaluation result artifacts in a static HTML report."""

from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import load_evaluation_result

METRICS = (
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("false_positive_rate", "False-positive rate"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
)


@dataclass(frozen=True)
class ComparisonRun:
    """One generic result artifact and its optional individual report."""

    result_path: Path
    result: dict[str, Any]
    report_path: Path


def _result_path(path: str | Path) -> Path:
    source = Path(path).expanduser()
    if source.is_dir():
        source = source / "run.json"
    if not source.is_file():
        raise FileNotFoundError(f"Evaluation result does not exist: {source}")
    return source.resolve()


def load_comparison_runs(paths: list[str | Path] | tuple[str | Path, ...]) -> tuple[ComparisonRun, ...]:
    """Load one or more generic result artifacts for comparison."""

    if not paths:
        raise ValueError("At least one evaluation result is required")
    runs = []
    for path in paths:
        result_path = _result_path(path)
        runs.append(
            ComparisonRun(
                result_path=result_path,
                result=load_evaluation_result(result_path),
                report_path=result_path.parent / "report.html",
            )
        )
    return tuple(runs)


def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _dataset_name(run: ComparisonRun) -> str:
    dataset = run.result.get("dataset", {})
    if isinstance(dataset, dict):
        return str(dataset.get("name", "Not available"))
    return str(dataset)


def _task_signature(run: ComparisonRun) -> tuple[Any, ...]:
    task = run.result.get("task", {})
    if not isinstance(task, dict):
        return (str(task), ())
    return (task.get("name"), _json_key(task.get("labels")))


def _protocol_signature(run: ComparisonRun) -> tuple[Any, ...]:
    split = run.result.get("split", {})
    if not isinstance(split, dict):
        return (str(split), run.result.get("evaluation_split"))
    return (
        split.get("strategy"),
        _json_key(split.get("config", {})),
        run.result.get("evaluation_split"),
    )


def find_compatibility_warnings(runs: tuple[ComparisonRun, ...]) -> list[str]:
    """Return warnings for comparisons that are not directly equivalent."""

    warnings: list[str] = []
    datasets = sorted({_dataset_name(run) for run in runs})
    if len(datasets) > 1:
        warnings.append(
            "Runs use multiple datasets; metric comparisons may not be directly comparable: "
            + ", ".join(datasets)
        )

    tasks = {_task_signature(run) for run in runs}
    if len(tasks) > 1:
        warnings.append("Runs use different task definitions or label sets.")

    protocols = {_protocol_signature(run) for run in runs}
    if len(protocols) > 1:
        warnings.append("Runs use different evaluation protocols.")

    missing_reports = [run.result.get("run_id", run.result_path.name) for run in runs if not run.report_path.is_file()]
    if missing_reports:
        warnings.append(
            "Individual reports are missing for: " + ", ".join(str(value) for value in missing_reports)
        )
    return warnings


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _json_text(value: Any) -> str:
    return _escape(json.dumps(value, sort_keys=True, default=str))


def _display(value: Any) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, (dict, list)):
        return _json_text(value)
    return _escape(value)


def _metric_value(run: ComparisonRun, name: str) -> Any:
    evaluation_split = run.result.get("evaluation_split")
    evaluations = run.result.get("evaluations", {})
    evaluation = evaluations.get(evaluation_split, {}) if isinstance(evaluations, dict) else {}
    metrics = evaluation.get("metrics", {}) if isinstance(evaluation, dict) else {}
    if not metrics:
        metrics = run.result.get("metrics", {})
    return metrics.get(name) if isinstance(metrics, dict) else None


def _metric_display(value: Any) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, float):
        return f"{value:.4f}"
    return _escape(value)


def _run_label(run: ComparisonRun) -> str:
    return str(run.result.get("run_id", run.result_path.stem))


def _leader(runs: tuple[ComparisonRun, ...], metric: str) -> tuple[str, float] | None:
    values = []
    for run in runs:
        value = _metric_value(run, metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append((_run_label(run), float(value)))
    return max(values, key=lambda item: item[1]) if values else None


def _analysis_summary(runs: tuple[ComparisonRun, ...], warnings: list[str]) -> str:
    datasets = sorted({_dataset_name(run) for run in runs})
    splits = sorted({str(run.result.get("evaluation_split", "Not available")) for run in runs})
    summary = (
        f"Compared {_escape(len(runs))} run(s) across {_escape(len(datasets))} dataset(s), "
        f"using {_escape(', '.join(splits))} evaluation split(s)."
    )
    if warnings:
        caution = (
            "Compatibility warnings are present. Treat rankings as directional only until "
            "runs share labels, task, dataset, and evaluation protocol."
        )
    else:
        caution = "Runs share compatible labels, task, dataset, and evaluation protocol."

    metric_leaders = []
    for metric, label in (
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("f1", "F1"),
        ("recall", "Recall"),
        ("precision", "Precision"),
        ("specificity", "Specificity"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
    ):
        leader = _leader(runs, metric)
        if leader:
            metric_leaders.append(
                f"<li><strong>{_escape(label)}:</strong> {_escape(leader[0])} "
                f"({_escape(f'{leader[1]:.4f}')})</li>"
            )

    window_groups: dict[str, list[ComparisonRun]] = {}
    for run in runs:
        features = run.result.get("features", {})
        window_count = features.get("window_count") if isinstance(features, dict) else None
        window_groups.setdefault(str(window_count or "Not available"), []).append(run)
    window_analysis = []
    for window_count, window_runs in sorted(window_groups.items()):
        leader = _leader(tuple(window_runs), "f1")
        run_names = ", ".join(_run_label(run) for run in window_runs)
        if leader:
            window_analysis.append(
                f"<li><strong>{_escape(window_count)} windows:</strong> best F1 "
                f"is {_escape(f'{leader[1]:.4f}')} ({_escape(leader[0])}); "
                f"runs: {_escape(run_names)}</li>"
            )
        else:
            window_analysis.append(
                f"<li><strong>{_escape(window_count)} windows:</strong> "
                f"runs: {_escape(run_names)}</li>"
            )

    accuracy_leader = _leader(runs, "accuracy")
    recall_leader = _leader(runs, "recall")
    precision_leader = _leader(runs, "precision")
    analysis = []
    if accuracy_leader and recall_leader:
        if accuracy_leader[0] == recall_leader[0]:
            analysis.append(
                f"{_escape(accuracy_leader[0])} leads both accuracy and recall, "
                "making it the strongest general-purpose candidate in this comparison."
            )
        else:
            analysis.append(
                f"Accuracy favors {_escape(accuracy_leader[0])}, while recall favors "
                f"{_escape(recall_leader[0])}; this exposes a false-alarm versus missed-event trade-off."
            )
    if precision_leader and recall_leader and precision_leader[0] != recall_leader[0]:
        analysis.append(
            f"{_escape(precision_leader[0])} favors precision, while {_escape(recall_leader[0])} "
            "favors recall. Choose based on whether false alarms or missed falls cost more."
        )
    analysis.append(
        "For fall detection, review sensitivity, specificity, false-positive rate, and per-participant "
        "metrics alongside accuracy. Aggregate accuracy alone can hide participant-level failures."
    )
    return (
        "<section class='analysis'><h2>Analysis and summary</h2>"
        f"<p>{summary}</p><p>{_escape(caution)}</p>"
        "<h3>Metric leaders</h3><ul>"
        f"{''.join(metric_leaders) or '<li>No comparable metrics available.</li>'}"
        "</ul><h3>Window-size analysis</h3><ul>"
        f"{''.join(window_analysis) or '<li>Window size not recorded.</li>'}"
        "</ul><h3>Human interpretation</h3>"
        f"<ul>{''.join(f'<li>{item}</li>' for item in analysis)}</ul></section>"
    )


def _report_link(run: ComparisonRun, output_path: Path) -> str:
    if not run.report_path.is_file():
        return "<span>Not available</span>"
    relative = Path(os.path.relpath(run.report_path, output_path.parent)).as_posix()
    return f"<a href='{_escape(relative)}'>Individual report</a>"


def _run_details(run: ComparisonRun, output_path: Path) -> str:
    result = run.result
    model = result.get("model", {})
    task = result.get("task", {})
    split = result.get("split", {})
    rows = (
        ("Source artifact", run.result_path.as_posix()),
        ("Run ID", result.get("run_id")),
        ("Model", model),
        ("Dataset", result.get("dataset", {})),
        ("Task", task),
        ("Features", result.get("features", {})),
        ("Preprocessing", result.get("preprocessing", {})),
        ("Split", split),
        ("Evaluation split", result.get("evaluation_split")),
        ("Individual report", _report_link(run, output_path)),
    )
    rendered = []
    for key, value in rows:
        value_html = value if key == "Individual report" else _display(value)
        rendered.append(f"<tr><th>{_escape(key)}</th><td>{value_html}</td></tr>")
    run_id = _escape(result.get("run_id", run.result_path.stem))
    return (
        f"<details><summary>{run_id} metadata</summary>"
        f"<table class='metadata'>{''.join(rendered)}</table></details>"
    )


def render_comparison_report(
    runs: tuple[ComparisonRun, ...], output_path: str | Path = "comparison.html"
) -> str:
    """Render a self-contained comparison report for generic result artifacts."""

    destination = Path(output_path).expanduser().resolve()
    warnings = find_compatibility_warnings(runs)
    warning_html = ""
    if warnings:
        items = "".join(f"<li>{_escape(warning)}</li>" for warning in warnings)
        warning_html = f"<section class='warnings'><h2>Compatibility warnings</h2><ul>{items}</ul></section>"

    header = "".join(
        f"<th>{_escape(label)}</th>" for _, label in (("run", "Run"), ("report", "Report"), ("family", "Model family"), ("dataset", "Dataset"), *METRICS)
    )
    body = []
    for run in runs:
        result = run.result
        model = result.get("model", {})
        model = model if isinstance(model, dict) else {}
        row = [
            _escape(result.get("run_id", run.result_path.stem)),
            _report_link(run, destination),
            _display(model.get("family")),
            _escape(_dataset_name(run)),
        ]
        row.extend(_metric_display(_metric_value(run, name)) for name, _ in METRICS)
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")

    details = "".join(_run_details(run, destination) for run in runs)
    analysis = _analysis_summary(runs, warnings)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evaluation comparison</title>
<style>
:root {{ color-scheme: light; font-family: system-ui, sans-serif; }}
body {{ margin: 0 auto; max-width: 1500px; padding: 2rem; color: #202124; background: #f8f9fa; }}
section, details {{ margin: 1.5rem 0; padding: 1rem 1.25rem; background: white; border: 1px solid #dadce0; border-radius: 8px; }}
h1 {{ margin-top: 0; }}
h2 {{ border-bottom: 1px solid #dadce0; padding-bottom: .4rem; }}
table {{ width: 100%; border-collapse: collapse; margin: .5rem 0 1.2rem; }}
th, td {{ padding: .45rem .6rem; border: 1px solid #dadce0; text-align: left; vertical-align: top; }}
th {{ background: #f1f3f4; }}
td {{ white-space: pre-wrap; word-break: break-word; }}
.comparison {{ display: block; overflow-x: auto; }}
.comparison table {{ min-width: 1250px; }}
.metadata th {{ width: 24%; }}
.warnings {{ border-color: #fbbc04; background: #fff8e1; }}
a {{ color: #1769aa; }}
.analysis {{ border-color: #8ab4f8; }}
.analysis li {{ margin: .35rem 0; }}
summary {{ cursor: pointer; font-weight: 600; }}
.note {{ color: #5f6368; }}
</style>
</head>
<body>
<h1>Evaluation comparison</h1>
<p class="note">Generated from model-agnostic evaluation result artifacts. Metrics use each run's primary evaluation split.</p>
{warning_html}
{analysis}
<section class="comparison">
<h2>Run comparison</h2>
<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>
</section>
<section>
<h2>Run metadata</h2>
{details}
</section>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    """Run the static comparison report command."""

    import argparse

    parser = argparse.ArgumentParser(description="Compare generic evaluation results as HTML")
    parser.add_argument("output", type=Path, help="static comparison HTML output")
    parser.add_argument(
        "results",
        nargs="+",
        type=Path,
        help="evaluation result JSON files or run directories",
    )
    args = parser.parse_args(argv)
    destination = render_comparison_report_file(args.results, args.output)
    print(f"Wrote comparison report: {destination}")
    return 0


def render_comparison_report_file(
    result_paths: list[str | Path] | tuple[str | Path, ...], output_path: str | Path
) -> Path:
    """Load generic artifacts and write a self-contained comparison report."""

    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    runs = load_comparison_runs(result_paths)
    destination.write_text(render_comparison_report(runs, destination), encoding="utf-8")
    return destination


if __name__ == "__main__":
    raise SystemExit(main())
