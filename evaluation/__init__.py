"""Generic model validation and static evaluation reporting."""

from .core import (
    EvaluationError,
    evaluate_classifier,
    load_evaluation_result,
    save_evaluation_result,
)
from .diagnostics import HardExampleObservation, analyze_hard_examples
from .report import render_html_report, render_html_report_file

__all__ = [
    "EvaluationError",
    "HardExampleObservation",
    "analyze_hard_examples",
    "evaluate_classifier",
    "load_evaluation_result",
    "render_html_report",
    "render_html_report_file",
    "save_evaluation_result",
]
