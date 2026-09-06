# Generate static HTML evaluation report

Status: open
Type: task
Label: ready-for-agent
Blocked by: wifall-svm-report-02-train-svm.md

## What to build

Generate a self-contained, browser-openable HTML evaluation report from one model-agnostic run artifact without retraining or importing model implementation code.

## Acceptance criteria

- [ ] Report renders confusion matrix with counts and percentages.
- [ ] Report renders overall and per-class metrics.
- [ ] Report shows fall-specific sensitivity, specificity, precision, F1, and false-positive rate.
- [ ] Report shows macro and weighted averages where applicable.
- [ ] ROC-AUC and PR-AUC are shown when scores exist and clearly omitted when unavailable.
- [ ] Report displays dataset, task, model, feature, split, and run metadata.
- [ ] Report can render a fixture artifact with different model and dataset metadata without renderer changes.
- [ ] Generated HTML works without a web server or project runtime dependency.
- [ ] Tests verify report generation and key metric/confusion-matrix content.
