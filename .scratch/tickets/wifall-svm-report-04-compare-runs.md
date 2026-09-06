# Compare model and dataset runs

Status: open
Type: task
Label: ready-for-agent
Blocked by: wifall-svm-report-02-train-svm.md, wifall-svm-report-03-static-report.md

## What to build

Generate a static comparison report that indexes multiple compatible run artifacts, supports future model families and datasets, and links each run to its detailed evaluation report.

## Acceptance criteria

- [ ] Comparison report generates an index linking to individual run reports.
- [ ] Runs can be compared by accuracy, balanced accuracy, precision, recall, F1, sensitivity, specificity, and false-positive rate.
- [ ] Comparison view displays model type, dataset, task, split strategy, and run metadata.
- [ ] Report warns when runs use incompatible labels, tasks, or evaluation protocols.
- [ ] Adding a future run requires only a valid run artifact; no visualization code changes are needed.
- [ ] Comparison output remains self-contained static HTML.
- [ ] Tests cover multiple model types, multiple dataset metadata values, incompatible runs, and missing optional metrics.
