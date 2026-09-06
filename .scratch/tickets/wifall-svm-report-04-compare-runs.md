# Compare baseline, improved SVM, and Random Forest runs

Status: open
Type: task
Label: ready-for-agent
Blocked by: wifall-model-improvement-01-improved-svm.md, wifall-model-improvement-02-random-forest.md

## What to build

Generate a static comparison report that indexes the current baseline SVM, improved SVM, and Random Forest run artifacts. Report model family, training code, feature configuration, threshold policy, and metrics. Keep support for future model families and datasets.

## Acceptance criteria

- [ ] Comparison report generates an index linking to individual run reports.
- [ ] Runs can be compared by accuracy, balanced accuracy, precision, recall, F1, sensitivity, specificity, false-positive rate, ROC-AUC, and PR-AUC.
- [ ] Comparison view displays model family, training code identifier, dataset, task, feature configuration, threshold policy, split strategy, and run metadata.
- [ ] Report warns when runs use incompatible labels, tasks, or evaluation protocols.
- [ ] Comparison identifies current baseline SVM, improved SVM, and Random Forest as separate runs and artifacts.
- [ ] Adding a future run requires only a valid run artifact; no visualization code changes are needed.
- [ ] Comparison output remains self-contained static HTML.
- [ ] Tests cover multiple model types, multiple dataset metadata values, incompatible runs, and missing optional metrics.
