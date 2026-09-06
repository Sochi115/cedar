# Train and evaluate WiFall SVM baseline

Status: open
Type: task
Label: ready-for-agent
Blocked by: wifall-svm-report-01-load-and-split.md

## What to build

Train and evaluate a reproducible SVM fall detector using the WiFall pipeline, with participant-held-out evaluation and a model-agnostic run artifact that captures everything required for later reporting.

## Acceptance criteria

- [ ] Baseline uses documented CSI preprocessing and recording-level features.
- [ ] Normalization and feature transforms are fitted on training data only.
- [ ] Complete preprocessing and SVM pipeline is persisted as a loadable model artifact.
- [ ] Held-out predictions and decision scores are generated without refitting.
- [ ] Evaluation reports accuracy, balanced accuracy, precision, recall/sensitivity, specificity, F1, false-positive rate, and support.
- [ ] Run artifact contains dataset/task metadata, model type/configuration, feature/preprocessing metadata, split strategy, label order, true labels, predictions, optional scores, metrics, and confusion matrix.
- [ ] Artifact format does not require visualization code to know SVM-specific implementation details.
- [ ] Tests verify no training-data leakage, stable artifact contents, and reproducible evaluation.
