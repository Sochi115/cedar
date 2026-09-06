# Build generic model validation and static evaluation report

Status: resolved
Type: task
Label: ready-for-agent
Blocked by: wifall-svm-report-02-train-svm.md

## What to build

Create generic classification validation code and a generic static HTML renderer. The validator accepts any model with a prediction interface and any validation dataset represented by features and labels. It produces a model-agnostic evaluation result artifact. The HTML renderer consumes only that artifact.

WiFall, WiFi CSI, SVM, and dataset-loader imports must stay outside generic validation and report code. The WiFall SVM baseline must use this validator instead of its own metric implementation.

## Acceptance criteria

- [x] Generic validator evaluates a model on supplied validation features and labels without fitting or refitting the model.
- [x] Generic validator supports models with `predict` and optional `decision_function` or `predict_proba`.
- [x] Generic result artifact contains run, dataset, task, model, feature, preprocessing, and split metadata.
- [x] Generic result artifact contains true labels, predicted labels, optional scores, label order, confusion matrix, overall metrics, per-class metrics, macro averages, and weighted averages.
- [x] Binary artifacts include sensitivity, specificity, precision, F1, and false-positive rate.
- [x] ROC-AUC and PR-AUC are included only when compatible scores and labels exist; unavailable values are explicit.
- [x] WiFall SVM training uses generic validation code and adds only WiFall-specific feature and split metadata.
- [x] Static HTML consumes only the generic result artifact and does not import model or dataset code.
- [x] HTML renders confusion matrix counts and row percentages, overall metrics, per-class metrics, binary metrics, optional AUC metrics, and metadata.
- [x] HTML clearly marks unavailable optional metrics.
- [x] Generic fixture with a different model name and dataset renders without renderer changes.
- [x] Generated HTML is self-contained and works without a web server or project runtime dependency.
- [x] Tests cover generic validation, no refitting, models without scores, artifact serialization, and HTML content.

## Answer

Implemented generic classification validation and static reporting. Validation accepts any prediction model, feature matrix, and label sequence. It calls `predict` without fitting and uses `decision_function` or `predict_proba` when available. It produces one JSON evaluation result format with metadata, predictions, scores, confusion matrix, overall metrics, per-class metrics, macro averages, weighted averages, and optional AUC metrics.

Moved metric generation out of WiFall SVM code. WiFall SVM now supplies features, labels, and WiFi CSI metadata to the generic validator. Generic code has no WiFall, CSI, or SVM imports.

Added self-contained HTML rendering and a generic report CLI. Report renderer consumes only the JSON evaluation result. It renders dataset/model/task metadata, confusion matrix counts and row percentages, metrics, averages, and unavailable-score notices.

Added generic fixture tests with a different dataset and model name. Verified full WiFall SVM training produces a generic result artifact and static HTML report.

Validation: full pytest suite, Ruff, `uv lock --check`, and Python compilation pass.
