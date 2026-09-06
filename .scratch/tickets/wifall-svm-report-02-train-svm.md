# Train and evaluate WiFall SVM baseline

Status: resolved
Type: task
Label: ready-for-agent
Blocked by: wifall-svm-report-01-load-and-split.md

## What to build

Train and evaluate a reproducible SVM fall detector using the WiFall pipeline, with participant-held-out evaluation and a model-agnostic run artifact that captures everything required for later reporting.

## Acceptance criteria

- [x] Baseline uses documented CSI preprocessing and recording-level features.
- [x] Normalization and feature transforms are fitted on training data only.
- [x] Complete preprocessing and SVM pipeline is persisted as a loadable model artifact.
- [x] Held-out predictions and decision scores are generated without refitting.
- [x] Evaluation reports accuracy, balanced accuracy, precision, recall/sensitivity, specificity, F1, false-positive rate, and support.
- [x] Run artifact contains dataset/task metadata, model type/configuration, feature/preprocessing metadata, split strategy, label order, true labels, predictions, optional scores, metrics, and confusion matrix.
- [x] Artifact format does not require visualization code to know SVM-specific implementation details.
- [x] Tests verify no training-data leakage, stable artifact contents, and reproducible evaluation.

## Answer

Implemented reproducible WiFall SVM training and evaluation. The baseline extracts 260 recording-level CSI amplitude statistics, fits `StandardScaler` and `SVC` on training participants only, evaluates train/validation/test splits, and persists the complete prediction pipeline with joblib.

Each run writes a model-agnostic JSON artifact containing dataset/task/model metadata, feature and preprocessing descriptions, participant split information, labels, predictions, decision scores, confusion matrices, and binary fall-detection metrics. Added CLI support through `train-svm`.

Verified against downloaded WiFall data. Full dataset training completes successfully and produces `model.joblib` plus `run.json`. Tests cover feature determinism, artifact contents, training-only scaler fitting, participant leakage prevention, loaded-model predictions, and reproducible artifacts.

Validation: full pytest suite, Ruff, and `uv lock --check` pass.
