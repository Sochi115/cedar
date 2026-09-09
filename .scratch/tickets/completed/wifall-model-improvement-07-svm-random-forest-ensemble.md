# Blend improved SVM and Random Forest scores

Status: resolved
Type: task
Label: ready-for-agent
Blocked by: completed/wifall-model-improvement-04-regularized-calibrated-random-forest.md

## What to build

Combine calibrated improved-SVM and Random Forest fall scores so the detector can trade SVM recall against Random Forest specificity. Compare both individual models and ensemble output through the generic report.

## Acceptance criteria

- [x] SVM and Random Forest scores are calibrated onto a compatible fall-score scale.
- [x] Blend weight and decision threshold are selected from grouped non-test validation only.
- [x] Test participants remain untouched until final ensemble evaluation.
- [x] Comparison reports accuracy, balanced accuracy, precision, recall, F1, sensitivity, specificity, ROC-AUC, and PR-AUC.
- [x] Ensemble artifact records source runs, calibration, blend weight, threshold, and split protocol.
- [x] Ensemble renders as a self-contained generic HTML report.
- [x] Tests verify score alignment, no test leakage, reproducibility, and complementary-model compatibility.

## Answer

Implemented `wifall.ensemble.train_ensemble` and `wifall cli train-ensemble`. Improved-SVM and Random-Forest positive scores are separately sigmoid-calibrated using train and validation data, then blended. Weight and threshold selection use equal-participant grouped validation only; test scores are evaluated afterward. Ensemble artifacts record source run paths and IDs, calibration, weights, threshold, and split protocol. Generic HTML and comparison reports include ensemble metrics and diagnostics. Added alignment, compatibility, reproducibility, and test-score-isolation tests. Regenerated current SVM, improved-SVM, Random-Forest, ensemble, and comparison artifacts. Verification: `uv run pytest` (41 passed); Ruff clean.
