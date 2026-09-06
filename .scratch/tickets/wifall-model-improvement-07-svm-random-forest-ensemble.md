# Blend improved SVM and Random Forest scores

Status: open
Type: task
Label: ready-for-agent
Blocked by: completed/wifall-model-improvement-04-regularized-calibrated-random-forest.md

## What to build

Combine calibrated improved-SVM and Random Forest fall scores so the detector can trade SVM recall against Random Forest specificity. Compare both individual models and ensemble output through the generic report.

## Acceptance criteria

- [ ] SVM and Random Forest scores are calibrated onto a compatible fall-score scale.
- [ ] Blend weight and decision threshold are selected from grouped non-test validation only.
- [ ] Test participants remain untouched until final ensemble evaluation.
- [ ] Comparison reports accuracy, balanced accuracy, precision, recall, F1, sensitivity, specificity, ROC-AUC, and PR-AUC.
- [ ] Ensemble artifact records source runs, calibration, blend weight, threshold, and split protocol.
- [ ] Ensemble renders as a self-contained generic HTML report.
- [ ] Tests verify score alignment, no test leakage, reproducibility, and complementary-model compatibility.
