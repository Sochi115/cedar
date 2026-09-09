# Analyze CNN fall-detection performance

Status: resolved
Type: task
Label: ready-for-agent
Blocked by: wifall-model-improvement-11-cnn.md

## What to build

Evaluate the CNN against improved SVM, Random Forest, and calibrated ensemble artifacts. Determine whether preserving temporal/subcarrier structure improves safety performance over the rejected MLP and existing models.

Use generic hard-example diagnostics and compare errors at the locked safety-oriented operating point. Explain whether CNN catches falls missed by other models and whether it introduces complementary or unacceptable false alarms.

## Safety objective

Report missed falls first. Include aggregate and per-participant recall/sensitivity, false-positive rate, specificity, precision, F1, ROC-AUC, PR-AUC, and alert volume. Identify any participant or activity with unacceptable misses.

## Acceptance criteria

- [x] Compares CNN, improved SVM, Random Forest, and ensemble using compatible generic artifacts.
- [x] Reports accuracy, balanced accuracy, precision, recall, sensitivity, specificity, F1, ROC-AUC, and PR-AUC.
- [x] Reports CNN false negatives and false positives by participant/group and source activity.
- [x] Reports CNN errors by duration, available signal-quality metadata, temporal pattern, and score range where available.
- [x] Identifies CNN-only errors, shared errors, and falls caught by CNN but missed by existing models.
- [x] Evaluates threshold and optional OR/alert rules using grouped non-test validation only.
- [x] Test participants remain untouched while selecting architecture, threshold, weighting, or alert rule.
- [x] Produces a reproducible comparison/report linked to source run artifacts.
- [x] Tests cover artifact compatibility, score/observation alignment, deterministic summaries, missing metadata, and report rendering.
- [x] Records recommendation: adopt, reject, or continue investigating CNN, with unresolved safety risks documented.

## Answer

Implemented `wifall.cnn_analysis` and `wifall cli analyze-cnn`. Analysis compares CNN, improved SVM, Random Forest, and ensemble artifacts; reports safety metrics, participant-level recall/specificity, source/activity errors, duration/signal-quality diagnostics, score ranges, shared and complementary errors, and validation-only operating points. Test participants are diagnostic-only. Current recommendation: `continue_investigating`; CNN catches some falls missed by comparator models, but test specificity is only 9.375% with 29 false positives. No validation operating point met 90%, 95%, or 100% minimum group recall with useful specificity. Verification: `52 tests passed`; Ruff and compile checks passed.
