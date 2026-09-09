# Analyze CNN fall-detection performance

Status: open
Type: task
Label: ready-for-agent
Blocked by: wifall-model-improvement-11-cnn.md

## What to build

Evaluate the CNN against improved SVM, Random Forest, and calibrated ensemble artifacts. Determine whether preserving temporal/subcarrier structure improves safety performance over the rejected MLP and existing models.

Use generic hard-example diagnostics and compare errors at the locked safety-oriented operating point. Explain whether CNN catches falls missed by other models and whether it introduces complementary or unacceptable false alarms.

## Safety objective

Report missed falls first. Include aggregate and per-participant recall/sensitivity, false-positive rate, specificity, precision, F1, ROC-AUC, PR-AUC, and alert volume. Identify any participant or activity with unacceptable misses.

## Acceptance criteria

- [ ] Compares CNN, improved SVM, Random Forest, and ensemble using compatible generic artifacts.
- [ ] Reports accuracy, balanced accuracy, precision, recall, sensitivity, specificity, F1, ROC-AUC, and PR-AUC.
- [ ] Reports CNN false negatives and false positives by participant/group and source activity.
- [ ] Reports CNN errors by duration, available signal-quality metadata, temporal pattern, and score range where available.
- [ ] Identifies CNN-only errors, shared errors, and falls caught by CNN but missed by existing models.
- [ ] Evaluates threshold and optional OR/alert rules using grouped non-test validation only.
- [ ] Test participants remain untouched while selecting architecture, threshold, weighting, or alert rule.
- [ ] Produces a reproducible comparison/report linked to source run artifacts.
- [ ] Tests cover artifact compatibility, score/observation alignment, deterministic summaries, missing metadata, and report rendering.
- [ ] Records recommendation: adopt, reject, or continue investigating CNN, with unresolved safety risks documented.
