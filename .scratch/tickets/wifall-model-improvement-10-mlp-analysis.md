# Analyze MLP fall-detection performance

Status: open
Type: task
Label: ready-for-agent
Blocked by: wifall-model-improvement-09-mlp.md

## What to build

Evaluate the MLP against improved SVM, Random Forest, and the calibrated ensemble using generic run artifacts. Explain whether MLP catches falls that other models miss, whether it introduces complementary false alarms, and whether its safety-oriented operating point is useful.

Use generic hard-example diagnostics for participant, activity, duration, signal quality, score, and recording-level analysis. Keep model and threshold selection separate from final test reporting.

## Safety objective

Prioritize missed-fall reduction. Report recall and sensitivity first, then false-positive rate, specificity, precision, and alert volume. Include per-participant recall and identify any participant with unacceptable validation or test misses.

## Acceptance criteria

- [ ] Compares MLP, improved SVM, Random Forest, and ensemble using compatible generic artifacts.
- [ ] Reports accuracy, balanced accuracy, precision, recall, sensitivity, specificity, F1, ROC-AUC, and PR-AUC.
- [ ] Reports false negatives and false positives by participant/group and source activity.
- [ ] Reports MLP errors by duration, available signal-quality metadata, and score range.
- [ ] Identifies MLP-only errors, errors shared with other models, and errors caught by MLP but missed by other models.
- [ ] Evaluates safety-oriented threshold and optional OR/alert rule using grouped non-test validation only.
- [ ] Test participants remain untouched while selecting architecture, threshold, weights, or alert rule.
- [ ] Produces a reproducible comparison/report linked to source run artifacts.
- [ ] Tests cover artifact compatibility, score/observation alignment, deterministic summaries, missing metadata, and report rendering.
- [ ] Records recommendation: adopt, reject, or continue investigating MLP, with unresolved risks documented.
