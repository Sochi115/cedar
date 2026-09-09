# Analyze MLP fall-detection performance

Status: resolved
Type: task
Label: wontfix
Blocked by: completed/wifall-model-improvement-09-mlp.md

## What to build

Evaluate the MLP against improved SVM, Random Forest, and the calibrated ensemble using generic run artifacts. Explain whether MLP catches falls that other models miss, whether it introduces complementary false alarms, and whether its safety-oriented operating point is useful.

Use generic hard-example diagnostics for participant, activity, duration, signal quality, score, and recording-level analysis. Keep model and threshold selection separate from final test reporting.

## Safety objective

Prioritize missed-fall reduction. Report recall and sensitivity first, then false-positive rate, specificity, precision, and alert volume. Include per-participant recall and identify any participant with unacceptable validation or test misses.

## Acceptance criteria

- [x] Initial MLP run evaluated against existing safety requirements.
- [x] MLP rejected as primary detector because safety-oriented operation produced 100% recall but 0% specificity and 32 false positives.

## Answer

Closed as `wontfix`. MLP research will not continue. Safety-first MLP reevaluation retained all test falls but classified every test recording as a fall, producing 32 false positives and 0% specificity. Additional architecture, PCA, partial balancing, grouped OOF calibration, and non-test refitting did not produce a usable operating point. Existing improved SVM remains preferred safety candidate; Random Forest and ensemble remain comparison candidates. No further MLP analysis or alert-rule tuning is justified.
