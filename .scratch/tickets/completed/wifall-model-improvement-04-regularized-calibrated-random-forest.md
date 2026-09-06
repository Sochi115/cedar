# Tune and calibrate Random Forest fall detection

Status: resolved
Type: task
Label: ready-for-agent
Blocked by: completed/wifall-model-improvement-03-grouped-validation.md

## What to build

Improve Random Forest generalization by searching regularized model configurations and calibrating fall scores across unseen participants. Select model parameters, calibration, and decision threshold through the grouped non-test validation protocol.

## Acceptance criteria

- [x] Search covers tree count, maximum depth, minimum leaf size, feature fraction, bootstrap sample size, and split criterion.
- [x] Selection uses participant-balanced grouped validation only.
- [x] Fall-score calibration is fit without test-participant data.
- [x] Threshold policy supports explicit balanced-accuracy, F1, and recall-constrained objectives.
- [x] Result artifact records candidate parameters, calibration configuration, score orientation, threshold policy, and selected values.
- [x] Aggregate, per-participant, confusion-matrix, and score metrics render in the generic HTML report.
- [x] Tests verify no test leakage, reproducibility, calibration metadata, and comparison with current Random Forest baseline.

## Answer

Implemented regularized Random Forest search across tree count, depth, leaf size, feature fraction, bootstrap sample size, and split criterion. Added sigmoid and isotonic score calibration fitted from grouped non-test out-of-fold scores. Added explicit F1, balanced-accuracy, and recall-constrained threshold policies. Persisted calibration, candidate parameters, selected configuration, score orientation, and threshold metadata in generic artifacts. Added tests, README documentation, and regenerated the Random Forest report.
