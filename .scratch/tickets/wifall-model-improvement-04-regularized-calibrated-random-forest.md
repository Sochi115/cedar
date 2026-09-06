# Tune and calibrate Random Forest fall detection

Status: open
Type: task
Label: ready-for-agent
Blocked by: completed/wifall-model-improvement-03-grouped-validation.md

## What to build

Improve Random Forest generalization by searching regularized model configurations and calibrating fall scores across unseen participants. Select model parameters, calibration, and decision threshold through the grouped non-test validation protocol.

## Acceptance criteria

- [ ] Search covers tree count, maximum depth, minimum leaf size, feature fraction, bootstrap sample size, and split criterion.
- [ ] Selection uses participant-balanced grouped validation only.
- [ ] Fall-score calibration is fit without test-participant data.
- [ ] Threshold policy supports explicit balanced-accuracy, F1, and recall-constrained objectives.
- [ ] Result artifact records candidate parameters, calibration configuration, score orientation, threshold policy, and selected values.
- [ ] Aggregate, per-participant, confusion-matrix, and score metrics render in the generic HTML report.
- [ ] Tests verify no test leakage, reproducibility, calibration metadata, and comparison with current Random Forest baseline.
