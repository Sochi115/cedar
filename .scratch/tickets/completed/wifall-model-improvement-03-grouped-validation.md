# Add participant-balanced grouped validation for Random Forest

Status: resolved
Type: task
Label: ready-for-agent
Blocked by: none

## What to build

Make Random Forest model selection robust to participant differences. Use participant-grouped validation across non-test participants and balance participant contributions during training, while preserving untouched final test participants.

## Acceptance criteria

- [x] Hyperparameter and threshold selection uses participant-grouped validation folds.
- [x] Test participants are excluded from every selection and fitting decision.
- [x] Participant contribution is balanced through documented weights or sampling.
- [x] Fold assignments, participant weights, and selection protocol appear in the result artifact.
- [x] Final evaluation reports aggregate and per-participant metrics.
- [x] Existing Random Forest artifact format and static report remain compatible.
- [x] Tests verify participant isolation, reproducibility, split parity, and artifact metadata.

## Answer

Implemented leave-one-participant-out selection across training and validation participants. Random Forest folds use inverse participant-frequency sample weights so each participant contributes equally. Final model fitting uses all non-test participants; test participants remain untouched. Artifacts now record selection participants, held-out test participants, fold assignments, participant weights, fit splits, and evaluation fit status. Added tests and README documentation. Regenerated the Random Forest model and HTML report.
