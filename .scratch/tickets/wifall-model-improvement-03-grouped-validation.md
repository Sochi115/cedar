# Add participant-balanced grouped validation for Random Forest

Status: open
Type: task
Label: ready-for-agent
Blocked by: none

## What to build

Make Random Forest model selection robust to participant differences. Use participant-grouped validation across non-test participants and balance participant contributions during training, while preserving untouched final test participants.

## Acceptance criteria

- [ ] Hyperparameter and threshold selection uses participant-grouped validation folds.
- [ ] Test participants are excluded from every selection and fitting decision.
- [ ] Participant contribution is balanced through documented weights or sampling.
- [ ] Fold assignments, participant weights, and selection protocol appear in the result artifact.
- [ ] Final evaluation reports aggregate and per-participant metrics.
- [ ] Existing Random Forest artifact format and static report remain compatible.
- [ ] Tests verify participant isolation, reproducibility, split parity, and artifact metadata.
