# Upgrade temporal CSI feature representation

Status: open
Type: task
Label: ready-for-agent
Blocked by: wifall-model-improvement-03-grouped-validation.md

## What to build

Give Random Forest richer temporal evidence for short fall events. Compare higher-resolution windows and event-focused motion features against the current four-window representation under participant-grouped validation.

## Acceptance criteria

- [ ] Feature configuration supports higher-resolution temporal windows.
- [ ] Features include documented motion, slope, energy, or change-point information.
- [ ] Feature variants are selected or rejected using non-test participant validation.
- [ ] Persisted model artifact contains the complete feature configuration needed for inference.
- [ ] Result artifact compares feature variants using accuracy, balanced accuracy, recall, F1, ROC-AUC, and PR-AUC.
- [ ] Tests verify deterministic extraction, participant isolation, artifact persistence, and generic report compatibility.
