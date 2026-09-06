# Upgrade temporal CSI feature representation

Status: resolved
Type: task
Label: ready-for-agent
Blocked by: completed/wifall-model-improvement-03-grouped-validation.md

## What to build

Give Random Forest richer temporal evidence for short fall events. Compare higher-resolution windows and event-focused motion features against the current four-window representation under participant-grouped validation.

## Acceptance criteria

- [x] Feature configuration supports higher-resolution temporal windows.
- [x] Features include documented motion, slope, energy, or change-point information.
- [x] Feature variants are selected or rejected using non-test participant validation.
- [x] Persisted model artifact contains the complete feature configuration needed for inference.
- [x] Result artifact compares feature variants using accuracy, balanced accuracy, recall, F1, ROC-AUC, and PR-AUC.
- [x] Tests verify deterministic extraction, participant isolation, artifact persistence, and generic report compatibility.

## Answer

Implemented configurable Random Forest temporal feature variants. Existing four-window statistics remain control; default search compares them with an eight-window motion representation containing delta magnitude, slope, energy, peak-to-peak, and change-point features. Variant selection uses grouped non-test validation only. Persisted model artifacts contain selected feature configuration, and run artifacts compare variants with accuracy, balanced accuracy, recall, F1, ROC-AUC, and PR-AUC. Regenerated the Random Forest model and report.
