# Train separate WiFall Random Forest model

Status: resolved
Type: task
Label: ready-for-agent
Blocked by: completed/wifall-model-improvement-01-improved-svm.md

## What to build

Create separate Random Forest training code and a separate model output for WiFall fall detection. Use the improved SVM pipeline's agreed feature representation, participant splits, generic validation contract, and static reporting contract.

## Acceptance criteria

- [x] Random Forest has a separate training entry point and does not overwrite baseline or improved SVM artifacts.
- [x] Random Forest uses the same participant-held-out split and agreed feature representation as improved SVM.
- [x] Random Forest parameters are selected with training/validation participants only.
- [x] Fall score orientation and decision threshold are explicit and selected without test-data leakage.
- [x] Final test evaluation reports aggregate and per-participant metrics.
- [x] Separate Random Forest model artifact contains feature preprocessing and model configuration needed for inference.
- [x] Separate generic result artifact records model family, training code identifier, feature configuration, parameters, threshold, split configuration, and metrics.
- [x] Random Forest result renders through the existing generic static HTML report.
- [x] Tests verify artifact separation, participant split parity, no test leakage, reproducibility, and generic report compatibility.

## Answer

Implemented separate Random Forest training in `wifall.random_forest` with the improved SVM feature representation, participant-held-out split validation, training/validation-only parameter and threshold selection, explicit fall-probability orientation, per-participant metrics, persisted model and generic result artifacts, dataset audit, and self-contained HTML report. Added `train-random-forest` CLI entry point, public loading/training APIs, README workflow documentation, and reproducibility/report compatibility tests.
