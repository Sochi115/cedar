# Train separate WiFall Random Forest model

Status: open
Type: task
Label: ready-for-agent
Blocked by: wifall-model-improvement-01-improved-svm.md

## What to build

Create separate Random Forest training code and a separate model output for WiFall fall detection. Use the improved SVM pipeline's agreed feature representation, participant splits, generic validation contract, and static reporting contract.

## Acceptance criteria

- [ ] Random Forest has a separate training entry point and does not overwrite baseline or improved SVM artifacts.
- [ ] Random Forest uses the same participant-held-out split and agreed feature representation as improved SVM.
- [ ] Random Forest parameters are selected with training/validation participants only.
- [ ] Fall score orientation and decision threshold are explicit and selected without test-data leakage.
- [ ] Final test evaluation reports aggregate and per-participant metrics.
- [ ] Separate Random Forest model artifact contains feature preprocessing and model configuration needed for inference.
- [ ] Separate generic result artifact records model family, training code identifier, feature configuration, parameters, threshold, split configuration, and metrics.
- [ ] Random Forest result renders through the existing generic static HTML report.
- [ ] Tests verify artifact separation, participant split parity, no test leakage, reproducibility, and generic report compatibility.
