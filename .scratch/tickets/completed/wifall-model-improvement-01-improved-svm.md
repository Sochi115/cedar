# Build separate improved WiFall SVM training pipeline

Status: resolved
Type: task
Label: ready-for-agent
Blocked by: none

## What to build

Create separate improved SVM training code and a separate model output for WiFall fall detection. Keep the current baseline training code, model artifact, and result artifact unchanged.

The improved pipeline combines data audit, temporal and normalized CSI features, participant-grouped validation, SVM parameter selection, score orientation, and validation threshold selection. It must use the existing generic validation and static reporting contracts.

## Acceptance criteria

- [x] Existing baseline training command and outputs continue to work unchanged.
- [x] Improved SVM has a separate training entry point and separate output directory.
- [x] Dataset audit reports participant/activity counts, recording durations, row counts, malformed rows, empty recordings, label mismatches, and fall/non-fall balance.
- [x] Feature extraction adds documented temporal or windowed CSI features and documented per-recording normalization.
- [x] Complete recordings and participants remain within one split.
- [x] SVM parameters are selected with participant-grouped validation.
- [x] Positive-class scores are explicit and oriented toward `fall`.
- [x] Decision threshold is selected from training/validation data only; default policy maximizes validation F1 with sensitivity as tie-breaker.
- [x] Test participants are not used for feature fitting, parameter selection, threshold selection, or model selection.
- [x] Evaluation includes aggregate and per-participant metrics, confusion matrices, score distributions, and threshold metrics.
- [x] Separate model artifact contains the complete feature, normalization, and SVM pipeline.
- [x] Separate generic result artifact records training code identifier, feature configuration, SVM parameters, threshold, split configuration, and metrics.
- [x] Static HTML report can display the improved run without changes to generic evaluation/report code.
- [x] Tests cover audit output, temporal features, normalization boundaries, grouped selection, threshold selection, and reproducible training.

## Answer

Implemented separate improved SVM training through `train-improved-svm`. Baseline `train-svm` code and outputs remain unchanged.

Improved pipeline adds per-recording CSI amplitude median/MAD normalization, four temporal windows, grouped parameter selection, positive-class score orientation, validation F1/sensitivity threshold selection, participant metrics, score distributions, and threshold tables.

Training persists four outputs: `model.joblib`, `dataset_audit.json`, `run.json`, and `report.html`. `run.json` uses the generic evaluation artifact contract. `report.html` uses the generic static renderer.

Verified full downloaded WiFall run: 928 valid recordings, 10 participants, 232,941 CSI rows. Test evaluation completed with 96.875% fall sensitivity and 81.25% accuracy for this selected configuration.

Validation: full pytest suite, Ruff, Python compilation, and full improved training run pass.
