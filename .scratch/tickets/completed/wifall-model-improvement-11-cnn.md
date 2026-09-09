# Train simple CNN fall detector

Status: resolved
Type: task
Label: ready-for-agent
Blocked by: completed/wifall-model-improvement-03-grouped-validation.md, completed/wifall-model-improvement-05-temporal-features.md

## What to build

Prototype and train a small CNN for WiFall fall detection using a structured CSI representation rather than the flattened MLP vector. Preserve temporal and subcarrier axes in a fixed-size tensor, document the tensor shape, and compare a minimal 1D or 2D convolutional design before adding complexity.

The first implementation may add a focused neural-network dependency if needed, but must keep inference and evaluation reproducible from repository commands. Do not build a deep sequence model or production-serving stack.

## Candidate representation

Start with per-recording normalized CSI amplitude arranged as temporal windows by subcarrier. Record how variable-length recordings become fixed-size input and whether the model sees raw window values, window statistics, or both. Keep phase out of first experiment unless separately justified.

## Safety objective

Prioritize fall recall. Select architecture, regularization, class balancing, calibration, and threshold from grouped non-test validation only. Support a minimum validation recall target and select specificity or alert volume only among eligible operating points. Test participants remain untouched until final evaluation.

## Acceptance criteria

- [x] CNN input representation and tensor shape are documented and deterministic.
- [x] A small configurable CNN architecture is implemented with explicit convolution, pooling, regularization, optimizer, epochs, batch size, and random seed.
- [x] Variable-length recordings are transformed without participant or split leakage.
- [x] Scaling, normalization, augmentation, and class balancing fit or apply only to non-test training data.
- [x] Architecture and safety-oriented threshold are selected from grouped non-test validation only.
- [x] Test participants remain untouched until final CNN evaluation.
- [x] CNN persists model, generic run artifact, dataset audit if applicable, and self-contained HTML report.
- [x] Artifact records tensor representation, architecture, parameters, score orientation, calibration, threshold, and split protocol.
- [x] CNN output is compatible with generic comparison and hard-example reports.
- [x] Tests cover deterministic preprocessing, tensor shape, participant isolation, threshold selection, artifact persistence, and inference reproduction.

## Answer

Implemented `wifall.cnn` and `wifall cli train-cnn` using a small 2D CNN over normalized `1 × time × subcarrier` tensors. Variable-length recordings are deterministically resampled to fixed shape. Channel/dropout candidates and safety threshold are selected with participant-grouped non-test OOF validation; final network fits non-test participants only. Persisted artifacts include model, audit, generic evaluation, hard-example diagnostics, and self-contained report. Added preprocessing, threshold, isolation, artifact, inference, and shape-validation tests. Verification: `49 tests passed`; Ruff and compile checks passed.

Initial CNN result: test recall/sensitivity 93.75%, specificity 9.375%, precision 50.85%, ROC-AUC 0.7813, 2 false negatives, and 29 false positives. CNN shows better false-negative performance than Random Forest but remains unsuitable without further analysis; ticket 12 remains open.
