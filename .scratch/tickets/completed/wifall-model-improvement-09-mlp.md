# Train simple MLP fall detector

Status: resolved
Type: task
Label: ready-for-agent
Blocked by: completed/wifall-model-improvement-03-grouped-validation.md, completed/wifall-model-improvement-05-temporal-features.md

## What to build

Train a small Multi-Layer Perceptron on fixed-size normalized CSI temporal features. Add MLP as a separate model family without changing baseline SVM, improved SVM, or Random Forest artifacts.

Use the existing recording-level feature seam first. Prefer a small regularized architecture over raw variable-length CSI or a recurrent network. Record whether training uses deterministic fall balancing or class weighting.

## Safety objective

Optimize fall recall before precision. Select decision threshold from grouped non-test validation only. Support a configurable minimum validation recall target and select specificity or accuracy only among eligible thresholds. Test participants remain untouched until final evaluation.

## Acceptance criteria

- [x] MLP uses fixed-size normalized temporal CSI features with documented configuration.
- [x] Architecture, optimizer/solver, regularization, random seed, balancing method, and feature configuration are configurable and persisted.
- [x] Scaling and any balancing fit or apply only to training participants.
- [x] Hyperparameters and threshold are selected from grouped non-test validation only.
- [x] Threshold selection supports a safety-oriented minimum recall target.
- [x] Test participants remain untouched until final MLP evaluation.
- [x] MLP persists model, generic run artifact, dataset audit if applicable, and self-contained HTML report.
- [x] Artifact records model family, architecture, parameters, score orientation, threshold, calibration, and split protocol.
- [x] MLP output is compatible with generic comparison and hard-example reports.
- [x] Tests cover deterministic training, participant isolation, feature/scaler fit isolation, recall-constrained threshold selection, artifact persistence, and inference reproduction.

## Answer

Implemented `wifall.mlp` and `wifall cli train-mlp`. The pipeline uses normalized four-window CSI amplitude statistics, training-only scaling and deterministic fall balancing, participant-grouped out-of-fold selection across all non-test participants, and a minimum-group-recall threshold policy. It persists model, audit, generic run artifact, hard-example diagnostics, and self-contained HTML report. Added deterministic extraction, threshold, artifact, isolation, and inference tests. Full verification: `uv run pytest` (45 passed); Ruff and compile checks passed.

Regenerated MLP result: test recall/sensitivity 1.0 with 0.0 specificity and 32 false positives. This safety-first operating point catches all test falls but is not yet a practical detector; ticket 10 should analyze and compare it before adoption.
