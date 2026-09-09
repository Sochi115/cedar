# Train simple MLP fall detector

Status: claimed
Type: task
Label: ready-for-agent
Blocked by: completed/wifall-model-improvement-03-grouped-validation.md, completed/wifall-model-improvement-05-temporal-features.md

## What to build

Train a small Multi-Layer Perceptron on fixed-size normalized CSI temporal features. Add MLP as a separate model family without changing baseline SVM, improved SVM, or Random Forest artifacts.

Use the existing recording-level feature seam first. Prefer a small regularized architecture over raw variable-length CSI or a recurrent network. Record whether training uses deterministic fall balancing or class weighting.

## Safety objective

Optimize fall recall before precision. Select decision threshold from grouped non-test validation only. Support a configurable minimum validation recall target and select specificity or accuracy only among eligible thresholds. Test participants remain untouched until final evaluation.

## Acceptance criteria

- [ ] MLP uses fixed-size normalized temporal CSI features with documented configuration.
- [ ] Architecture, optimizer/solver, regularization, random seed, balancing method, and feature configuration are configurable and persisted.
- [ ] Scaling and any balancing fit or apply only to training participants.
- [ ] Hyperparameters and threshold are selected from grouped non-test validation only.
- [ ] Threshold selection supports a safety-oriented minimum recall target.
- [ ] Test participants remain untouched until final MLP evaluation.
- [ ] MLP persists model, generic run artifact, dataset audit if applicable, and self-contained HTML report.
- [ ] Artifact records model family, architecture, parameters, score orientation, threshold, calibration, and split protocol.
- [ ] MLP output is compatible with generic comparison and hard-example reports.
- [ ] Tests cover deterministic training, participant isolation, feature/scaler fit isolation, recall-constrained threshold selection, artifact persistence, and inference reproduction.
