# Evaluate stacked LSTM fall detector

Status: open
Type: task
Label: ready-for-agent
Blocked by: completed/wifall-model-improvement-13-lstm-duration-normalized.md

## What to build

Add a configurable stacked LSTM with two or three recurrent layers for duration-normalized CSI sequences. Compare it against the tuned one-layer LSTM baseline without changing existing model artifacts.

Use dropout between recurrent layers, optional layer normalization, and a documented final-state or sequence-pooling strategy. Keep architecture search small enough for participant-grouped validation.

## Safety objective

Select layer count, hidden size, regularization, calibration, and threshold from grouped non-test participants only. Prioritize minimum participant fall recall, then specificity and false-alarm rate. Test participants remain untouched until final evaluation.

## Acceptance criteria

- [ ] Supports configurable layer count, hidden size, inter-layer dropout, optimizer, epochs, and seed.
- [ ] Documents how stacked recurrent outputs become one recording-level fall score.
- [ ] Architecture and threshold selection use participant-grouped non-test validation only.
- [ ] Compares one-layer and stacked LSTM results using identical duration-normalized inputs and split protocol.
- [ ] Produces model, generic run artifact, dataset audit, and self-contained HTML report.
- [ ] Reports aggregate, per-participant, false-positive, and false-negative metrics.
- [ ] Tests cover deterministic training, participant isolation, shape handling, persisted inference, and artifact compatibility.
