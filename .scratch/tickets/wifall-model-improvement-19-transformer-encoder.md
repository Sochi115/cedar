# Evaluate small Transformer encoder fall detector

Status: open
Type: task
Label: ready-for-agent
Blocked by: completed/wifall-model-improvement-13-lstm-duration-normalized.md

## What to build

Evaluate a small Transformer encoder over duration-normalized CSI sequences. Project each 52-subcarrier timestep into an embedding, add normalized-time positional information, apply a small encoder stack, and pool sequence outputs into a fall score.

Use a deliberately small configuration because WiFall has few participants. Compare Transformer performance with LSTM, CNN-BiLSTM, TCN, and Random Forest without using test results for design decisions.

## Safety objective

Select embedding size, attention heads, encoder depth, feed-forward width, dropout, optimizer, calibration, and threshold from grouped non-test participants only. Prioritize minimum participant fall recall, then specificity and false-alarm rate. Test participants remain untouched until final evaluation.

## Acceptance criteria

- [ ] Documents embedding, positional encoding, attention, pooling, and output semantics.
- [ ] Model depth, embedding size, attention heads, feed-forward width, dropout, optimizer, epochs, and seed are configurable.
- [ ] Architecture and threshold selection use participant-grouped non-test validation only.
- [ ] Includes regularization or early stopping appropriate for the small participant count.
- [ ] Produces model, generic run artifact, dataset audit, and self-contained HTML report.
- [ ] Compares Transformer performance with compatible duration-normalized model runs.
- [ ] Reports aggregate, per-participant, false-positive, false-negative, and complementarity metrics.
- [ ] Tests cover tensor shapes, deterministic training, split isolation, persisted inference, and artifact compatibility.
