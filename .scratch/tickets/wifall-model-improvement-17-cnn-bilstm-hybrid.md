# Evaluate CNN-BiLSTM hybrid fall detector

Status: open
Type: task
Label: ready-for-agent
Blocked by: completed/wifall-model-improvement-11-cnn.md, completed/wifall-model-improvement-13-lstm-duration-normalized.md

## What to build

Create a hybrid model that uses temporal convolutions to detect short CSI movement patterns, then a bidirectional LSTM to model their order and longer context. Use duration-normalized CSI amplitude as the first input representation.

Start with a small 1D convolution front end over time and subcarriers, followed by a one- or two-layer BiLSTM and a recording-level classifier. Compare against standalone CNN and LSTM runs.

## Safety objective

Select convolution widths, channel counts, recurrent size, dropout, calibration, and threshold from grouped non-test participants only. Prioritize minimum participant fall recall, then specificity and false-alarm rate. Test participants remain untouched until final evaluation.

## Acceptance criteria

- [ ] Documents hybrid tensor shape and whether convolutions operate across time, subcarriers, or both.
- [ ] CNN channels, kernel sizes, BiLSTM size/layers, dropout, optimizer, epochs, and seed are configurable.
- [ ] Architecture and threshold selection use participant-grouped non-test validation only.
- [ ] Supports persisted-model inference from the same duration-normalized representation.
- [ ] Produces model, generic run artifact, dataset audit, and self-contained HTML report.
- [ ] Compares hybrid performance with standalone CNN and LSTM using compatible metrics and splits.
- [ ] Tests cover tensor shapes, determinism, split isolation, threshold selection, persisted inference, and artifact compatibility.
