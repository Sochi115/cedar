# Evaluate temporal convolutional network fall detector

Status: open
Type: task
Label: ready-for-agent
Blocked by: completed/wifall-model-improvement-13-lstm-duration-normalized.md

## What to build

Evaluate a temporal convolutional network as a recurrent-free alternative to the LSTM. Use residual 1D convolution blocks with dilation to capture short movement bursts and longer temporal context from duration-normalized CSI sequences.

Keep the first network small. Compare causal and non-causal semantics explicitly; offline non-causal processing is acceptable for recording-level classification, but real-time implications must be documented.

## Safety objective

Select channel count, kernel size, dilation schedule, block count, dropout, optimizer, calibration, and threshold from grouped non-test participants only. Prioritize minimum participant fall recall, then specificity and false-alarm rate. Test participants remain untouched until final evaluation.

## Acceptance criteria

- [ ] Implements configurable residual temporal convolution blocks and dilation schedule.
- [ ] Documents input shape, receptive field, causal/non-causal behavior, and pooling strategy.
- [ ] Architecture and threshold selection use participant-grouped non-test validation only.
- [ ] Produces model, generic run artifact, dataset audit, and self-contained HTML report.
- [ ] Compares TCN against compatible CNN, LSTM, and Random Forest runs.
- [ ] Reports aggregate and per-participant recall, specificity, false-positive rate, F1, and error overlap.
- [ ] Tests cover receptive-field shape handling, determinism, split isolation, persisted inference, and artifact compatibility.
