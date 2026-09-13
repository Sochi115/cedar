# Tune causal LSTM baseline for real-time alerts

Status: resolved
Type: task
Label: ready-for-agent
Blocked by: completed/wifall-model-improvement-13-lstm-duration-normalized.md

## What to build

Improve current duration-normalized LSTM training for low-latency alerts. Train the existing unidirectional LSTM for enough epochs to converge, add early stopping and learning-rate scheduling, and search a small set of hidden sizes, layers, dropout values, and optimizer settings.

Keep the model causal: each prediction may use only current and past samples. Define the rolling-window or recurrent-state inference contract, state reset behavior, alert cadence, and maximum lookback. Do not add bidirectional processing because it requires future samples and conflicts with minimal-delay alerts.

## Safety objective

Select all training parameters, architecture variants, calibration, and decision threshold from grouped non-test participants only. Prioritize minimum participant fall recall, then specificity and false-alarm rate. Test participants remain untouched until final evaluation.

## Acceptance criteria

- [x] Unidirectional LSTM remains the only recurrent architecture in this ticket.
- [x] Hidden size, layer count, dropout, epoch budget, learning rate, weight decay, and seed are configurable and persisted.
- [x] Training supports early stopping and learning-rate scheduling using non-test validation only.
- [x] Inference uses only current and past samples with documented rolling-window or recurrent-state semantics.
- [x] Maximum inference lookback, alert cadence, state reset behavior, and expected alert delay are documented.
- [x] Architecture and threshold selection use participant-grouped non-test validation only.
- [x] Produces model, generic run artifact, dataset audit, and self-contained HTML report.
- [x] Report includes alert latency and false-alarm behavior, not only recording-level metrics.
- [x] Tests cover determinism, tensor shapes, causal inference, selection isolation, early stopping, persisted inference, and artifact compatibility.

## Answer

Implemented causal rolling-window LSTM training and inference. The model uses only fixed-rate current/past samples, per-window median/MAD normalization, a 32-sample maximum lookback, one-sample alert cadence, and stateless session resets. Raw non-normalized recordings are rejected by persisted-model inference.

Added grouped outer OOF selection with inner participant validation for early stopping and `ReduceLROnPlateau`, configurable Adam/AdamW optimizer candidates, hidden size, layer count, dropout, learning rate, weight decay, epoch, and seed. Reports include alert latency in samples and estimated seconds plus false-alarm behavior.

Generated report: `data/processed/wifall/lstm-duration-normalized-run/report.html`

Validation: `80 passed`; Ruff clean.
