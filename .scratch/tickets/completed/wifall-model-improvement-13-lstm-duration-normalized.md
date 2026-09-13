# Train simple LSTM on duration-normalized CSI

Status: resolved
Type: task
Label: ready-for-agent
Blocked by: none

## What to build

Train a deterministic, participant-safe LSTM fall detector using persisted duration-normalized CSI data. Use sequence input organized by normalized time and subcarrier, select architecture and threshold using non-test participant-grouped validation, and persist model-agnostic run artifacts and an HTML report.

## Acceptance criteria

- [x] Loads persisted duration-normalized data without regenerating raw recordings.
- [x] Uses fixed-shape normalized-time sequences with documented input semantics.
- [x] Keeps participants and recordings isolated across training, validation, and test splits.
- [x] Supports configurable hidden size, layer count, dropout, training epochs, and random seed.
- [x] Selects architecture and decision threshold using validation data only.
- [x] Produces reproducible model, run artifact, dataset audit, and self-contained HTML report.
- [x] Records input representation, model configuration, fit participants, selection protocol, and score orientation.
- [x] Tests verify tensor shapes, deterministic training, split isolation, persisted-model inference, and artifact compatibility.

## Answer

Implemented `wifall.lstm` and `train-lstm`. LSTM consumes persisted duration-normalized CSI, performs participant-grouped non-test architecture and threshold selection, persists model/run/audit/report artifacts, and exposes configurable hidden size, layer count, dropout, epochs, batch size, learning rate, recall target, and seed.

Validation: `76 passed`; Ruff clean.

Generated report: `data/processed/wifall/lstm-duration-normalized-run/report.html`
