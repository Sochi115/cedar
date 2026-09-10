# Train simple LSTM on duration-normalized CSI

Status: open
Type: task
Label: ready-for-agent
Blocked by: none

## What to build

Train a deterministic, participant-safe LSTM fall detector using persisted duration-normalized CSI data. Use sequence input organized by normalized time and subcarrier, select architecture and threshold using non-test participant-grouped validation, and persist model-agnostic run artifacts and an HTML report.

## Acceptance criteria

- [ ] Loads persisted duration-normalized data without regenerating raw recordings.
- [ ] Uses fixed-shape normalized-time sequences with documented input semantics.
- [ ] Keeps participants and recordings isolated across training, validation, and test splits.
- [ ] Supports configurable hidden size, layer count, dropout, training epochs, and random seed.
- [ ] Selects architecture and decision threshold using validation data only.
- [ ] Produces reproducible model, run artifact, dataset audit, and self-contained HTML report.
- [ ] Records input representation, model configuration, fit participants, selection protocol, and score orientation.
- [ ] Tests verify tensor shapes, deterministic training, split isolation, persisted-model inference, and artifact compatibility.
