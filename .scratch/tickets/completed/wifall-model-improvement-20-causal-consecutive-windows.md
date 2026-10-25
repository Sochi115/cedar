# Build causal consecutive-window WiFall input pipeline

Status: completed
Type: task
Label: ready-for-agent
Blocked by: completed/wifall-model-improvement-15-lstm-tuning.md

## What to build

Create a real-time-compatible input representation from raw timestamped WiFall CSI. Do not use full-recording duration normalization for this pipeline. Participant splitting remains the leakage barrier; consecutive windows are generated only after each recording has inherited its participant split.

Generate fixed-length chronological windows from each raw recording. Prefer a fixed elapsed-time window when timestamps support it; otherwise support a documented fixed-row fallback. Windows may overlap by a configurable stride, but may not cross recording or participant boundaries.

The causal preprocessing path must use only samples at or before each window's end time. It must not use final recording duration, future interpolation, future normalization statistics, or data from another recording.

## Label limitation

Raw WiFall labels recordings by activity, not exact fall onset or recovery. Windows inherit their parent recording label and must be marked as weak recording-level labels. Reports must not claim event-level alert latency until event timestamps are available.

## Safety objective

Create the participant-held-out split before generating windows. Keep all windows from one participant in one split. Fit any train-global preprocessing only on training participants. Validation and test participants remain untouched until evaluation.

## Acceptance criteria

- [ ] Participant, recording, and window split relationships are explicit and validated.
- [ ] Windows are generated after split assignment and never cross recording boundaries.
- [ ] Window duration or row count, stride, timestamp policy, and padding/warmup behavior are configurable and persisted.
- [ ] Causal preprocessing uses only current and past samples and avoids full-recording duration normalization.
- [ ] Window metadata records source recording ID, participant, activity/target, start time, end time, and split.
- [ ] Inherited recording labels are documented as weak labels; event-level alert claims are excluded.
- [ ] Output can feed the causal LSTM without regenerating future-dependent features at inference.
- [ ] Produces deterministic persisted data and metadata suitable for participant-held-out training.
- [ ] Tests cover participant leakage, recording-boundary isolation, chronological windows, deterministic output, causal future mutation, and artifact compatibility.
