# ENetFall zero-shot adapter

Standalone adapter for feeding ENetFall event clips through the existing WiFall model pipelines. Code lives outside `wifall/`; it does not train models or change saved thresholds.

## Validate local data

```bash
uv run python -m experiments.enetfall_zero_shot.adapter data/raw/ENetFall
```

Reads five MAT files, uses `dataset_CSI_t` once per event, checks expected labels/room counts, and prints source and adapted shapes. Existing files remain untouched.

## Use a saved WiFall pipeline

```python
import joblib
from experiments.enetfall_zero_shot.adapter import iter_enetfall_events

model = joblib.load("data/processed/wifall/improved-svm-run/model.joblib")
for event in iter_enetfall_events("data/raw/ENetFall"):
    receiver_recordings = event.to_wifall_recordings(
        allow_approximate_subcarrier_mapping=True,
    )
    scores_by_receiver = model.decision_function_recordings(list(receiver_recordings))
    # Keep event.event_id and event.room; combine/report three receiver scores
    # as one event, not as three independent samples.
```

All current WiFall model pipelines can consume each adapted `Recording`. Evaluate receiver streams separately first; any score fusion must be specified before examining ENetFall labels. Keep each model's saved WiFall threshold fixed for zero-shot evaluation.

## Transformations and limits

- Source MAT layout `(events, 625, 90)` is deinterleaved to `(625, 3 receivers, 30 subcarriers)`, matching authors’ training code.
- Numeric label `1` maps to `fall`; `0` maps to `non-fall`. Adapter checks published totals: 321 falls and 436 non-falls.
- Source stream contains amplitude, not complex CSI. Negative interpolation artifacts are clipped at zero; phase cannot be reconstructed. Amplitude is represented as real-only complex CSI because WiFall extractors apply `abs()`.
- Each receiver is adapted separately. Never flatten or average 90 columns into one stream.
- Each stream is linearly interpolated from 30 to 52 **by subcarrier order**, then polyphase-resampled from 125 Hz to 84 Hz. Result per receiver: `(420, 52)`.
- 84 Hz approximates WiFall’s observed sample rate. ENetFall source clips are 5 seconds; WiFall recordings are typically about 3 seconds, so recording-level duration shift remains.
- **Subcarrier interpolation is approximate, not calibrated physical-frequency alignment.** Intel 5300 and ESP32-S3 grids/bands differ. Adapter requires explicit `allow_approximate_subcarrier_mapping=True` so zero-shot scores cannot silently imply exact frequency correspondence. Use scores as exploratory only until physical bins are verified.
- Participant IDs are unavailable in a consistent form. The synthetic `Recording` uses `participant_id="unknown"`; group conclusions must use room/event metadata, not that field.
- Timestamps generated for compatibility are synthetic, monotonic timestamps at 84 Hz—not original capture timestamps.

## Checks

```bash
uv run pytest experiments/enetfall_zero_shot/test_adapter.py
uv run ruff check experiments/enetfall_zero_shot
```
