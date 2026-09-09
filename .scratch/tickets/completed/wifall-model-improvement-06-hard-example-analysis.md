# Analyze Random Forest hard examples

Status: resolved
Type: task
Label: ready-for-agent
Blocked by: completed/wifall-model-improvement-03-grouped-validation.md

## What to build

Explain Random Forest errors by participant, source activity, recording duration, signal quality, score, and temporal behavior. Produce an actionable error summary that guides feature and augmentation work without using test labels for model selection.

## Acceptance criteria

- [x] Reports false negatives and false positives by participant and activity.
- [x] Reports error patterns by duration and available signal-quality metadata.
- [x] Identifies missed-fall score ranges and recording-level examples.
- [x] Separates diagnostic test reporting from model or threshold selection.
- [x] Error summary is reproducible and linked to the generic run artifact.
- [x] Tests cover grouping, missing optional metadata, deterministic output, and report rendering.

## Answer

Implemented deterministic, model-independent hard-example diagnostics in `evaluation/diagnostics.py`. Random Forest, improved SVM, and baseline SVM now adapt aligned observation metadata into the shared interface. Finalized evaluation artifacts include group/activity error summaries, duration buckets, optional signal-quality fields, missed-fall score ranges, and observation-level examples. Diagnostics record source run and observation IDs and are explicitly excluded from model, feature, calibration, and threshold selection. Generic HTML reports render the diagnostic section. Added tests for model-independent observations, grouping, missing metadata, deterministic output, score alignment, and rendering. Verification: `uv run pytest` (39 passed); `uv run ruff check evaluation wifall tests pyproject.toml` passed.
