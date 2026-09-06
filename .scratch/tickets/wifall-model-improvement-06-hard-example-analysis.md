# Analyze Random Forest hard examples

Status: open
Type: task
Label: ready-for-agent
Blocked by: completed/wifall-model-improvement-03-grouped-validation.md

## What to build

Explain Random Forest errors by participant, source activity, recording duration, signal quality, score, and temporal behavior. Produce an actionable error summary that guides feature and augmentation work without using test labels for model selection.

## Acceptance criteria

- [ ] Reports false negatives and false positives by participant and activity.
- [ ] Reports error patterns by duration and available signal-quality metadata.
- [ ] Identifies missed-fall score ranges and recording-level examples.
- [ ] Separates diagnostic test reporting from model or threshold selection.
- [ ] Error summary is reproducible and linked to the generic run artifact.
- [ ] Tests cover grouping, missing optional metadata, deterministic output, and report rendering.
