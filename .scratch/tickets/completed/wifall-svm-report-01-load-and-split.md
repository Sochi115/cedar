# Make WiFall recordings reproducibly loadable

Status: resolved
Type: task
Label: ready-for-agent
Blocked by: none

## What to build

Create a reproducible WiFall data pipeline that discovers recordings, parses CSI samples, creates recording-level examples, preserves participant and activity metadata, and produces deterministic participant-held-out splits for fall detection.

## Acceptance criteria

- [x] Pipeline discovers 10 participants, 928 recordings, and 232,941 non-empty CSI rows from downloaded WiFall data.
- [x] Each CSI sample is parsed from 104 integer values into 52 complex subcarriers.
- [x] Pipeline produces binary `fall` and `non-fall` targets while preserving original activity labels.
- [x] Recording-level examples never combine rows from different recordings.
- [x] Complete participants and recordings remain in one split; no participant leakage occurs.
- [x] Split configuration and resulting class counts are persisted for downstream training and reporting.
- [x] Re-running with the same configuration produces identical examples and splits.
- [x] Tests cover CSI parsing, activity labels, blank rows, malformed rows, and deterministic splitting.

## Answer

Implemented WiFall loading and split pipeline in the `wifall` package. It discovers and validates recordings, parses CSI into 52 complex subcarriers, preserves source activity labels, derives binary fall-detection targets, loads recording-level examples, and writes deterministic participant-held-out JSON manifests.

Added CLI commands for dataset summaries and split generation. Verified downloaded WiFall counts: 10 participants, 928 recordings, and 232,941 non-empty rows. Added 12 self-contained tests covering parsing, malformed and blank rows, labels, deterministic splits, leakage prevention, and manifest round trips.

Validation: full pytest suite, Ruff, and `uv lock --check` pass.
