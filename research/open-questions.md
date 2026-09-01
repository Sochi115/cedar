# Open Questions

## OQ-001 — Verify initial quantitative claims

- Area: evidence quality
- Topic: paper results
- Description: Verify reported accuracy, sensitivity, recall, false-alarm, and dataset claims against full primary papers.
- Relevance: POC choices must not depend on copied or non-comparable headline metrics.

## OQ-002 — Find usable GitHub repositories

- Area: reproducibility
- Topic: repositories
- Description: Identify maintained repositories containing CSI parsing, preprocessing, baselines, or fall-detection implementations.
- Relevance: Reusable tooling could reduce POC setup risk.

## OQ-003 — Compare generalization evidence

- Area: evaluation
- Topic: cross-environment and cross-participant performance
- Description: Determine which studies separate participants and environments between training and testing.
- Relevance: Controlled same-environment accuracy may not predict real-world performance.

## OQ-004 — Identify suitable public fall datasets

- Area: dataset
- Topic: dataset selection
- Description: Verify access, licensing, labels, CSI format, and fall coverage for candidate datasets.
- Relevance: Dataset availability constrains reproducible POC design.

## OQ-005 — Define minimum evaluation protocol

- Area: evaluation
- Topic: fall-detection metrics
- Description: Decide required splits and metrics, including sensitivity, specificity, precision, recall, F1, and false alarms.
- Relevance: Consistent evaluation is needed for meaningful paper and POC comparisons.
