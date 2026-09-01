# Reproducibility

## Why this topic matters

A readable research result is not enough for a POC. Others need enough information to obtain the data, recreate preprocessing, run the method, and interpret the evaluation.

## Current understanding

The initial notes frequently lack exact hardware configurations, CSI extraction details, participant splits, preprocessing parameters, labels, licensing, or source code. No qualifying GitHub repository was identified in the initial inventory.

## Comparison

| Reproducibility item | Current status |
|---|---|
| Paper URLs | Available for initial paper notes |
| Hardware details | Partial |
| CSI format and extraction | Partial |
| Dataset access and licensing | Mostly unverified |
| Preprocessing parameters | Often missing |
| Evaluation splits | Often missing |
| GitHub implementations | None identified in initial pass |

## Papers

All paper notes in [research README](README.md) are relevant to reproducibility because each records missing details explicitly.

## Datasets and repositories

- [ElderAL-CSI](dataset-elderal-csi.md)
- [CSI-Bench](dataset-csi-bench.md)
- [CSI-BFI-HAR](dataset-csi-bfi-har.md)

## Limitations and disagreements

A source can be public without being reproducible. Access, license, data format, tooling, and experiment protocol must be checked separately.

## Open questions

- [OQ-001 — Verify initial quantitative claims](open-questions.md)
- [OQ-002 — Find usable GitHub repositories](open-questions.md)
- [OQ-004 — Identify suitable public fall datasets](open-questions.md)
