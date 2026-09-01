# Evaluation and Generalization

## Why this topic matters

A fall detector must distinguish falls from normal activity while remaining useful across people, rooms, layouts, and wireless conditions. Evaluation design determines whether reported performance measures that capability.

## Current understanding

The initial papers report strong headline results, but participant separation, environment separation, class balance, false-alarm definitions, and train/test split design are often not available in the initial notes.

## Comparison

| Evaluation concern | Why it matters |
|---|---|
| Participant-separated testing | Tests performance on unseen people |
| Environment-separated testing | Tests transfer across rooms and layouts |
| Fall/non-fall balance | Prevents accuracy from hiding weak fall recall |
| False-alarm rate | Critical for practical safety use |
| Sensitivity and specificity | Separates missed falls from normal-activity alarms |
| Repeated-trial handling | Prevents near-duplicate samples crossing splits |

## Papers

- [WiFall](paper-2014-wifall.md)
- [RT-Fall](paper-2016-rt-fall.md)
- [WmFall](paper-2018-wmfall.md)
- [Anti-fall](paper-2015-anti-fall.md)
- [Deep Learning-Based Fall Detection Using WiFi CSI](paper-2023-deep-learning-based-fall-detection-using-wifi-csi.md)
- [DapFall](paper-2025-dapfall.md)

## Datasets and repositories

- [CSI-Bench](dataset-csi-bench.md) — reported broad user and environment coverage.
- [ElderAL-CSI](dataset-elderal-csi.md) — reported controlled activity collection.

## Limitations and disagreements

Metrics from different studies are not directly rankable. The current landscape lacks verified, common evaluation protocols.

## Open questions

- [OQ-001 — Verify initial quantitative claims](open-questions.md)
- [OQ-003 — Compare generalization evidence](open-questions.md)
- [OQ-005 — Define minimum evaluation protocol](open-questions.md)
