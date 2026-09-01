# Preprocessing and Representation

## Why this topic matters

CSI is noisy and high-dimensional. Filtering, segmentation, normalization, and representation choices can materially affect performance and transferability.

## Current understanding

The initial papers report filtering, noise reduction, downsampling, reshaping, segmentation, phase processing, and dynamic amplitude probability density representations. Exact parameters are frequently missing.

## Comparison

| Approach | Sources | What remains unclear |
|---|---|---|
| Filtering and noise reduction | [Anti-fall](paper-2015-anti-fall.md), [RT-Fall](paper-2016-rt-fall.md) | Filter design and parameter selection |
| Segmentation | [WmFall](paper-2018-wmfall.md) | Window size and event-boundary rules |
| Downsampling and reshaping | [2023 deep-learning study](paper-2023-deep-learning-based-fall-detection-using-wifi-csi.md) | Exact transformation and information loss |
| Dynamic amplitude probability density | [DapFall](paper-2025-dapfall.md) | Reproducible implementation details |
| Time variability and spatial diversity | [WiFall](paper-2014-wifall.md) | Feature construction and robustness |

## Papers

- [WiFall](paper-2014-wifall.md)
- [RT-Fall](paper-2016-rt-fall.md)
- [WmFall](paper-2018-wmfall.md)
- [Anti-fall](paper-2015-anti-fall.md)
- [Deep Learning-Based Fall Detection Using WiFi CSI](paper-2023-deep-learning-based-fall-detection-using-wifi-csi.md)
- [DapFall](paper-2025-dapfall.md)

## Datasets and repositories

Dataset-specific preprocessing is not sufficiently documented for the current candidates.

## Limitations and disagreements

Reported method names do not guarantee equivalent implementations. Preprocessing may leak activity, participant, or environment information if performed before splitting.

## Open questions

- [OQ-001 — Verify initial quantitative claims](open-questions.md)
- [OQ-003 — Compare generalization evidence](open-questions.md)
