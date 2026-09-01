# Datasets and Activities

## Why this topic matters

Fall detection depends on representative fall and non-fall activities, reliable labels, participant diversity, and environments that expose false alarms.

## Current understanding

The initial resources vary from small controlled collections to broader human-activity benchmarks. A dataset can be useful for activity baselines without being suitable for fall-event detection.

## Comparison

| Resource | Activities / scope | Initial suitability |
|---|---|---|
| [ElderAL-CSI](dataset-elderal-csi.md) | Falls, standing, walking, sleeping, drinking, sitting | Potential controlled fall-detection candidate; only 3 participants reported |
| [CSI-Bench](dataset-csi-bench.md) | Fall detection and other tasks | Potential generalization candidate; labels and access need verification |
| [CSI-BFI-HAR](dataset-csi-bfi-har.md) | 20 human activities | Potential non-fall baseline; direct fall coverage needs verification |
| [WiPE-FaLl](dataset-wipe-fall.md) | Timed Up-and-Go and fall-risk analysis | Outside current fall-event scope |
| [FallDeWideo](dataset-falldewideo.md) | Falls and daily activities with video | Outside current single-modality scope |

## Papers

- [Deep Learning-Based Fall Detection Using WiFi CSI](paper-2023-deep-learning-based-fall-detection-using-wifi-csi.md)
- [WmFall](paper-2018-wmfall.md)

## Datasets and repositories

- [ElderAL-CSI](dataset-elderal-csi.md)
- [CSI-Bench](dataset-csi-bench.md)
- [CSI-BFI-HAR](dataset-csi-bfi-har.md)
- [WiPE-FaLl](dataset-wipe-fall.md)
- [FallDeWideo](dataset-falldewideo.md)

## Limitations and disagreements

Small participant pools, controlled environments, unclear access conditions, and inconsistent labels limit cross-dataset comparison.

## Open questions

- [OQ-004 — Identify suitable public fall datasets](open-questions.md)
- [OQ-003 — Compare generalization evidence](open-questions.md)
