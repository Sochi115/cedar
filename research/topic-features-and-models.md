# Features and Models

## Why this topic matters

Model comparisons are meaningful only when feature construction, labels, splits, and baselines are comparable.

## Current understanding

The initial landscape spans hand-designed temporal and spatial CSI features, phase and amplitude representations, classical classifiers, multistage systems, and deep-learning approaches.

## Comparison

| Approach | Example sources | Initial observation |
|---|---|---|
| One-class SVM | [WiFall](paper-2014-wifall.md) | Useful baseline when normal activity is better represented than falls |
| Random Forest | [WiFall](paper-2014-wifall.md) | Classical supervised baseline |
| Phase and amplitude features | [RT-Fall](paper-2016-rt-fall.md), [Anti-fall](paper-2015-anti-fall.md) | Depends heavily on CSI calibration and hardware |
| Multistage detection | [WmFall](paper-2018-wmfall.md) | Can expose false-alarm trade-offs |
| Deep learning | [2023 deep-learning study](paper-2023-deep-learning-based-fall-detection-using-wifi-csi.md), [DapFall](paper-2025-dapfall.md) | Promising reported metrics; data diversity and split rigor remain concerns |

## Papers

- [WiFall](paper-2014-wifall.md)
- [RT-Fall](paper-2016-rt-fall.md)
- [WmFall](paper-2018-wmfall.md)
- [Anti-fall](paper-2015-anti-fall.md)
- [Deep Learning-Based Fall Detection Using WiFi CSI](paper-2023-deep-learning-based-fall-detection-using-wifi-csi.md)
- [DapFall](paper-2025-dapfall.md)

## Datasets and repositories

- [CSI-Bench](dataset-csi-bench.md) — reported benchmark coverage across users and environments.

## Limitations and disagreements

Headline accuracy, precision, recall, and detection rate values cannot be ranked until evaluation splits and label definitions are verified.

## Open questions

- [OQ-001 — Verify initial quantitative claims](open-questions.md)
- [OQ-005 — Define minimum evaluation protocol](open-questions.md)
