# CSI vs RSSI

## Why this topic matters

CSI and RSSI expose different levels of information about changes in wireless propagation. Keeping them distinct prevents claims about CSI from being mixed with aggregate signal-strength results.

## Current understanding

The initial paper set is CSI-focused. WiFall, RT-Fall, Anti-fall, WmFall, the 2023 deep-learning study, and DapFall all use or report CSI. No RSSI-specific fall-detection source has yet been documented.

CSI may expose temporal, spatial, amplitude, phase, and subcarrier-level changes. RSSI provides a lower-dimensional comparison signal. Direct performance comparison requires matched environments, participants, activities, and evaluation splits.

## Comparison

| Signal | Information level | Current evidence in this KB | POC role |
|---|---|---|---|
| CSI | Subcarrier-level channel response, including amplitude and phase | Primary evidence base | Primary candidate |
| RSSI | Aggregate received signal strength | No dedicated source note yet | Comparison baseline |

## Papers

- [WiFall](paper-2014-wifall.md) — early CSI features based on time variability and spatial diversity.
- [RT-Fall](paper-2016-rt-fall.md) — CSI phase-difference approach.
- [Anti-fall](paper-2015-anti-fall.md) — CSI amplitude and phase.

## Datasets and repositories

No RSSI-specific dataset or repository has been documented yet.

## Limitations and disagreements

The current evidence does not support a fair CSI-versus-RSSI performance claim. Signal definitions, hardware, and collection protocols vary across studies.

## Open questions

- [OQ-003 — Compare generalization evidence](open-questions.md)
- [OQ-005 — Define minimum evaluation protocol](open-questions.md)
