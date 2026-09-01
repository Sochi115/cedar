# Hardware and CSI Extraction

## Why this topic matters

Hardware and extraction choices determine which CSI measurements are available, how reproducible collection is, and whether a research result can be recreated.

## Current understanding

The initial evidence includes commercial 802.11n laptop NICs, Intel 5300 hardware, commodity WiFi devices, routers, USRP equipment, and mixed commercial devices. These setups are not interchangeable.

Important details include transmitter and receiver roles, chipset, antenna count, channel width, subcarriers, sampling rate, firmware, CSI extraction software, packet rate, and synchronization.

## Comparison

| Resource | Reported hardware | Key missing details |
|---|---|---|
| [WiFall](paper-2014-wifall.md) | Commercial 802.11n laptop NICs | Arrangement, antennas, extraction, sampling |
| [Anti-fall](paper-2015-anti-fall.md) | Intel 5300 chipset | Collection and extraction parameters |
| [ElderAL-CSI](dataset-elderal-csi.md) | ZTE AX3000 router; 2x3; 512 subcarriers reported | Access and exact format |
| [WiPE-FaLl](dataset-wipe-fall.md) | USRP X300 | Data access details |

## Papers

- [WiFall](paper-2014-wifall.md)
- [RT-Fall](paper-2016-rt-fall.md)
- [Anti-fall](paper-2015-anti-fall.md)

## Datasets and repositories

- [ElderAL-CSI](dataset-elderal-csi.md)
- [CSI-Bench](dataset-csi-bench.md)
- [CSI-BFI-HAR](dataset-csi-bfi-har.md)

## Limitations and disagreements

Hardware details are incomplete in several notes. A high reported result on one chipset or channel configuration should not be assumed to transfer to another.

## Open questions

- [OQ-004 — Identify suitable public fall datasets](open-questions.md)
- [OQ-002 — Find usable GitHub repositories](open-questions.md)
