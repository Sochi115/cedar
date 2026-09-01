# WiFi CSI Fall-Detection Research Landscape

Initial evidence map. This document synthesizes the paper and resource notes; it is not a claim that the literature search is complete.

## Main patterns

- Research commonly uses commodity WiFi devices and CSI amplitude, phase, or both.
- Early work such as WiFall established CSI time-variability and spatial-diversity features before later real-time and deep-learning systems.
- Studies usually distinguish falls from daily activities such as walking, sitting, or lying.
- Controlled indoor environments dominate the initial sample.
- Reported accuracy and sensitivity are difficult to compare because datasets, labels, splits, and metrics differ.
- Environmental change, participant diversity, and false alarms recur as limitations.
- Reproducibility is frequently limited by missing hardware, preprocessing, or dataset details.
- Cross-environment and data-efficient detection are active concerns.

## Paper comparison

| Paper | Year | Signal / method | Reported setting | Reported result | Main limitation |
|---|---:|---|---|---|---|
| [WiFall](paper-2014-wifall.md) | 2014 | CSI time variability and spatial diversity; one-class SVM and Random Forest | Typical indoor scenarios | 87–90% precision; 15–18% false-alarm rate reported | Full setup and generalization details require verification |
| [Deep Learning-Based Fall Detection Using WiFi CSI](paper-2023-deep-learning-based-fall-detection-using-wifi-csi.md) | 2023 | CSI; deep learning | 4 environments, 22 volunteers | 96–99% accuracy reported | Limited diversity and environmental sensitivity |
| [RT-Fall](paper-2016-rt-fall.md) | 2016 | CSI phase differences; real-time system | 4 indoor scenarios | Sensitivity reported higher than WiFall | Wireless-configuration sensitivity |
| [WmFall](paper-2018-wmfall.md) | 2018 | CSI; multistage detection | Laboratory | 89% accuracy; 8% false alarms reported | Complex-environment false alarms |
| [Anti-fall](paper-2015-anti-fall.md) | 2015 | Intel 5300 CSI amplitude and phase | 2 environments | 89% detection rate reported | Environmental changes |
| [DapFall](paper-2025-dapfall.md) | 2025 | CSI; dynamic amplitude probability density and deep learning | Cross-environment focus | 94.3% recall reported | Environmental sensitivity remains |

## Public resources

- [ElderAL-CSI](dataset-elderal-csi.md) includes falls and daily activities in a controlled indoor setup.
- [CSI-Bench](dataset-csi-bench.md) reports broader device, user, and environment coverage, but its access and licensing details need verification.
- [WiPE-FaLl](dataset-wipe-fall.md) concerns fall-risk analysis and is outside the initial fall-detection scope.
- [FallDeWideo](dataset-falldewideo.md) combines WiFi CSI with video and is outside the current single-modality scope.
- [CSI-BFI-HAR](dataset-csi-bfi-har.md) is a broad HAR resource; its direct fall-detection usefulness needs verification.
- No qualifying GitHub repository was identified in this initial pass.

## Comparison cautions

Do not compare headline accuracy values without checking participant separation, environment separation, activity balance, label definitions, false-alarm treatment, and train/test split design. Those details are incomplete for several initial notes.

## Related open questions

See [open-questions.md](open-questions.md), especially questions about source verification, public GitHub tooling, generalization, and evaluation protocol.
