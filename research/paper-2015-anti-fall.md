# Anti-Fall: Non-Intrusive Real-Time Fall Detector

- Year: 2015
- URL: https://doi.org/10.1007/978-3-319-19312-0_15
- Review status: Initial pass; verify against full paper before relying on quantitative claims.

## Research question

Whether WiFi CSI can provide non-intrusive real-time fall detection.

## Hardware and CSI setup

The initial report identifies an Intel 5300 WiFi chipset and CSI amplitude and phase data.

## Participants and environment

Two environments with multiple volunteers were reported. Exact participant count and environment details: Not reported.

## Dataset

Falls were distinguished from sitting.

## Preprocessing

Filtering and noise reduction were reported. Exact parameters: Not reported.

## Features and model

The initial report mentions SVM and deep-learning activity classification. Exact method used for reported results: Not reported.

## Evaluation protocol

Performance was compared with baseline systems. Split design: Not reported.

## Results

Reported detection rate: 89%.

## Limitations

Environmental changes can degrade performance.

## Relevance to this project

Useful early reference for Intel 5300 CSI and environmental robustness concerns.

## Related topics

- [CSI vs RSSI](topic-csi-vs-rssi.md)
- [Hardware and CSI Extraction](topic-hardware-and-csi-extraction.md)
- [Preprocessing and Representation](topic-preprocessing-and-representation.md)
- [Features and Models](topic-features-and-models.md)
- [Evaluation and Generalization](topic-evaluation-and-generalization.md)
- [Reproducibility](topic-reproducibility.md)
