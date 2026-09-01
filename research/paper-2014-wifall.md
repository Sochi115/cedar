# WiFall: Device-Free Fall Detection by Wireless Networks

- Authors: Chunmei Han, Yuxi Wang, Kaishun Wu, Lionel M. Ni
- Year: 2014
- URL: https://ieeexplore.ieee.org/document/6847948/
- Venue: IEEE Transactions on Mobile Computing
- Review status: Initial pass; verify full methodology before relying on quantitative claims.

## Research question

Whether existing wireless infrastructure can support unobtrusive, device-free fall detection without wearables or additional sensing hardware.

## Hardware and CSI setup

The initial review reports commercial IEEE 802.11n network interface cards in laptops. Exact transmitter/receiver arrangement, antenna configuration, CSI extraction tool, and sampling details: Not reported.

## Participants and environment

The initial review reports typical indoor scenarios with varied layouts. Participant count, room dimensions, and trial counts: Not reported.

## Dataset

The study focused primarily on fall detection. Exact fall and non-fall activity labels: Not reported.

## Preprocessing

The approach analyzes CSI changes caused by human movement. Exact filtering, calibration, segmentation, and normalization steps: Not reported.

## Features and model

The initial review identifies CSI time variability and spatial diversity as key features and reports one-class SVM and Random Forest classifiers.

## Evaluation protocol

Exact train/test split, participant separation, environment separation, and class balance: Not reported.

## Results

The initial review reports 87–90% detection precision and a 15–18% false-alarm rate.

## Limitations

The reported false-alarm rate is substantial for a safety-related application. Full-text review is required to understand environmental robustness, participant diversity, and reproducibility.

## Relevance to this project

WiFall is an early foundational CSI fall-detection paper. It establishes a useful baseline for comparing later systems such as RT-Fall and for separating classical feature-based approaches from later deep-learning approaches.

## Related topics

- [CSI vs RSSI](topic-csi-vs-rssi.md)
- [Hardware and CSI Extraction](topic-hardware-and-csi-extraction.md)
- [Preprocessing and Representation](topic-preprocessing-and-representation.md)
- [Features and Models](topic-features-and-models.md)
- [Evaluation and Generalization](topic-evaluation-and-generalization.md)
- [Reproducibility](topic-reproducibility.md)
