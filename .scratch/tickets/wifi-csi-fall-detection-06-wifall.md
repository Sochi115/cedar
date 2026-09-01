# Research WiFall Fall-Detection Study

Status: resolved
Type: research
Label: wayfinder:research
Blocked by: none

## Question

What is WiFall, and what does its original research establish about WiFi-based fall detection?

Locate the primary paper or authoritative source. Record citation, URL, publication year, signal and hardware setup, CSI or RSSI usage, participants and environments, activities and labels, preprocessing, features and model, evaluation protocol, results, limitations, and reproducibility. Explain how WiFall relates to RT-Fall and to the current CSI-focused evidence landscape. Mark any details that require full-text verification.

## Answer

WiFall is Han, Wang, Wu, and Ni's 2014 IEEE Transactions on Mobile Computing paper, [WiFall: Device-Free Fall Detection by Wireless Networks](https://ieeexplore.ieee.org/document/6847948/). The initial review identifies CSI from commercial IEEE 802.11n laptop NICs, CSI time variability and spatial diversity features, and one-class SVM and Random Forest classifiers. It reports 87–90% precision and 15–18% false-alarm rate. Participant counts, exact layouts, signal-processing parameters, split design, and reproducibility details remain unverified.

WiFall is an early CSI baseline. It is distinct from RT-Fall, which emphasizes phase differences and real-time detection. WiFall belongs in the CSI evidence landscape and does not change the current scope.

Asset: [WiFall paper note](../../research/paper-2014-wifall.md)
