# Evaluate CSI Data Augmentation

Status: resolved
Type: task
Label: ready-for-agent
Blocked by: none

## What to build

Build reusable, participant-safe CSI data augmentation for hard fall patterns and channel variation. Keep raw-CSI transformations independent from model training and leave validation and test recordings untouched.

Model training, metric comparison, and report integration remain downstream responsibilities of each model pipeline.

## Acceptance criteria

- [x] Supports controlled CSI noise, temporal jitter, subcarrier dropout, and safe window perturbation.
- [x] Augmentation applies only to training participants and preserves participant grouping.
- [x] Augmented examples retain labels and recording metadata.
- [x] Each augmentation family can run independently or as a deterministic combined configuration.
- [x] Result metadata records augmentation configuration and generated-example counts.
- [x] Tests verify deterministic augmentation, split isolation, source immutability, label preservation, and configuration compatibility.

## Answer

Implemented reusable raw-CSI augmentation in `wifall/augmentation.py`. It supports deterministic CSI noise, temporal jitter, subcarrier dropout, and temporal-window perturbation. `augment_split_records` augments training records only and returns JSON-compatible configuration and generated-example metadata. Source recordings, validation records, and test records remain unchanged. CNN, improved SVM, and Random Forest trainers now accept augmentation configuration and persist augmentation metadata in run artifacts.
