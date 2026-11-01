# ENetFall RBF SVM

Train an event-level binary RBF SVM on local ENetFall MAT files. This uses native 625-sample, 3-receiver, 30-subcarrier amplitudes; it does not interpolate/resample into WiFall format.

```bash
uv run python -m enetfall.train_svm \
  data/raw/ENetFall \
  data/processed/enetfall/rbf-svm-run
```

Training only runs when this command is explicitly invoked. It writes `model.joblib` and generic evaluation artifact `run.json`. Test metrics are the primary metrics; validation metrics are also recorded. No report rendering or model fitting happens during setup/checks.

## Decisions

- **Task:** classify each five-second event as `fall` or `non-fall`, from `dataset_labels` (`1` and `0` respectively). Receiver streams remain separate features within one event; they are not counted as three independent examples.
- **Feature vector:** for every receiver/subcarrier pair, compute temporal mean, standard deviation, minimum, maximum, and standard deviation of adjacent-sample differences. Concatenate statistic-first: 5 × 3 × 30 = **450 features/event**. No 30-to-52 interpolation, resampling, or per-event scaling.
- **Estimator:** `StandardScaler` followed by `SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")`. Scaler fits training events only. Parameters fixed; no search or validation-driven threshold selection.
- **Split:** fixed by nominal environment, never by event. Train: `living_room_A` and `meeting_room_B` (149 falls, 233 non-falls; 382 events). Validation: both home-lab MAT files (`home_lab_left` and `home_lab_right`, 76 falls, 94 non-falls; 170 events). Test: `lecture_room_D` (96 falls, 109 non-falls; 205 events). This uses all 757 published events (321 falls, 436 non-falls), with test room untouched until final evaluation. Home-lab L/R stay together because both identify one nominal environment.
- **Interpretation limit:** ENetFall release lacks reliable participant IDs. Environment-held-out split prevents event and nominal-environment leakage, but cannot guarantee participant independence; same volunteers may occur across splits. Report scores as environment transfer, not person-independent generalization. Only one held-out test environment limits uncertainty estimates.
- **Data checks:** script rejects unexpected published per-room label counts and invalid event feature shapes/non-finite values.

## Train EfficientNet-B0 on native CSI clips

Train a three-channel EfficientNet-B0 directly on ENetFall's 625 × 3 × 30 event clips. The fixed room-held-out split trains on living room and meeting room, selects the score threshold on home-lab rooms, then evaluates once on lecture room:

```bash
uv run python -m enetfall.train_efficientnet \
  data/raw/ENetFall \
  data/processed/enetfall/efficientnet-run
```

Writes `model.pt`, `run.json`, and `report.md`. Report includes validation loss by epoch and validation/test confusion matrices. Only best-validation-loss checkpoint `model.pt` is retained. CSI uses per-event, per-receiver/subcarrier median/MAD normalization. EfficientNet starts from random weights; no ImageNet weights used. Participant independence is not guaranteed because ENetFall lacks reliable participant IDs.

Optional training controls: `--epochs`, `--batch-size`, `--learning-rate`, `--weight-decay`, and `--patience`.

## Run downloaded ENetFall training script

Run bounded replica of `data/raw/ENetFall/train.py`, preserving its pretrained EfficientNet-B0, classifier head, ImageNet normalization, Adam settings, and random 80/20 split:

```bash
uv run python -m enetfall.train_downloaded \
  data/raw/ENetFall \
  data/processed/enetfall/downloaded-train-run
```

Writes one `model.pt`, `run.json`, and standalone `report.html`. Replica fixes original script's potentially unbounded restart loop, steps its StepLR once per epoch, and evaluates all holdout events. Original holdout still mixes rooms and is monitored during training; scores are not room-independent estimates.

## Train compact CSI CNN

Train a smaller CNN with per-event z-score normalization and a stratified event-level random split. Outer split holds out 20% for test; 20% of remaining development data serves as validation. Batch sizes 8, 16, 32, and 60 are compared; lowest validation loss selects batch size and checkpoint. Test set is evaluated only after selection.

```bash
uv run python -m enetfall.train_cnn \
  data/raw/ENetFall \
  data/processed/enetfall/random-cnn-batch-comparison
```

Uses learning rate `1e-4`, weight decay `1e-3`, and validation-loss early stopping. Writes one selected `model.pt`, `run.json`, and standalone `report.html`. Report includes batch comparison, train/validation loss, confusion matrices, score distributions, and room counts per random split. Random split shares rooms across subsets, so this measures event-level interpolation, not unseen-room transfer. The prior room-held-out EfficientNet run remains available separately.

## Zero-shot compact CNN evaluation on WiFall

Evaluate the random-split compact CNN on WiFall's participant-held-out test recordings without training, fine-tuning, or changing ENetFall validation threshold:

```bash
uv run python -m enetfall.evaluate_cnn_wifall_zero_shot \
  data/processed/enetfall/random-cnn-batch-comparison \
  data/raw/wifall/WiFall \
  data/processed/wifall/split.json \
  data/processed/enetfall/wifall-zero-shot-cnn
```

Writes `run.json` and standalone `report.html`. Adapter linearly maps 52 WiFall subcarriers to 30, resamples approximately 84 Hz to 125 Hz, and duplicates the single WiFall stream into three pseudo-receiver channels. This only approximates model input shape; it does not remove hardware or domain shift.

Evaluate every recording, including train/validation/test participants, for descriptive all-data results:

```bash
uv run python -m enetfall.evaluate_cnn_wifall_zero_shot --all-recordings
```

All-data scores are pooled descriptive results, not held-out generalization metrics.

Evaluate the ImageNet-pretrained EfficientNet replica from downloaded `train.py` on all WiFall recordings:

```bash
uv run python -m enetfall.evaluate_downloaded_wifall_zero_shot
```

Writes `data/processed/enetfall/downloaded-train-wifall-all/report.html` and `run.json`. Uses saved ENetFall preprocessing statistics and original 0.5 argmax decision; no WiFall tuning.

## Compare compact temporal features and classifiers

Run the follow-up experiment explicitly:

```bash
uv run python -m enetfall.compare_models \
  data/raw/ENetFall \
  data/processed/enetfall/model-comparison
```

Writes `experiment.json`; does not overwrite the baseline SVM run. It compares the existing global-feature RBF baseline with raw and per-event median/MAD-normalized 4-window features (216 values each). Window features average 30 subcarriers into six contiguous bands, then measure mean, standard deviation, and adjacent-difference standard deviation per receiver/band/window. Candidate models: linear SVM, logistic regression, and RBF SVM, each at `C=0.1, 1, 10`.

Each scaler fits training rooms only. Thresholds maximize balanced accuracy on validation rooms and transfer unchanged to test. Test results are descriptive, not for candidate selection; there is only one test environment, and participant independence is unknown. The original test room has already been evaluated by the baseline, so treat this comparison as exploratory rather than a fresh blinded test.

## Train the random forest

Train a random forest on the same environment-held-out split and robust 4-window features. The forest fits training rooms only; fall-probability threshold is selected on validation rooms and then fixed for test evaluation.

```bash
uv run python -m enetfall.train_random_forest \
  data/raw/ENetFall \
  data/processed/enetfall/random-forest-run

uv run python -m evaluation.cli \
  data/processed/enetfall/random-forest-run/run.json \
  data/processed/enetfall/random-forest-run/report.html
```

Outputs: `model.joblib`, generic `run.json`, and self-contained `report.html`. Forest configuration: 400 trees, `max_features="sqrt"`, `min_samples_leaf=2`, balanced-subset class weights, seed 42. No hyperparameter search; validation only selects the operating threshold.

## Zero-shot ENetFall forest evaluation on WiFall

This experiment applies the saved ENetFall forest to WiFall test participants without fitting or changing its ENetFall-selected threshold:

```bash
uv run python -m enetfall.evaluate_wifall_zero_shot \
  data/processed/enetfall/random-forest-run/model.joblib \
  data/raw/wifall/WiFall \
  data/processed/wifall/split.json \
  data/processed/enetfall/wifall-zero-shot-run

uv run python -m evaluation.cli \
  data/processed/enetfall/wifall-zero-shot-run/run.json \
  data/processed/enetfall/wifall-zero-shot-run/report.html
```

This is deliberately labeled **approximate**: WiFall's single 52-subcarrier stream is interpolated by index to 30 bins, temporally resampled from approximately 84 to 125 Hz, then copied into three identical pseudo-receiver channels. This matches model input shape, not physical receiver/frequency alignment. WiFall's shorter clip duration and different hardware remain domain shifts; scores are exploratory only.

Default script scope is WiFall test participants only. To evaluate every recording (train, validation, and test participants) without fitting or threshold changes:

```bash
uv run python -m enetfall.evaluate_wifall_zero_shot \
  data/processed/enetfall/random-forest-run/model.joblib \
  data/raw/wifall/WiFall \
  data/processed/wifall/split.json \
  --all-recordings

uv run python -m evaluation.cli \
  data/processed/enetfall/wifall-zero-shot-all-run/run.json \
  data/processed/enetfall/wifall-zero-shot-all-run/report.html
```

All-data evaluation pools every WiFall participant and includes source manifest train/validation/test partitions; it is descriptive, not a held-out generalization estimate. The script uses the saved manifest, makes no fitting or threshold changes, and keeps data loading, feature extraction, and evaluation inside `enetfall/`. HTML rendering uses generic `evaluation.cli`.

## Evaluate the Random Forest on CSI-Bench

Run the saved ENetFall Random Forest against every sample in CSI-Bench's `FallDetection/metadata/sample_metadata.csv` (not only the official test split):

```bash
uv run python -m enetfall.evaluate_csi_bench_zero_shot
```

Writes `data/processed/enetfall/csi-bench-zero-shot-all-run/run.json` and self-contained `report.html`. Model and validation-selected threshold stay fixed. CSI-Bench samples have one stream and variable subcarrier counts; the adapter interpolates bins to 30 and duplicates the stream into three pseudo-receivers. This shape adapter is not physical alignment. Since this all-data evaluation includes the official train, validation, and test partitions, treat it as descriptive domain-transfer evidence, not a held-out benchmark.
