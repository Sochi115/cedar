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
