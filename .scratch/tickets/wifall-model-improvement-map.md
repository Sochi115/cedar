# WiFall Model Improvement and Model Comparison

Status: open
Label: ready-for-agent

## Destination

Improve WiFall fall detection with a separate SVM training pipeline, train a separate Random Forest model, and compare model artifacts and training implementations using the generic evaluation report.

## Scope

- Keep current baseline SVM code and model outputs unchanged.
- Build improved SVM as separate training code and separate model output.
- Build Random Forest as separate training code and separate model output.
- Use participant-held-out evaluation and the generic evaluation artifact format.
- Compare model family, training code, features, parameters, thresholds, and metrics.

## Tickets

- [Build separate improved WiFall SVM training pipeline](completed/wifall-model-improvement-01-improved-svm.md)
- [Train separate WiFall Random Forest model](completed/wifall-model-improvement-02-random-forest.md)
- [Upgrade temporal CSI feature representation](completed/wifall-model-improvement-05-temporal-features.md)
- [Compare baseline, improved SVM, and Random Forest runs](wifall-svm-report-04-compare-runs.md)
- [Train simple MLP fall detector](wifall-model-improvement-09-mlp.md)
- [Analyze MLP fall-detection performance](completed/wifall-model-improvement-10-mlp-analysis.md)
- [Train simple CNN fall detector](completed/wifall-model-improvement-11-cnn.md)
- [Analyze CNN fall-detection performance](completed/wifall-model-improvement-12-cnn-analysis.md)
- [Train simple LSTM on duration-normalized CSI](completed/wifall-model-improvement-13-lstm-duration-normalized.md)
- [Analyze LSTM fall-detection usefulness](completed/wifall-model-improvement-14-lstm-analysis.md)
- [Tune causal LSTM baseline for real-time alerts](completed/wifall-model-improvement-15-lstm-tuning.md)
- [Evaluate stacked LSTM fall detector](wifall-model-improvement-16-stacked-lstm.md)
- [Evaluate CNN-BiLSTM hybrid fall detector](wifall-model-improvement-17-cnn-bilstm-hybrid.md)
- [Evaluate temporal convolutional network fall detector](wifall-model-improvement-18-temporal-convolutional-network.md)
- [Evaluate small Transformer encoder fall detector](wifall-model-improvement-19-transformer-encoder.md)
- [Build causal consecutive-window WiFall input pipeline](completed/wifall-model-improvement-20-causal-consecutive-windows.md)

## Decisions so far

- [Build separate improved WiFall SVM training pipeline](completed/wifall-model-improvement-01-improved-svm.md): Separate improved SVM training now persists model, audit, generic result, and HTML report artifacts.
- Current baseline remains available for comparison.
- Ticket 01 uses SVM.
- Ticket 02 uses Random Forest.
- Generic validation and static reporting remain model-independent.
- Test participants remain untouched during feature, parameter, and threshold selection.
- [Upgrade temporal CSI feature representation](completed/wifall-model-improvement-05-temporal-features.md): Random Forest now compares baseline four-window features with an eight-window motion representation under grouped non-test validation.
- [Train separate WiFall Random Forest model](completed/wifall-model-improvement-02-random-forest.md): Separate Random Forest training now persists model, audit, generic result, and HTML report artifacts using the improved feature representation and validation-only selection.
- [Analyze Random Forest hard examples](completed/wifall-model-improvement-06-hard-example-analysis.md): Finalized Random Forest, improved SVM, and baseline SVM evaluations now share deterministic, diagnostic-only error analysis by group, activity, duration, signal quality, score range, and observation example.
- [Blend improved SVM and Random Forest scores](completed/wifall-model-improvement-07-svm-random-forest-ensemble.md): Calibrated source scores are blended with weight and threshold selected by equal-participant validation; test participants remain untouched until final ensemble evaluation.
- [Train simple MLP fall detector](completed/wifall-model-improvement-09-mlp.md): Small regularized MLP uses fixed-size normalized temporal CSI features, PCA reduction, grouped OOF sigmoid calibration, and safety-oriented recall-constrained threshold selection. Reevaluation catches all test falls but produces 32 false positives.
- [Analyze MLP fall-detection performance](completed/wifall-model-improvement-10-mlp-analysis.md): MLP research closed as `wontfix`; safety-first operation has 0% specificity and no usable operating point.
- [Train simple CNN fall detector](completed/wifall-model-improvement-11-cnn.md): CNN preserves temporal and subcarrier structure in a fixed-size tensor and uses grouped safety-oriented validation. Initial test recall is 93.75% with 9.375% specificity.
- [Analyze CNN fall-detection performance](completed/wifall-model-improvement-12-cnn-analysis.md): CNN catches some falls missed by comparator models but has no useful validation operating point at 90%+ minimum participant recall; continue investigating rather than adopt.
- [Train simple LSTM on duration-normalized CSI](completed/wifall-model-improvement-13-lstm-duration-normalized.md): LSTM training now consumes persisted duration-normalized sequences, selects architecture and threshold with participant-grouped non-test validation, and writes reproducible model, run, audit, and HTML report artifacts.
- [Analyze LSTM fall-detection usefulness](completed/wifall-model-improvement-14-lstm-analysis.md): LSTM analysis compares compatible duration-normalized model runs, keeps recommendation logic validation-only, and rejects LSTM adoption because validation specificity trails the best comparator by more than 0.10 at 95% minimum participant recall.
- [Tune causal LSTM baseline for real-time alerts](completed/wifall-model-improvement-15-lstm-tuning.md): LSTM now uses causal rolling windows with bounded lookback, validation-only early stopping and scheduling, optimizer candidates, and real-time alert diagnostics. Current run remains unsuitable due to a 100% false-alarm rate.
- [Build causal consecutive-window WiFall input pipeline](completed/wifall-model-improvement-20-causal-consecutive-windows.md): Removed duration-based preprocessing, added deterministic participant-safe raw CSI windows with weak recording labels, persisted metadata, optional fixed elapsed-time windows, and causal LSTM consumption.
- [Evaluate CSI Data Augmentation](completed/wifall-model-improvement-08-csi-augmentation.md): Reusable deterministic raw-CSI augmentation supports noise, temporal jitter, subcarrier dropout, and window perturbation while augmenting training records only; CNN, improved SVM, and Random Forest trainers persist augmentation metadata.
