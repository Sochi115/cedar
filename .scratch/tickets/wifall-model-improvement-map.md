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

## Decisions so far

- [Build separate improved WiFall SVM training pipeline](completed/wifall-model-improvement-01-improved-svm.md): Separate improved SVM training now persists model, audit, generic result, and HTML report artifacts.
- Current baseline remains available for comparison.
- Ticket 01 uses SVM.
- Ticket 02 uses Random Forest.
- Generic validation and static reporting remain model-independent.
- Test participants remain untouched during feature, parameter, and threshold selection.
- [Upgrade temporal CSI feature representation](completed/wifall-model-improvement-05-temporal-features.md): Random Forest now compares baseline four-window features with an eight-window motion representation under grouped non-test validation.
- [Train separate WiFall Random Forest model](completed/wifall-model-improvement-02-random-forest.md): Separate Random Forest training now persists model, audit, generic result, and HTML report artifacts using the improved feature representation and validation-only selection.
