# Analyze LSTM fall-detection usefulness

Status: resolved
Type: research
Label: ready-for-agent
Blocked by: completed/wifall-model-improvement-13-lstm-duration-normalized.md

## What to build

Analyze the finalized LSTM run against duration-normalized CNN, improved SVM, and Random Forest runs. Determine whether LSTM improves aggregate or per-participant fall detection and whether it contributes complementary detections that justify further investigation.

## Acceptance criteria

- [x] Reports aggregate and per-participant accuracy, balanced accuracy, precision, recall, specificity, false-positive rate, and F1.
- [x] Compares LSTM metrics against compatible duration-normalized model runs.
- [x] Identifies false negatives, false positives, and participant-specific failure patterns.
- [x] Measures overlap and complementarity of LSTM detections and errors against comparator models.
- [x] Keeps test results out of architecture, threshold, and model selection decisions.
- [x] Produces a self-contained analysis artifact and HTML report.
- [x] Records limitations, uncertainty from the held-out participant count, and a clear adopt, investigate, or reject recommendation.

## Answer

Implemented `wifall.lstm_analysis` and `analyze-lstm`. Analysis compares LSTM with duration-normalized CNN, improved SVM, and Random Forest runs, reports aggregate and participant metrics, enumerates errors, measures complementarity, and selects recommendation basis from validation operating points only.

Result: `reject` LSTM for adoption. At 95% minimum participant recall, validation specificity trails best comparator by more than 0.10. Test results remain diagnostic only. LSTM catches all test falls but specificity is 15.625% with 84.375% false-positive rate.

Generated report: `data/processed/wifall/lstm-analysis/report.html`

Validation: `79 passed`; Ruff clean.
