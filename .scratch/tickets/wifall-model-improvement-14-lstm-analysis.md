# Analyze LSTM fall-detection usefulness

Status: open
Type: research
Label: ready-for-agent
Blocked by: wifall-model-improvement-13-lstm-duration-normalized.md

## What to build

Analyze the finalized LSTM run against duration-normalized CNN, improved SVM, and Random Forest runs. Determine whether LSTM improves aggregate or per-participant fall detection and whether it contributes complementary detections that justify further investigation.

## Acceptance criteria

- [ ] Reports aggregate and per-participant accuracy, balanced accuracy, precision, recall, specificity, false-positive rate, and F1.
- [ ] Compares LSTM metrics against compatible duration-normalized model runs.
- [ ] Identifies false negatives, false positives, and participant-specific failure patterns.
- [ ] Measures overlap and complementarity of LSTM detections and errors against comparator models.
- [ ] Keeps test results out of architecture, threshold, and model selection decisions.
- [ ] Produces a self-contained analysis artifact and HTML report.
- [ ] Records limitations, uncertainty from the held-out participant count, and a clear adopt, investigate, or reject recommendation.
