# WiFall SVM Training and Static Evaluation Reports

Status: open
Label: ready-for-agent

## Destination

Create a reproducible WiFall fall-detection baseline using an SVM, persist model-agnostic evaluation artifacts, and render individual and comparative static HTML reports.

## Scope

- Initial task: binary fall versus non-fall detection.
- Preserve original activity labels for future multiclass experiments.
- Use participant-held-out evaluation to prevent participant leakage.
- Keep visualization independent from model implementation and dataset-specific code.

## Tickets

- [Make WiFall recordings reproducibly loadable](completed/wifall-svm-report-01-load-and-split.md)
- [Train and evaluate WiFall SVM baseline](completed/wifall-svm-report-02-train-svm.md)
- [Build generic model validation and static evaluation report](completed/wifall-svm-report-03-static-report.md)
- [Compare model and dataset runs](completed/wifall-svm-report-04-compare-runs.md)

## Decisions so far

- [Make WiFall recordings reproducibly loadable](completed/wifall-svm-report-01-load-and-split.md): WiFall loading, CSI parsing, recording-level examples, and deterministic participant-held-out manifests implemented and verified against downloaded dataset counts.
- [Train and evaluate WiFall SVM baseline](completed/wifall-svm-report-02-train-svm.md): 260 CSI amplitude-statistic features feed a training-only StandardScaler and SVC; generic run artifacts store predictions, scores, metrics, and split metadata.
- [Build generic model validation and static evaluation report](completed/wifall-svm-report-03-static-report.md): Generic validation now produces model- and dataset-independent result artifacts; static HTML consumes only those artifacts.
- SVM baseline starts with binary fall detection.
- Complete recordings and participants remain within one split.
- Run artifacts must contain enough metadata and predictions for model-agnostic reporting.
- Static reports must work without a web server.
- [Compare model and dataset runs](completed/wifall-svm-report-04-compare-runs.md): Generic comparison report now indexes valid run artifacts, links individual reports, compares metrics and metadata, and warns about incompatible protocols.

## Out of scope

- Production alerting or deployment.
- Real-time inference.
- Cross-dataset benchmarking before dataset compatibility is established.
