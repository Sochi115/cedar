# Evaluate CSI augmentation for Random Forest

Status: open
Type: task
Label: ready-for-agent
Blocked by: completed/wifall-model-improvement-04-regularized-calibrated-random-forest.md, completed/wifall-model-improvement-06-hard-example-analysis.md

## What to build

Test participant-safe CSI augmentation for hard fall patterns and channel variation. Compare augmented and untouched Random Forest training without changing validation or test recordings.

## Acceptance criteria

- [ ] Supports controlled CSI noise, temporal jitter, subcarrier dropout, and safe window perturbation.
- [ ] Augmentation applies only to training participants and preserves participant grouping.
- [ ] Augmented examples retain labels only when transformation preserves fall semantics.
- [ ] Each augmentation family is evaluated separately before combined use.
- [ ] Result artifact records augmentation configuration and generated-example counts.
- [ ] Report compares augmented and untouched runs with per-participant metrics.
- [ ] Tests verify deterministic augmentation, split isolation, label preservation, and artifact compatibility.
