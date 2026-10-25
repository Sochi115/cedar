from __future__ import annotations

import numpy as np

from enetfall.train_random_forest import build_random_forest


def test_random_forest_fits_event_features_and_returns_fall_probabilities() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(24, 216))
    labels = np.array(["non-fall"] * 12 + ["fall"] * 12)
    features[12:, :8] += 2.0

    model = build_random_forest()
    model.fit(features, labels)
    probabilities = model.predict_proba(features)

    assert model.n_estimators == 400
    assert model.class_weight == "balanced_subsample"
    assert probabilities.shape == (24, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
