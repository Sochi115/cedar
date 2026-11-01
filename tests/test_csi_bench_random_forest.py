import argparse
import json

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from csi_bench import random_forest
from csi_bench.random_forest import build_random_forest, extract_features


def test_extracts_finite_temporal_features_and_builds_forest():
    values = np.random.default_rng(7).normal(size=(500, 232))

    features = extract_features(values)
    forest = build_random_forest()

    assert features.shape == (192,)
    assert np.isfinite(features).all()
    assert forest.n_estimators == 400
    assert forest.class_weight == "balanced_subsample"
    assert forest.n_jobs == 2


def test_train_saves_model_and_reports_validation_and_test(tmp_path, monkeypatch):
    class Samples:
        def __init__(self, labels):
            self.samples = [
                (torch.full((1, 500, 232), float(label) + index / 100), label)
                for index, label in enumerate(labels)
            ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    datasets = {
        "train_id": Samples([0, 1, 0, 1]),
        "val_id": Samples([0, 1]),
        "test_id": Samples([0, 1]),
    }
    user_split = {"train_id": ["U1"], "val_id": ["U2"], "test_id": ["U3"]}
    user_split_path = tmp_path / "user_split.json"
    user_split_path.write_text(json.dumps(user_split), encoding="utf-8")

    def load_split(*args, **kwargs):
        assert kwargs["user_split"] == user_split
        return tmp_path, ["Fall", "Nonfall"], datasets

    monkeypatch.setattr(random_forest, "load_task_datasets", load_split)
    monkeypatch.setattr(
        random_forest,
        "build_random_forest",
        lambda jobs: RandomForestClassifier(
            n_estimators=5, random_state=42, n_jobs=jobs
        ),
    )

    result = random_forest.train(
        argparse.Namespace(
            dataset_root=tmp_path,
            task="FallDetection",
            data_key="CSI_amps",
            jobs=2,
            user_split=user_split_path,
            output_dir=tmp_path / "run",
        )
    )

    assert result["sample_counts"] == {
        name: len(data) for name, data in datasets.items()
    }
    assert result["test_metrics"]["confusion_matrix"]
    assert result["training_history"][-1]["trees"] == 5
    assert (tmp_path / "run" / "csi-bench-random-forest.joblib").is_file()
    assert (tmp_path / "run" / "run.json").is_file()
    assert json.loads((tmp_path / "run" / "user_split.json").read_text()) == user_split
    assert result["split"]["strategy"] == "user-held-out"
    report = (tmp_path / "run" / "report.html").read_text(encoding="utf-8")
    assert "Training and validation loss" in report
    assert "Held-out test confusion matrix" in report
    assert "Data split: user-held-out" in report
