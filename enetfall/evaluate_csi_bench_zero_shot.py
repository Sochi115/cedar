"""Evaluate an ENetFall Random Forest on all CSI-Bench fall samples, zero-shot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import joblib
import numpy as np

from evaluation.core import evaluate_classifier, save_evaluation_result
from evaluation.report import render_html_report_file

from .compare_models import extract_windowed_features
from .train_random_forest import EnetFallRandomForest
from .train_svm import LABELS

TARGET_SUBCARRIERS = 30
PSEUDO_RECEIVERS = 3


def adapt_csi_bench_amplitude(values: np.ndarray) -> np.ndarray:
    """Map CSI-Bench's single variable-bin stream to ENetFall's 3 x 30 input."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1 or values.shape[0] < 4:
        raise ValueError("CSI amplitude must have shape (at least 4 samples, 1+ subcarriers)")
    if not np.isfinite(values).all():
        raise ValueError("CSI amplitude contains non-finite values")

    source_grid = np.arange(values.shape[1], dtype=np.float64)
    target_grid = np.linspace(0.0, values.shape[1] - 1, TARGET_SUBCARRIERS)
    mapped = np.stack([np.interp(target_grid, source_grid, row) for row in values])
    mapped = np.maximum(mapped, 0.0).astype(np.float32, copy=False)
    return np.repeat(mapped[:, None, :], PSEUDO_RECEIVERS, axis=1)


def _task_dir(dataset_root: Path, task: str) -> Path:
    for candidate in (dataset_root / task, dataset_root / "Multitask" / task, dataset_root):
        if (candidate / "metadata" / "sample_metadata.csv").is_file():
            return candidate
    raise FileNotFoundError(f"Could not find metadata for {task!r} under {dataset_root}")


def _load_records(task_dir: Path, data_key: str) -> tuple[list[dict[str, str]], np.ndarray]:
    with (task_dir / "metadata" / "label_mapping.json").open(encoding="utf-8") as source:
        label_mapping = json.load(source)["label_to_idx"]
    if set(label_mapping) != {"Fall", "Nonfall"}:
        raise ValueError(f"Expected Fall/Nonfall labels, got {sorted(label_mapping)}")

    with (task_dir / "metadata" / "sample_metadata.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        reader = csv.DictReader(source)
        required = {"id", "file_path", "label", "user", "environment", "Difficulty"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Metadata must contain columns {sorted(required)}")
        rows = list(reader)

    records = []
    features = []
    seen_ids = set()
    for row in rows:
        sample_id = row["id"]
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"Missing or duplicate sample ID: {sample_id!r}")
        seen_ids.add(sample_id)
        label = {"Fall": "fall", "Nonfall": "non-fall"}.get(row["label"])
        if label is None:
            raise ValueError(f"Unknown CSI-Bench label {row['label']!r} for {sample_id}")

        relative_path = Path(row["file_path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe CSI-Bench sample path: {relative_path}")
        sample_path = (task_dir / relative_path).resolve()
        if not sample_path.is_relative_to(task_dir.resolve()):
            raise ValueError(f"CSI-Bench sample path escapes task directory: {relative_path}")
        with h5py.File(sample_path, "r") as sample:
            if data_key not in sample:
                raise ValueError(f"{sample_path} has no HDF5 dataset {data_key!r}")
            csi = np.asarray(sample[data_key])
        if csi.ndim != 3 or csi.shape[0] < 1 or csi.shape[1:] != (500, 1):
            raise ValueError(f"Expected CSI shape (subcarriers, 500, 1), got {csi.shape} in {sample_path}")

        amplitude = adapt_csi_bench_amplitude(csi[:, :, 0].T)
        features.append(extract_windowed_features(amplitude, robust_normalize=True))
        records.append(
            {
                "id": sample_id,
                "label": label,
                "user": row["user"],
                "environment": row["environment"],
                "difficulty": row["Difficulty"],
                "subcarrier_count": str(csi.shape[0]),
            }
        )

    if not records or {record["label"] for record in records} != set(LABELS):
        raise ValueError("CSI-Bench metadata must contain both fall and non-fall samples")
    return records, np.vstack(features)


class _ThresholdedModel:
    """Expose the saved ENetFall validation threshold to generic evaluation."""

    def __init__(self, model: EnetFallRandomForest) -> None:
        self.model = model
        self.classes_ = np.asarray(["non-fall", "fall"])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        estimator = self.model.estimator
        indexes = [list(estimator.classes_).index(label) for label in self.classes_]
        return estimator.predict_proba(features)[:, indexes]

    def predict(self, features: np.ndarray) -> np.ndarray:
        positive_scores = self.predict_proba(features)[:, 1]
        return np.where(positive_scores >= self.model.threshold, "fall", "non-fall")


def evaluate_csi_bench(
    model_path: str | Path = "data/processed/enetfall/random-forest-run/model.joblib",
    dataset_root: str | Path = "data/raw/csi-bench",
    output_dir: str | Path = "data/processed/enetfall/csi-bench-zero-shot-all-run",
    *,
    task: str = "FallDetection",
    data_key: str = "CSI_amps",
) -> dict[str, Any]:
    model = joblib.load(Path(model_path).expanduser())
    if not isinstance(model, EnetFallRandomForest):
        raise TypeError("Model artifact is not an EnetFallRandomForest")
    if model.estimator.n_features_in_ != 216:
        raise ValueError(f"Expected 216-feature ENetFall forest, got {model.estimator.n_features_in_}")

    task_dir = _task_dir(Path(dataset_root).expanduser().resolve(), task)
    records, features = _load_records(task_dir, data_key)
    source_splits = {}
    for split_name in ("train_id", "val_id", "test_id"):
        split_path = task_dir / "splits" / f"{split_name}.json"
        if split_path.is_file():
            source_splits[split_name] = len(json.loads(split_path.read_text(encoding="utf-8")))

    result = evaluate_classifier(
        _ThresholdedModel(model),
        features,
        [record["label"] for record in records],
        label_order=LABELS,
        positive_label="fall",
        run_id="enetfall-random-forest-csi-bench-zero-shot-all-v1",
        split_name="all_samples",
        dataset_metadata={
            "name": "CSI-Bench",
            "task": task,
            "sample_count": len(records),
            "label_counts": {
                label: sum(record["label"] == label for record in records) for label in LABELS
            },
            "user_count": len({record["user"] for record in records}),
            "environment_count": len({record["environment"] for record in records}),
        },
        task_metadata={"name": "binary_fall_detection", "signal": "WiFi CSI amplitude"},
        model_metadata={
            "family": "random_forest",
            "name": "ENetFall RandomForestClassifier (zero-shot)",
            "artifact": Path(model_path).name,
            "trained_on": "ENetFall",
            "threshold": model.threshold,
        },
        feature_metadata={
            "name": "enetfall_window_robust_temporal_v1_with_approximate_csi_bench_adapter",
            "count": int(features.shape[1]),
            "source_shape": ["subcarriers", 500, 1],
            "source_subcarrier_counts": sorted({int(record["subcarrier_count"]) for record in records}),
            "model_shape": [500, PSEUDO_RECEIVERS, TARGET_SUBCARRIERS],
        },
        preprocessing_metadata={
            "steps": [
                "transpose CSI-Bench samples to time x subcarrier",
                "linear interpolation by subcarrier index: 232 to 30",
                "duplicate single stream into 3 pseudo-receiver channels",
                "ENetFall per-event median/MAD normalization and 216 temporal features",
            ],
            "warning": (
                "CSI-Bench's 232-bin single-stream CSI is not physically aligned to ENetFall's "
                "three 30-bin receivers. Linear frequency interpolation and pseudo-receiver "
                "duplication are shape adapters only; score is exploratory domain transfer."
            ),
        },
        split_metadata={
            "strategy": "all rows in CSI-Bench task sample_metadata.csv; no fitting or tuning",
            "source_official_split_counts": source_splits,
            "warning": (
                "Descriptive all-data evaluation includes CSI-Bench's original train, validation, "
                "and test partitions; not an untouched held-out benchmark."
            ),
            "threshold_source": "fixed from ENetFall validation; no CSI-Bench labels used for threshold selection",
        },
        observation_ids=[record["id"] for record in records],
        group_ids=[record["user"] for record in records],
    )
    output = Path(output_dir).expanduser()
    result_path = save_evaluation_result(result, output / "run.json")
    report_path = render_html_report_file(result_path, output / "report.html")
    print(f"Evaluated {len(records)} CSI-Bench samples")
    print(f"Fixed ENetFall threshold: {model.threshold:.6f}")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Saved evaluation: {result_path}")
    print(f"Saved report: {report_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", nargs="?", default="data/processed/enetfall/random-forest-run/model.joblib")
    parser.add_argument("dataset_root", nargs="?", default="data/raw/csi-bench")
    parser.add_argument("output_dir", nargs="?", default="data/processed/enetfall/csi-bench-zero-shot-all-run")
    parser.add_argument("--task", default="FallDetection")
    parser.add_argument("--data-key", default="CSI_amps")
    args = parser.parse_args()
    evaluate_csi_bench(args.model_path, args.dataset_root, args.output_dir, task=args.task, data_key=args.data_key)


if __name__ == "__main__":
    main()
