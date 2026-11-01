"""Evaluate a trained CSI-Bench Random Forest on ENetFall and WiFall."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from enetfall.evaluate_wifall_zero_shot import _read_wifall_recording
from enetfall.train_svm import ROOM_SPLITS, iter_enetfall_events
from evaluation.core import evaluate_classifier, save_evaluation_result
from evaluation.report import render_html_report_file

from .cnn import FEATURE_SIZE, WINDOW_LENGTH
from .random_forest import MODEL_NAME, extract_features

LABELS = ("non-fall", "fall")


def _canonical_label(name: str) -> str:
    normalized = re.sub(r"[^a-z]", "", name.casefold())
    if normalized in {"fall", "falling"}:
        return "fall"
    if normalized in {"nonfall", "notfall", "nofall", "nonfalling"}:
        return "non-fall"
    raise ValueError(f"CSI-Bench class name is not a fall/non-fall label: {name!r}")


class _BinaryClassifier:
    """Expose CSI-Bench numeric classes using shared fall-detection labels."""

    classes_ = np.asarray(LABELS)

    def __init__(self, artifact: dict[str, Any]) -> None:
        self.estimator = artifact["estimator"]
        names = artifact["class_names"]
        self.class_labels = {}
        for class_id in self.estimator.classes_:
            index = int(class_id)
            if index < 0 or index >= len(names):
                raise ValueError(f"Model class index outside class_names: {class_id!r}")
            self.class_labels[class_id] = _canonical_label(names[index])
        if set(self.class_labels.values()) != set(LABELS):
            raise ValueError(
                "CSI-Bench Random Forest must contain fall and non-fall classes"
            )

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(
            [self.class_labels[value] for value in self.estimator.predict(features)]
        )

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = self.estimator.predict_proba(features)
        return probabilities[
            :,
            [
                next(
                    index
                    for index, value in enumerate(self.estimator.classes_)
                    if self.class_labels[value] == label
                )
                for label in LABELS
            ],
        ]


def _csi_image(values: np.ndarray) -> np.ndarray:
    """Interpolate one time-by-feature map to CSI-Bench's normalized input shape."""
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2 or min(source.shape) < 1 or not np.isfinite(source).all():
        raise ValueError("CSI map must be a non-empty finite time-by-feature array")

    # shortcut: index-based interpolation is not physical channel alignment; replace when dataset-specific calibration is available.
    time = np.linspace(0, source.shape[0] - 1, WINDOW_LENGTH)
    feature = np.linspace(0, source.shape[1] - 1, FEATURE_SIZE)
    by_time = np.column_stack(
        [
            np.interp(time, np.arange(source.shape[0]), source[:, index])
            for index in range(source.shape[1])
        ]
    )
    resized = np.vstack(
        [np.interp(feature, np.arange(source.shape[1]), row) for row in by_time]
    )
    scale = max(float(resized.std()), 1e-8)
    normalized = (resized - resized.mean()) / scale
    return normalized.astype(np.float32)


def _csi_features(values: np.ndarray) -> np.ndarray:
    return extract_features(_csi_image(values))


def _enetfall_test(
    dataset_root: str | Path,
    evaluation_scope: str,
    feature_extractor: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[list[str], np.ndarray, list[str]]:
    feature_extractor = feature_extractor or _csi_features
    ids: list[str] = []
    features = []
    labels: list[str] = []
    for event in iter_enetfall_events(dataset_root):
        if evaluation_scope == "test" and ROOM_SPLITS[event["room"]] != "test":
            continue
        ids.append(event["event_id"])
        features.append(
            feature_extractor(
                event["amplitude"].reshape(event["amplitude"].shape[0], -1)
            )
        )
        labels.append(event["label"])
    if not features:
        raise ValueError(
            f"No ENetFall events found for evaluation scope {evaluation_scope!r}"
        )
    return ids, np.stack(features), labels


def _wifall_test(
    dataset_root: str | Path,
    split_manifest: str | Path,
    evaluation_scope: str,
    feature_extractor: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[list[str], np.ndarray, list[str], list[str]]:
    root = Path(dataset_root).expanduser().resolve()
    manifest = json.loads(Path(split_manifest).expanduser().read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or "recordings" not in manifest:
        raise ValueError("Unsupported or incomplete WiFall split manifest")
    if evaluation_scope not in ("test", "all"):
        raise ValueError("evaluation_scope must be 'test' or 'all'")
    feature_extractor = feature_extractor or _csi_features

    ids: list[str] = []
    features = []
    labels: list[str] = []
    participants: list[str] = []
    for entry in manifest["recordings"]:
        if evaluation_scope == "test" and entry.get("split") != "test":
            continue
        relative = Path(entry["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"Unsafe WiFall recording path in split manifest: {relative}"
            )
        target = entry.get("target")
        if target not in LABELS:
            raise ValueError(f"Invalid target in WiFall split manifest: {target!r}")
        path = root / relative
        amplitudes = _read_wifall_recording(
            path, target, expected_rows=entry.get("row_count")
        )
        ids.append(str(entry["recording_id"]))
        features.append(feature_extractor(amplitudes))
        labels.append(target)
        participants.append(str(entry["participant_id"]))

    if not features or set(labels) != set(LABELS):
        raise ValueError(
            f"WiFall {evaluation_scope} selection must contain fall and non-fall recordings"
        )
    return ids, np.stack(features), labels, participants


def _confusion_table(evaluation: dict[str, Any]) -> str:
    labels = evaluation["labels"]
    header = "".join(f"<th>{html.escape(label)}</th>" for label in labels)
    rows = []
    for label, counts in zip(labels, evaluation["confusion_matrix"]):
        cells = "".join(f"<td>{count}</td>" for count in counts)
        rows.append(f"<tr><th>{html.escape(label)}</th>{cells}</tr>")
    return (
        "<table><thead><tr><th>True \\ Predicted</th>" + header + "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def _combined_report(
    results: list[dict[str, Any]], model_name: str = "CSI-Bench Random Forest"
) -> str:
    sections = []
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "sensitivity",
        "specificity",
        "roc_auc",
        "pr_auc",
    )
    for result in results:
        scope = result["evaluation_split"]
        evaluation = result["evaluations"][scope]
        metrics = evaluation["metrics"]
        scope_title = "full dataset" if scope == "all" else "held-out test"
        scope_detail = (
            "All recordings/events, including source train and validation partitions"
            if scope == "all"
            else "Held-out test split only"
        )
        rows = "".join(
            f"<tr><th>{html.escape(name.replace('_', ' ').title())}</th><td>{metrics[name]:.4f}</td></tr>"
            for name in metric_names
        )
        sections.append(
            f"<section><h2>{html.escape(result['dataset']['name'])} {scope_title}</h2>"
            f"<p>{len(evaluation['y_true'])} recordings/events · {scope_detail}</p>"
            f"<h3>Metrics</h3><table>{rows}</table><h3>Confusion matrix</h3>"
            f"<p>Rows are true labels; columns are predicted labels.</p>{_confusion_table(evaluation)}"
            f"<p><a href='{html.escape(result['dataset']['name'].lower())}/report.html'>Detailed report</a></p></section>"
        )
    return (
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>External model evaluation</title>
<style>body{font:16px system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#222}section{margin:2rem 0;padding:1rem;border:1px solid #ccc;border-radius:8px}table{border-collapse:collapse;margin:1rem 0}th,td{border:1px solid #ccc;padding:.5rem 1rem;text-align:left}p.note{color:#555}</style></head><body>
<h1>External model evaluation</h1>
<p class="note">Zero-shot tests. Dataset signals are index-interpolated to CSI-Bench's 500 × 232 input shape; feature alignment is approximate, so results are exploratory and not directly comparable to in-domain CSI-Bench metrics.</p>
"""
        + f'<p class="note">Model: {html.escape(model_name)}</p>'
        + "".join(sections)
        + "</body></html>"
    )


def run_evaluations(
    model_path: str | Path,
    enetfall_root: str | Path,
    wifall_root: str | Path,
    wifall_split: str | Path,
    output_dir: str | Path,
    evaluation_scope: str = "test",
) -> Path:
    artifact = joblib.load(Path(model_path).expanduser())
    if not isinstance(artifact, dict) or not {"estimator", "class_names"}.issubset(
        artifact
    ):
        raise ValueError("Not a CSI-Bench Random Forest artifact")
    model = _BinaryClassifier(artifact)
    if model.estimator.n_features_in_ != 192:
        raise ValueError("CSI-Bench Random Forest must accept 192 temporal features")

    if evaluation_scope not in ("test", "all"):
        raise ValueError("evaluation_scope must be 'test' or 'all'")
    enetfall_ids, enetfall_features, enetfall_labels = _enetfall_test(
        enetfall_root, evaluation_scope
    )
    wifall_ids, wifall_features, wifall_labels, wifall_groups = _wifall_test(
        wifall_root, wifall_split, evaluation_scope
    )
    tests = {
        "enetfall": (
            "ENetFall",
            enetfall_ids,
            enetfall_features,
            enetfall_labels,
            None,
        ),
        "wifall": ("WiFall", wifall_ids, wifall_features, wifall_labels, wifall_groups),
    }
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for run_id, (dataset_name, ids, features, labels, groups) in tests.items():
        result = evaluate_classifier(
            model,
            features,
            labels,
            label_order=LABELS,
            positive_label="fall",
            run_id=f"csi-bench-random-forest-{run_id}-zero-shot-{evaluation_scope}",
            split_name=evaluation_scope,
            dataset_metadata={"name": dataset_name},
            task_metadata={"name": "binary_fall_detection", "signal": "WiFi CSI"},
            model_metadata={
                "family": "random_forest",
                "name": MODEL_NAME,
                "trained_on": "CSI-Bench",
            },
            feature_metadata={"name": "csi_bench_temporal_statistics", "features": 192},
            preprocessing_metadata={
                "steps": [
                    "linear interpolation to 500 time samples and 232 features",
                    "CSI-Bench global normalization",
                    "CSI-Bench robust temporal features",
                ],
                "approximation_warning": "Index interpolation does not establish physical feature alignment between datasets.",
            },
            split_metadata={
                "strategy": (
                    "all recordings/events across source partitions"
                    if evaluation_scope == "all"
                    else "held-out test split"
                ),
                "threshold_selection": "none; model predictions used as trained",
            },
            observation_ids=ids,
            group_ids=groups,
        )
        run_dir = destination / run_id
        result_path = save_evaluation_result(result, run_dir / "run.json")
        render_html_report_file(result_path, run_dir / "report.html")
        results.append(result)

    report_path = destination / "report.html"
    report_path.write_text(_combined_report(results), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "data/processed/csi-bench/random-forest/csi-bench-random-forest.joblib"
        ),
    )
    parser.add_argument("--enetfall-root", type=Path, default=Path("data/raw/ENetFall"))
    parser.add_argument(
        "--wifall-root", type=Path, default=Path("data/raw/wifall/WiFall")
    )
    parser.add_argument(
        "--wifall-split", type=Path, default=Path("data/processed/wifall/split.json")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/csi-bench/external-random-forest"),
    )
    parser.add_argument(
        "--all-data",
        action="store_true",
        help="evaluate all source recordings/events, not only held-out test splits",
    )
    args = parser.parse_args()
    print(
        f"Wrote combined report: {run_evaluations(args.model, args.enetfall_root, args.wifall_root, args.wifall_split, args.output_dir, 'all' if args.all_data else 'test')}"
    )


if __name__ == "__main__":
    main()
