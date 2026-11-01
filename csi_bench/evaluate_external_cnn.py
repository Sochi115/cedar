"""Evaluate a trained CSI-Bench CNN on ENetFall and WiFall."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from evaluation.core import evaluate_classifier, save_evaluation_result
from evaluation.report import render_html_report_file

from .cnn import (
    FEATURE_SIZE,
    MODEL_ARCHITECTURE,
    MODEL_NAME,
    WINDOW_LENGTH,
    DeepCnn,
)
from .evaluate_external import (
    LABELS,
    _canonical_label,
    _combined_report,
    _csi_image,
    _enetfall_test,
    _wifall_test,
)


class _CnnClassifier:
    """Expose CNN probabilities in shared non-fall/fall label order."""

    classes_ = np.asarray(LABELS)

    def __init__(
        self, model: nn.Module, class_names: list[str], device: torch.device
    ) -> None:
        self.model = model.eval()
        self.device = device
        model_labels = [_canonical_label(name) for name in class_names]
        if set(model_labels) != set(LABELS):
            raise ValueError("CSI-Bench CNN must contain fall and non-fall classes")
        self.class_indexes = [model_labels.index(label) for label in LABELS]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (WINDOW_LENGTH, FEATURE_SIZE):
            raise ValueError(
                f"Expected CNN inputs shaped (N, {WINDOW_LENGTH}, {FEATURE_SIZE}), got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("CNN inputs contain NaN or infinity")

        batches = []
        with torch.inference_mode():
            for start in range(0, len(values), 32):
                batch = torch.from_numpy(values[start : start + 32]).to(self.device)
                logits = self.model(batch.unsqueeze(1))
                batches.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(batches)[:, self.class_indexes]

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(LABELS)[self.predict_proba(features).argmax(axis=1)]


def _device(name: str) -> torch.device:
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


def _load_classifier(model_path: str | Path, device: torch.device) -> _CnnClassifier:
    checkpoint = torch.load(
        Path(model_path).expanduser(), map_location=device, weights_only=True
    )
    if checkpoint.get("architecture") != MODEL_ARCHITECTURE:
        raise ValueError("Unsupported CSI-Bench CNN architecture")
    if checkpoint.get("input_shape") != [1, WINDOW_LENGTH, FEATURE_SIZE]:
        raise ValueError("CSI-Bench CNN checkpoint has unexpected input shape")
    class_names = checkpoint["class_names"]
    model = DeepCnn(len(class_names)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return _CnnClassifier(model, class_names, device)


def run_evaluations(
    model_path: str | Path,
    enetfall_root: str | Path,
    wifall_root: str | Path,
    wifall_split: str | Path,
    output_dir: str | Path,
    evaluation_scope: str = "test",
    device: str = "auto",
) -> Path:
    if evaluation_scope not in ("test", "all"):
        raise ValueError("evaluation_scope must be 'test' or 'all'")
    inference_device = _device(device)
    model = _load_classifier(model_path, inference_device)
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for run_id, dataset_name in (("enetfall", "ENetFall"), ("wifall", "WiFall")):
        if run_id == "enetfall":
            ids, features, labels = _enetfall_test(
                enetfall_root, evaluation_scope, _csi_image
            )
            groups = None
        else:
            ids, features, labels, groups = _wifall_test(
                wifall_root, wifall_split, evaluation_scope, _csi_image
            )
        result = evaluate_classifier(
            model,
            features,
            labels,
            label_order=LABELS,
            positive_label="fall",
            run_id=f"csi-bench-cnn-{run_id}-zero-shot-{evaluation_scope}",
            split_name=evaluation_scope,
            dataset_metadata={"name": dataset_name},
            task_metadata={"name": "binary_fall_detection", "signal": "WiFi CSI"},
            model_metadata={
                "family": "cnn",
                "name": MODEL_NAME,
                "trained_on": "CSI-Bench",
                "device": str(inference_device),
            },
            feature_metadata={
                "name": "csi_bench_normalized_time_feature_map",
                "input_shape": [1, WINDOW_LENGTH, FEATURE_SIZE],
            },
            preprocessing_metadata={
                "steps": [
                    "linear interpolation to 500 time samples and 232 features",
                    "CSI-Bench per-sample global normalization",
                    "DeepCnn inference",
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
        del features

    report_path = destination / "report.html"
    report_path.write_text(_combined_report(results, "CSI-Bench CNN"), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/processed/csi-bench/cnn/csi-bench-cnn.pt"),
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
        default=Path("data/processed/csi-bench/external-cnn"),
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--all-data", action="store_true")
    args = parser.parse_args()
    print(
        f"Wrote combined report: {run_evaluations(args.model, args.enetfall_root, args.wifall_root, args.wifall_split, args.output_dir, 'all' if args.all_data else 'test', args.device)}"
    )


if __name__ == "__main__":
    main()
