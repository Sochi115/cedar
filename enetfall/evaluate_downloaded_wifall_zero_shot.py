"""Zero-shot evaluate downloaded ENetFall EfficientNet on all WiFall recordings."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from torchvision.transforms import Normalize

from wifall.loader import discover_recordings, load_recording

from .evaluate_wifall_zero_shot import adapt_wifall_amplitude
from .train_downloaded import build_model, get_default_device


def _load_events(dataset_root: str | Path, split_manifest: str | Path) -> list[dict[str, Any]]:
    root = Path(dataset_root).expanduser().resolve()
    manifest = json.loads(Path(split_manifest).expanduser().read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or "recordings" not in manifest:
        raise ValueError("Unsupported or incomplete WiFall split manifest")
    references = {reference.recording_id: reference for reference in discover_recordings(root)}
    events = []
    for entry in manifest["recordings"]:
        reference = references.get(entry["recording_id"])
        if reference is None:
            raise ValueError(f"WiFall recording missing from dataset: {entry['recording_id']}")
        recording = load_recording(reference)
        if recording.target != entry["target"] or recording.row_count != entry["row_count"]:
            raise ValueError(f"WiFall manifest metadata mismatch: {entry['recording_id']}")
        events.append(
            {
                "event_id": recording.reference.recording_id,
                "participant_id": recording.reference.participant_id,
                "label": recording.target,
                "amplitude": np.abs(recording.csi).astype(np.float32, copy=False),
                "source_samples": recording.row_count,
            }
        )
    return events


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    truth = labels == 1
    predicted = scores >= threshold
    matrix = confusion_matrix(truth, predicted, labels=[False, True])
    return {
        "support": int(labels.size),
        "threshold": threshold,
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "fall_precision": float(np.sum(truth & predicted) / max(np.sum(predicted), 1)),
        "fall_recall": float(np.sum(truth & predicted) / max(np.sum(truth), 1)),
        "confusion_matrix_nonfall_fall": matrix.tolist(),
    }


def evaluate_all_wifall(
    model_run: str | Path = "data/processed/enetfall/downloaded-train-run",
    dataset_root: str | Path = "data/raw/wifall/WiFall",
    split_manifest: str | Path = "data/processed/wifall/split.json",
    output_dir: str | Path = "data/processed/enetfall/downloaded-train-wifall-all",
) -> dict[str, Any]:
    run_dir = Path(model_run).expanduser()
    model_metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    preprocessing = model_metadata.get("preprocessing")
    if not preprocessing:
        raise ValueError("Training run is missing preprocessing statistics; retrain or update run.json")
    checkpoint = run_dir / model_metadata["model"]["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")

    events = _load_events(dataset_root, split_manifest)
    if {event["label"] for event in events} != {"fall", "non-fall"}:
        raise ValueError("WiFall dataset must contain both target classes")

    device = get_default_device()
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    imagenet_normalize = Normalize(
        mean=preprocessing["imagenet_mean"], std=preprocessing["imagenet_std"]
    )
    scale = float(preprocessing["training_max_after_imagenet_normalization"])
    center = float(preprocessing["training_mean_after_global_scaling"])
    scores = []
    with torch.inference_mode():
        for event in events:
            adapted = adapt_wifall_amplitude(event["amplitude"])
            channels_first = torch.from_numpy(adapted.transpose(1, 0, 2).copy()).unsqueeze(0)
            inputs = imagenet_normalize(channels_first) / scale - center
            logits = model(inputs.to(device))
            scores.append(float(torch.softmax(logits, dim=1)[0, 1].cpu()))

    labels = np.asarray([event["label"] == "fall" for event in events], dtype=np.int64)
    threshold = 0.5  # Original script uses argmax over two logits.
    metrics = _metrics(labels, np.asarray(scores), threshold)
    counts = {
        label: sum(event["label"] == label for event in events)
        for label in ("fall", "non-fall")
    }
    result = {
        "run_id": "enetfall-downloaded-train-wifall-zero-shot-all-v1",
        "dataset": {
            "name": "WiFall",
            "evaluation_scope": "all recordings pooled across source train/validation/test participants",
            "recording_count": len(events),
            "row_count": sum(event["source_samples"] for event in events),
            "class_counts": counts,
            "participants": sorted({event["participant_id"] for event in events}),
        },
        "source_model": {
            "dataset": "ENetFall",
            "model": "ImageNet-pretrained EfficientNet-B0",
            "run_id": model_metadata["run_id"],
            "checkpoint": str(checkpoint),
            "threshold": threshold,
            "threshold_source": "original argmax decision; no WiFall threshold tuning",
        },
        "adapter": {
            "operations": [
                "complex CSI magnitude",
                "linear interpolation from 52 to 30 subcarriers",
                "resampling from approximately 84 Hz to 125 Hz",
                "duplicate single WiFall stream across three pseudo-receiver channels",
                "original ImageNet channel normalization and ENetFall train-only global scaling",
            ],
            "caveat": "Approximate input bridge; WiFall and ENetFall hardware, receivers, rooms, and duration differ.",
        },
        "metrics": metrics,
        "predictions": [
            {
                "event_id": event["event_id"],
                "participant_id": event["participant_id"],
                "label": event["label"],
                "predicted_label": "fall" if score >= threshold else "non-fall",
                "fall_score": score,
            }
            for event, score in zip(events, scores, strict=True)
        ],
    }
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "report.html").write_text(render_report(result), encoding="utf-8")
    return result


def render_report(result: dict[str, Any]) -> str:
    dataset, metrics = result["dataset"], result["metrics"]
    matrix = metrics["confusion_matrix_nonfall_fall"]
    participants = ", ".join(dataset["participants"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Downloaded ENetFall model zero-shot WiFall report</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#18212b}}
h1,h2{{color:#17365d}}table{{border-collapse:collapse;margin:1rem 0;width:100%}}
th,td{{border:1px solid #ccd3dc;padding:.55rem;text-align:right}}th:first-child{{text-align:left}}
.card{{background:#f0f4f8;padding:1rem;border-radius:6px}}.warning{{background:#fff4d6;padding:1rem}}</style></head><body>
<h1>Downloaded ENetFall EfficientNet-B0 → WiFall zero-shot</h1>
<p>No WiFall training, fine-tuning, or threshold selection performed.</p>
<div class="card"><b>Recordings:</b> {dataset['recording_count']} ({dataset['class_counts']['fall']} falls, {dataset['class_counts']['non-fall']} non-falls)<br>
<b>Participants:</b> {html.escape(participants)}<br><b>Threshold:</b> {metrics['threshold']:.3f} (fixed)<br>
<b>Accuracy:</b> {metrics['accuracy']:.3f} &nbsp; <b>Balanced accuracy:</b> {metrics['balanced_accuracy']:.3f}<br>
<b>ROC-AUC:</b> {metrics['roc_auc']:.3f} &nbsp; <b>Fall precision:</b> {metrics['fall_precision']:.3f} &nbsp; <b>Fall recall:</b> {metrics['fall_recall']:.3f}</div>
<h2>Confusion matrix</h2><p>Rows actual; columns predicted. Class order: non-fall, fall.</p>
<table><thead><tr><th>Actual / predicted</th><th>Non-fall</th><th>Fall</th></tr></thead><tbody>
<tr><th>Non-fall</th><td>{matrix[0][0]}</td><td>{matrix[0][1]}</td></tr>
<tr><th>Fall</th><td>{matrix[1][0]}</td><td>{matrix[1][1]}</td></tr></tbody></table>
<p><b>Adapter:</b> map 52→30 subcarriers, resample ~84→125 Hz, duplicate single stream to three pseudo-receivers, apply source training normalization.</p>
<p class="warning">All WiFall recordings pooled, including saved train/validation/test participants; descriptive only, not an independent held-out estimate. Approximate hardware/signal bridge. ENetFall source model's random holdout also mixed rooms and was monitored during training.</p>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_run", nargs="?", default="data/processed/enetfall/downloaded-train-run")
    parser.add_argument("dataset_root", nargs="?", default="data/raw/wifall/WiFall")
    parser.add_argument("split_manifest", nargs="?", default="data/processed/wifall/split.json")
    parser.add_argument("output_dir", nargs="?", default="data/processed/enetfall/downloaded-train-wifall-all")
    args = parser.parse_args()
    result = evaluate_all_wifall(
        args.model_run, args.dataset_root, args.split_manifest, args.output_dir
    )
    print(json.dumps(result["metrics"], indent=2))
    print(f"HTML report: {Path(args.output_dir).expanduser() / 'report.html'}")


if __name__ == "__main__":
    main()
