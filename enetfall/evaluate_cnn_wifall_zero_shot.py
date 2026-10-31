"""Zero-shot evaluate ENetFall compact CNN on WiFall held-out participants."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wifall.loader import discover_recordings, load_recording

from .evaluate_wifall_zero_shot import adapt_wifall_amplitude
from .train_cnn import CompactCsiCnn, _device, _metrics, normalize_event


def _load_test_events(
    dataset_root: str | Path,
    split_manifest: str | Path,
    evaluation_scope: str = "test",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(dataset_root).expanduser().resolve()
    manifest = json.loads(Path(split_manifest).expanduser().read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or "recordings" not in manifest:
        raise ValueError("Unsupported or incomplete WiFall split manifest")
    if evaluation_scope not in ("test", "all"):
        raise ValueError("evaluation_scope must be 'test' or 'all'")
    references = {reference.recording_id: reference for reference in discover_recordings(root)}
    selected_entries = [
        entry
        for entry in manifest["recordings"]
        if evaluation_scope == "all" or entry.get("split") == "test"
    ]
    if not selected_entries:
        raise ValueError(f"WiFall {evaluation_scope} selection has no recordings")

    events = []
    for entry in selected_entries:
        recording_id = entry["recording_id"]
        reference = references.get(recording_id)
        if reference is None:
            raise ValueError(f"WiFall recording missing from dataset: {recording_id}")
        recording = load_recording(reference)
        if recording.target != entry["target"] or recording.row_count != entry["row_count"]:
            raise ValueError(f"WiFall manifest metadata mismatch: {recording_id}")
        amplitude = np.abs(recording.csi).astype(np.float32, copy=False)
        adapted = adapt_wifall_amplitude(amplitude)
        model_input = normalize_event(adapted)
        events.append(
            {
                "event_id": recording_id,
                "participant_id": recording.reference.participant_id,
                "label": recording.target,
                "input": torch.from_numpy(model_input),
                "source_samples": recording.row_count,
                "adapted_samples": int(adapted.shape[0]),
            }
        )
    if {event["label"] for event in events} != {"fall", "non-fall"}:
        raise ValueError(f"WiFall {evaluation_scope} selection must contain both target classes")
    return events, manifest


def evaluate_zero_shot(
    model_run: str | Path = "data/processed/enetfall/random-cnn-batch-comparison",
    dataset_root: str | Path = "data/raw/wifall/WiFall",
    split_manifest: str | Path = "data/processed/wifall/split.json",
    output_dir: str | Path = "data/processed/enetfall/wifall-zero-shot-cnn",
    *,
    evaluation_scope: str = "test",
) -> dict[str, Any]:
    run_dir = Path(model_run).expanduser()
    model_metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if model_metadata.get("model", {}).get("name") != "CompactCsiCnn":
        raise ValueError("Run artifact is not a CompactCsiCnn model")
    model_path = run_dir / model_metadata["model"]["artifact"]
    if not model_path.is_file():
        raise FileNotFoundError(f"CNN checkpoint not found: {model_path}")

    events, _ = _load_test_events(dataset_root, split_manifest, evaluation_scope)
    device = _device()
    model = CompactCsiCnn().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    scores = []
    with torch.inference_mode():
        for event in events:
            logits = model(event["input"].unsqueeze(0).to(device))
            scores.append(float(torch.softmax(logits, dim=1)[0, 1].cpu()))

    labels = np.asarray([1 if event["label"] == "fall" else 0 for event in events])
    threshold = float(model_metadata["threshold"])
    metrics = _metrics(labels, np.asarray(scores), threshold)
    predictions = [
        {
            "event_id": event["event_id"],
            "participant_id": event["participant_id"],
            "label": event["label"],
            "predicted_label": "fall" if score >= threshold else "non-fall",
            "fall_score": score,
            "source_samples": event["source_samples"],
            "adapted_samples": event["adapted_samples"],
        }
        for event, score in zip(events, scores, strict=True)
    ]
    target_counts = {
        label: sum(event["label"] == label for event in events)
        for label in ("fall", "non-fall")
    }
    source_counts = {
        "recordings": len(events),
        "rows": sum(event["source_samples"] for event in events),
        "targets": target_counts,
    }
    result = {
        "run_id": f"enetfall-compact-cnn-wifall-zero-shot-{evaluation_scope}-v1",
        "dataset": {
            "name": "WiFall",
            "evaluation_split": evaluation_scope,
            "recording_count": len(events),
            "participants": sorted({event["participant_id"] for event in events}),
            "source_counts": source_counts,
        },
        "source_model": {
            "dataset": "ENetFall",
            "model": "CompactCsiCnn",
            "run_id": model_metadata["run_id"],
            "checkpoint": str(model_path),
            "threshold": threshold,
            "threshold_source": "ENetFall random-split validation; unchanged for zero-shot evaluation",
        },
        "adapter": {
            "source": "52-subcarrier complex CSI amplitude, one stream",
            "transform": [
                "linear interpolation from 52 to 30 subcarriers",
                "resample from approximately 84 Hz to 125 Hz",
                "copy single stream into three identical pseudo-receiver channels",
                "per-recording, per-subcarrier z-score normalization",
            ],
            "caveat": "Approximate input-shape bridge; not physical receiver or domain equivalence.",
        },
        "metrics": metrics,
        "predictions": predictions,
    }
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "report.html").write_text(render_report(result), encoding="utf-8")
    return result


def render_report(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    matrix = metrics["confusion_matrix_nonfall_fall"]
    participants = ", ".join(result["dataset"]["participants"])
    source_counts = result["dataset"]["source_counts"]["targets"]
    evaluation_scope = result["dataset"]["evaluation_split"]
    participant_label = "Participants included" if evaluation_scope == "all" else "Held-out participants"
    scope_note = (
        "All manifest recordings are pooled here, including source train/validation/test participants; "
        "descriptive only, not a held-out WiFall generalization estimate."
        if evaluation_scope == "all"
        else "WiFall test participants held out by the saved participant split."
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ENetFall CNN zero-shot WiFall report</title><style>
body{{font:16px/1.5 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#18212b}}
h1,h2{{color:#17365d}}table{{border-collapse:collapse;margin:1rem 0;width:100%}}
th,td{{border:1px solid #ccd3dc;padding:.55rem;text-align:right}}th:first-child{{text-align:left}}
.card{{background:#f0f4f8;padding:1rem;border-radius:6px}}.warning{{background:#fff4d6;padding:1rem}}
</style></head><body>
<h1>ENetFall compact CNN → WiFall zero-shot report</h1>
<p>No WiFall training, fine-tuning, or threshold selection performed.</p>
<div class="card"><b>Threshold:</b> {metrics['threshold']:.6f} (fixed from ENetFall validation)<br>
<b>Recordings:</b> {metrics['support']} ({source_counts.get('fall', 0)} falls, {source_counts.get('non-fall', 0)} non-falls)<br>
<b>{participant_label}:</b> {html.escape(participants)}<br>
<b>Accuracy:</b> {metrics['accuracy']:.3f} &nbsp; <b>Balanced accuracy:</b> {metrics['balanced_accuracy']:.3f}<br>
<b>ROC-AUC:</b> {metrics['roc_auc']:.3f} &nbsp; <b>Fall precision:</b> {metrics['fall_precision']:.3f} &nbsp; <b>Fall recall:</b> {metrics['fall_recall']:.3f}</div>
<h2>Confusion matrix</h2><p>Rows are actual; columns predicted. Class order: non-fall, fall.</p>
<table><thead><tr><th>Actual / predicted</th><th>Non-fall</th><th>Fall</th></tr></thead><tbody>
<tr><th>Non-fall</th><td>{matrix[0][0]}</td><td>{matrix[0][1]}</td></tr>
<tr><th>Fall</th><td>{matrix[1][0]}</td><td>{matrix[1][1]}</td></tr></tbody></table>
<h2>Input adaptation</h2><ul><li>52 → 30 subcarriers by linear interpolation</li>
<li>Approximately 84 → 125 Hz temporal resampling</li><li>Single WiFall stream duplicated into three pseudo-receivers</li>
<li>Per-recording z-score normalization</li></ul>
<p class="warning">Adapter is approximate: WiFall has one CSI stream and different hardware, environment, event duration, and collection protocol. ENetFall model was selected with a random event split; this zero-shot result does not measure unseen-room or participant-independent ENetFall performance. {scope_note}</p>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_run", nargs="?", default="data/processed/enetfall/random-cnn-batch-comparison")
    parser.add_argument("dataset_root", nargs="?", default="data/raw/wifall/WiFall")
    parser.add_argument("split_manifest", nargs="?", default="data/processed/wifall/split.json")
    parser.add_argument("output_dir", nargs="?")
    parser.add_argument(
        "--all-recordings",
        action="store_true",
        help="evaluate every manifest recording, including train and validation participants",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or (
        "data/processed/enetfall/wifall-zero-shot-cnn-all"
        if args.all_recordings
        else "data/processed/enetfall/wifall-zero-shot-cnn"
    )
    scope = "all" if args.all_recordings else "test"
    result = evaluate_zero_shot(
        args.model_run,
        args.dataset_root,
        args.split_manifest,
        output_dir,
        evaluation_scope=scope,
    )
    print(json.dumps(result["metrics"], indent=2))
    print(f"HTML report: {Path(output_dir).expanduser() / 'report.html'}")


if __name__ == "__main__":
    main()
