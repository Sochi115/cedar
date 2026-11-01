"""Combine CSI-Bench cross-dataset evaluations into one HTML report."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ENetFall", "WiFall")
MODEL_SOURCES = (
    (
        "CSI-Bench Random Forest (standard split)",
        "data/processed/csi-bench/external-random-forest",
        "data/processed/csi-bench/random-forest/run.json",
    ),
    (
        "CSI-Bench CNN (standard split)",
        "data/processed/csi-bench/external-cnn",
        "data/processed/csi-bench/cnn/run.json",
    ),
    (
        "CSI-Bench CNN-LSTM (user v3)",
        "data/processed/csi-bench/cnn-lstm-user-v3/cross-dataset",
        "data/processed/csi-bench/cnn-lstm-user-v3/run.json",
    ),
    (
        "CSI-Bench Random Forest (user split)",
        "data/processed/csi-bench/random-forest-user/cross-dataset",
        "data/processed/csi-bench/random-forest-user/run.json",
    ),
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def load_models(sources: tuple = MODEL_SOURCES) -> list[dict[str, Any]]:
    """Load paired all-data evaluations and model training metadata."""
    models = []
    for name, evaluation_dir, training_path in sources:
        directory = ROOT / evaluation_dir
        evaluations = {}
        model_family = None
        for dataset in DATASETS:
            artifact_path = directory / dataset.lower() / "run.json"
            artifact = _read_json(artifact_path)
            if artifact.get("evaluation_split") != "all":
                raise ValueError(f"Expected full-dataset evaluation in {artifact_path}")
            if artifact.get("dataset", {}).get("name") != dataset:
                raise ValueError(f"Unexpected dataset in {artifact_path}")
            family = artifact.get("model", {}).get("family")
            if model_family is not None and family != model_family:
                raise ValueError(
                    f"Model family differs between evaluations: {directory}"
                )
            model_family = family
            evaluations[dataset] = artifact["evaluations"]["all"]
        if model_family not in {"random_forest", "cnn", "cnn_lstm"}:
            raise ValueError(
                f"Unsupported model family for {directory}: {model_family!r}"
            )
        models.append(
            {
                "name": name,
                "family": model_family,
                "evaluation_dir": directory,
                "evaluations": evaluations,
                "training": _read_json(ROOT / training_path),
            }
        )
    return models


def average_balanced_accuracy(model: dict[str, Any]) -> float:
    """Weight datasets equally so larger dataset does not dominate ranking."""
    scores = [
        model["evaluations"][dataset]["metrics"]["balanced_accuracy"]
        for dataset in DATASETS
    ]
    return sum(scores) / len(scores)


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _metric(value: Any) -> str:
    return "Not available" if value is None else f"{value:.4f}"


def _confusion_table(evaluation: dict[str, Any]) -> str:
    labels = evaluation["labels"]
    columns = "".join(f"<th>{_escape(label)}</th>" for label in labels)
    rows = "".join(
        f"<tr><th>{_escape(label)}</th>"
        + "".join(f"<td>{count}</td>" for count in counts)
        + "</tr>"
        for label, counts in zip(labels, evaluation["confusion_matrix"])
    )
    return (
        "<table class='confusion'><thead><tr><th>True \\ Predicted</th>"
        + columns
        + f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def _split_section(models: list[dict[str, Any]]) -> str:
    groups = {}
    for model in models:
        training = model["training"]
        split = training.get("split", {})
        strategy = (
            training.get("config", {}).get("split_strategy")
            or split.get("strategy")
            or "official CSI-Bench ID lists"
        )
        counts = training.get("sample_counts", {})
        users = split.get("users", {})
        key = (
            strategy,
            json.dumps(counts, sort_keys=True),
            json.dumps(users, sort_keys=True),
        )
        groups.setdefault(
            key, {"strategy": strategy, "counts": counts, "users": users, "models": []}
        )
        groups[key]["models"].append(model["name"])

    rows = []
    for group in groups.values():
        counts = group["counts"]
        count_text = ", ".join(
            f"{name.removesuffix('_id')}: {counts[name]}"
            for name in ("train_id", "val_id", "test_id")
            if name in counts
        )
        users = group["users"]
        if users:
            user_text = "; ".join(
                f"{name.removesuffix('_id')} users: {', '.join(values)}"
                for name, values in users.items()
            )
            split_detail = f"By user. {_escape(user_text)}"
        else:
            split_detail = "Uses CSI-Bench's supplied train_id.json, val_id.json, and test_id.json lists."
        rows.append(
            f"<tr><th>{_escape(group['strategy'])}</th>"
            f"<td>{_escape(', '.join(group['models']))}</td>"
            f"<td>{_escape(count_text)}</td><td>{split_detail}</td></tr>"
        )

    enetfall_count = len(models[0]["evaluations"]["ENetFall"]["y_true"])
    wifall_evaluation = models[0]["evaluations"]["WiFall"]
    wifall_count = len(wifall_evaluation["y_true"])
    wifall_groups = wifall_evaluation.get("group_ids") or []
    participant_count = len(set(wifall_groups))
    return (
        "<section><h2>What data splits mean</h2>"
        "<p>Training-split counts below refer to CSI-Bench samples used to fit and select each model. "
        "Training uses the train split; validation supports model selection/monitoring; CSI-Bench test split is held out.</p>"
        "<table><thead><tr><th>Training split</th><th>Models</th><th>Sample counts</th>"
        f"<th>Assignment</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "<h3>Cross-dataset evaluation scope</h3>"
        f"<p>Cross-dataset reports score all {enetfall_count} ENetFall events and all {wifall_count} WiFall recordings"
        + (f" from {participant_count} participants" if participant_count else "")
        + ", including each source dataset's original train, validation, and test partitions. "
        "No ENetFall or WiFall records were used to fit or tune these models; labels were used only for evaluation metrics.</p></section>"
    )


def _architecture(model: dict[str, Any], diagrams: dict[str, str]) -> str:
    config = model["training"].get("config", {})
    if model["family"] == "random_forest":
        params = {
            key: config.get(key)
            for key in (
                "n_estimators",
                "max_features",
                "min_samples_leaf",
                "class_weight",
            )
        }
        return (
            "<p>Non-neural baseline: Random Forest over 192 robust-normalized temporal features. "
            "Feature vector summarizes 4 time windows × 16 CSI bands × mean, standard deviation, "
            "and adjacent-difference standard deviation. "
            f"Parameters: <code>{_escape(json.dumps(params, sort_keys=True))}</code>.</p>"
        )
    if model["family"] == "cnn_lstm":
        return (
            "<p>Residual CNN encoder → frequency-axis pooling → 32-step sequence → one-layer, "
            f"{_escape(config.get('lstm_hidden_size', 64))}-unit LSTM → two logits. "
            f"Dropout config: <code>{_escape(json.dumps(config.get('dropout', {}), sort_keys=True))}</code>.</p>"
            f"<div class='diagram'>{diagrams['cnn_lstm']}</div>"
        )
    return (
        "<p>ResNet-18-style CSI image classifier.</p>"
        f"<div class='diagram'>{diagrams['cnn']}</div>"
    )


def render_report(models: list[dict[str, Any]], output_path: str | Path) -> str:
    """Render executive ranking, per-dataset metrics, matrices, and architectures."""
    if not models:
        raise ValueError("At least one cross-dataset model result is required")
    winner = max(models, key=average_balanced_accuracy)
    winner_score = average_balanced_accuracy(winner)
    diagrams = {
        "cnn": (ROOT / "docs/csi-bench-cnn-architecture.svg").read_text(
            encoding="utf-8"
        ),
        "cnn_lstm": (
            ROOT / "docs/csi-bench-cnn-lstm-user-v3-architecture.svg"
        ).read_text(encoding="utf-8"),
    }

    ranking_rows = []
    for model in sorted(models, key=average_balanced_accuracy, reverse=True):
        scores = [
            model["evaluations"][dataset]["metrics"]["balanced_accuracy"]
            for dataset in DATASETS
        ]
        ranking_rows.append(
            f"<tr><th>{_escape(model['name'])}</th>"
            + "".join(f"<td>{score:.4f}</td>" for score in scores)
            + f"<td><strong>{average_balanced_accuracy(model):.4f}</strong></td></tr>"
        )

    model_sections = []
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
    output_parent = Path(output_path).expanduser().resolve().parent
    for model in models:
        dataset_sections = []
        for dataset in DATASETS:
            evaluation = model["evaluations"][dataset]
            metrics = evaluation["metrics"]
            metric_rows = "".join(
                f"<tr><th>{_escape(name.replace('_', ' ').title())}</th>"
                f"<td>{_metric(metrics.get(name))}</td></tr>"
                for name in metric_names
            )
            dataset_sections.append(
                f"<div class='dataset'><h3>{_escape(dataset)} · {len(evaluation['y_true'])} samples</h3>"
                f"<table class='metrics'>{metric_rows}</table><h4>Confusion matrix</h4>"
                f"<p>Rows are true labels; columns are predictions.</p>"
                f"{_confusion_table(evaluation)}</div>"
            )
        config = model["training"].get("config", {})
        split = model["training"].get("split", {}).get("strategy")
        split_name = (
            config.get("split_strategy") or split or "official CSI-Bench splits"
        )
        report_path = model["evaluation_dir"] / "report.html"
        report_href = Path(os.path.relpath(report_path, output_parent)).as_posix()
        model_sections.append(
            f"<section><h2>{_escape(model['name'])}</h2>"
            f"<p>Training split: {_escape(split_name)} · Mean balanced accuracy: "
            f"<strong>{average_balanced_accuracy(model):.4f}</strong> · "
            f"<a href='{_escape(report_href)}'>Individual cross-dataset report</a></p>"
            f"{''.join(dataset_sections)}<details><summary>Model architecture</summary>"
            f"{_architecture(model, diagrams)}</details></section>"
        )

    ranking_header = "".join(
        f"<th>{_escape(dataset)} balanced accuracy</th>" for dataset in DATASETS
    )
    winner_scores = winner["evaluations"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CSI-Bench cross-dataset comparison</title>
<style>
body{{font:16px system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem;color:#202124;background:#f8f9fa}}
section,.summary{{margin:1.5rem 0;padding:1.25rem;background:#fff;border:1px solid #d4dbe4;border-radius:10px}}
h1{{margin-bottom:.3rem}}h2{{border-bottom:1px solid #d4dbe4;padding-bottom:.4rem}}h3{{margin:.7rem 0 .4rem}}
table{{border-collapse:collapse;margin:.6rem 0 1rem}}th,td{{padding:.45rem .65rem;border:1px solid #d4dbe4;text-align:left}}th{{background:#eef2f7}}
.dataset{{display:inline-block;vertical-align:top;width:min(100%,580px);margin-right:1.2rem}}
.confusion td,.confusion th{{text-align:center}}.diagram{{overflow-x:auto;margin:.8rem 0}}.diagram svg{{max-width:100%;height:auto}}
details summary{{cursor:pointer;font-weight:650}}.note{{color:#526477}}.winner{{border-left:5px solid #188038}}
a{{color:#1769aa}}
</style></head><body>
<h1>CSI-Bench cross-dataset results</h1>
<p class="note">Zero-shot evaluation on full ENetFall and WiFall datasets. Rows in confusion matrices are true labels; columns are predictions.</p>
<section class="summary winner"><h2>Executive summary</h2>
<p><strong>Best performing model: {_escape(winner["name"])}</strong>, with equal-weight mean balanced accuracy of <strong>{winner_score:.1%}</strong> across ENetFall and WiFall.</p>
<p>Balanced accuracy is averaged equally across datasets. Winner scores: ENetFall {winner_scores["ENetFall"]["metrics"]["balanced_accuracy"]:.1%}; WiFall {winner_scores["WiFall"]["metrics"]["balanced_accuracy"]:.1%}.</p>
<table><thead><tr><th>Model</th>{ranking_header}<th>Mean balanced accuracy</th></tr></thead><tbody>{"".join(ranking_rows)}</tbody></table>
</section>
{_split_section(models)}
<section><h2>Results and model architecture</h2>{"".join(model_sections)}</section>
<section><h2>Interpretation limits</h2><p>All source recordings/events are included, including original train, validation, and test partitions. Cross-dataset inputs use index interpolation to CSI-Bench's 500 × 232 representation; this is approximate channel alignment. Treat ranking as exploratory, not calibrated deployment performance.</p></section>
</body></html>"""


def generate_report(
    output_path: str
    | Path = "data/processed/csi-bench/cross-dataset-comparison/report.html",
    sources: tuple = MODEL_SOURCES,
) -> Path:
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    html_report = render_report(load_models(sources), destination)
    destination.write_text(html_report, encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("data/processed/csi-bench/cross-dataset-comparison/report.html"),
    )
    args = parser.parse_args()
    print(f"Wrote combined report: {generate_report(args.output)}")


if __name__ == "__main__":
    main()
