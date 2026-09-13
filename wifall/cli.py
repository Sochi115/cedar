"""Command-line interface for validating and splitting WiFall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cnn import train_cnn
from .cnn_analysis import analyze_cnn_runs
from .ensemble import train_ensemble
from .improved_svm import train_improved_svm
from .loader import (
    SplitConfig,
    generate_split_manifest,
    save_split_manifest,
    summarize_dataset,
)
from .lstm import LstmConfig, LstmFeatureConfig, train_lstm
from .lstm_analysis import analyze_lstm_runs
from .mlp import train_mlp
from .processed import DurationNormalizedConfig, build_duration_normalized_dataset
from .random_forest import train_random_forest
from .svm import train_svm


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and split a WiFall dataset")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser("summarize", help="validate and summarize recordings")
    summarize.add_argument("dataset_root", type=Path)

    split = subparsers.add_parser("split", help="create a participant-held-out manifest")
    split.add_argument("dataset_root", type=Path)
    split.add_argument("output", type=Path)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--test-participants", type=int, default=2)
    split.add_argument("--validation-participants", type=int, default=1)

    train = subparsers.add_parser("train-svm", help="train and evaluate the SVM baseline")
    train.add_argument("dataset_root", type=Path)
    train.add_argument("split_manifest", type=Path)
    train.add_argument("output_dir", type=Path)

    improved = subparsers.add_parser(
        "train-improved-svm", help="train and evaluate the improved SVM"
    )
    improved.add_argument("dataset_root", type=Path)
    improved.add_argument("split_manifest", type=Path)
    improved.add_argument("output_dir", type=Path)
    improved.add_argument("--processed-dataset", type=Path)

    mlp = subparsers.add_parser("train-mlp", help="train and evaluate a small MLP")
    mlp.add_argument("dataset_root", type=Path)
    mlp.add_argument("split_manifest", type=Path)
    mlp.add_argument("output_dir", type=Path)

    lstm = subparsers.add_parser(
        "train-lstm", help="train and evaluate an LSTM on duration-normalized CSI"
    )
    lstm.add_argument("dataset_root", type=Path)
    lstm.add_argument("split_manifest", type=Path)
    lstm.add_argument("output_dir", type=Path)
    lstm.add_argument("--processed-dataset", type=Path, required=True)
    lstm.add_argument("--hidden-size", type=int, default=32)
    lstm.add_argument("--layer-count", type=int, default=1)
    lstm.add_argument("--dropout", type=float, default=0.1)
    lstm.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    lstm.add_argument("--epochs", type=int, default=20)
    lstm.add_argument("--batch-size", type=int, default=32)
    lstm.add_argument("--learning-rate", type=float, default=1e-3)
    lstm.add_argument("--weight-decay", type=float, default=1e-4)
    lstm.add_argument("--early-stopping-patience", type=int, default=5)
    lstm.add_argument("--scheduler-patience", type=int, default=2)
    lstm.add_argument("--scheduler-factor", type=float, default=0.5)
    lstm.add_argument("--min-epochs", type=int, default=1)
    lstm.add_argument("--causal-window", type=int, default=32)
    lstm.add_argument("--window-stride", type=int, default=4)
    lstm.add_argument("--minimum-recall", type=float, default=0.95)
    lstm.add_argument("--seed", type=int, default=42)

    preprocess = subparsers.add_parser(
        "preprocess-duration-normalized",
        help="persist duration-normalized, sectioned CSI data",
    )
    preprocess.add_argument("dataset_root", type=Path)
    preprocess.add_argument("split_manifest", type=Path)
    preprocess.add_argument("output_dir", type=Path)
    preprocess.add_argument("--sample-count", type=int, default=128)
    preprocess.add_argument("--section-count", type=int, default=8)

    cnn = subparsers.add_parser("train-cnn", help="train and evaluate a small CNN")
    cnn.add_argument("dataset_root", type=Path)
    cnn.add_argument("split_manifest", type=Path)
    cnn.add_argument("output_dir", type=Path)
    cnn.add_argument("--processed-dataset", type=Path)

    lstm_analysis = subparsers.add_parser(
        "analyze-lstm", help="analyze LSTM errors and model complementarity"
    )
    lstm_analysis.add_argument("output_dir", type=Path)
    lstm_analysis.add_argument("lstm_run", type=Path)
    lstm_analysis.add_argument("cnn_run", type=Path)
    lstm_analysis.add_argument("improved_svm_run", type=Path)
    lstm_analysis.add_argument("random_forest_run", type=Path)

    cnn_analysis = subparsers.add_parser(
        "analyze-cnn", help="analyze CNN errors and model complementarity"
    )
    cnn_analysis.add_argument("output_dir", type=Path)
    cnn_analysis.add_argument("cnn_run", type=Path)
    cnn_analysis.add_argument("improved_svm_run", type=Path)
    cnn_analysis.add_argument("random_forest_run", type=Path)
    cnn_analysis.add_argument("ensemble_run", type=Path)

    random_forest = subparsers.add_parser(
        "train-random-forest", help="train and evaluate the Random Forest"
    )
    random_forest.add_argument("dataset_root", type=Path)
    random_forest.add_argument("split_manifest", type=Path)
    random_forest.add_argument("output_dir", type=Path)
    random_forest.add_argument("--processed-dataset", type=Path)

    ensemble = subparsers.add_parser(
        "train-ensemble", help="blend improved SVM and Random Forest scores"
    )
    ensemble.add_argument("dataset_root", type=Path)
    ensemble.add_argument("split_manifest", type=Path)
    ensemble.add_argument("improved_svm_run", type=Path)
    ensemble.add_argument("random_forest_run", type=Path)
    ensemble.add_argument("output_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "preprocess-duration-normalized":
        run = build_duration_normalized_dataset(
            args.dataset_root,
            args.split_manifest,
            args.output_dir,
            DurationNormalizedConfig(
                sample_count=args.sample_count,
                section_count=args.section_count,
            ),
        )
        print(f"Wrote processed CSI: {run.data_path}")
        print(f"Wrote metadata: {run.metadata_path}")
        print(json.dumps(run.counts, indent=2, sort_keys=True))
        return 0

    if args.command == "summarize":
        summary = summarize_dataset(args.dataset_root)
        print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "split":
        config = SplitConfig(
            seed=args.seed,
            test_participants=args.test_participants,
            validation_participants=args.validation_participants,
        )
        manifest = generate_split_manifest(args.dataset_root, config)
        destination = save_split_manifest(manifest, args.output)
        print(f"Wrote split manifest: {destination}")
        print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
        return 0

    if args.command == "train-svm":
        run = train_svm(args.dataset_root, args.split_manifest, args.output_dir)
        print(f"Wrote model: {run.model_path}")
        print(f"Wrote run artifact: {run.artifact_path}")
        return 0

    if args.command == "train-improved-svm":
        run = train_improved_svm(
            args.dataset_root,
            args.split_manifest,
            args.output_dir,
            processed_dataset=args.processed_dataset,
        )
        print(f"Wrote model: {run.model_path}")
        print(f"Wrote audit: {run.audit_path}")
        print(f"Wrote run artifact: {run.artifact_path}")
        print(f"Wrote report: {run.report_path}")
        return 0

    if args.command == "train-mlp":
        run = train_mlp(args.dataset_root, args.split_manifest, args.output_dir)
        print(f"Wrote model: {run.model_path}")
        print(f"Wrote audit: {run.audit_path}")
        print(f"Wrote run artifact: {run.artifact_path}")
        print(f"Wrote report: {run.report_path}")
        return 0

    if args.command == "train-lstm":
        run = train_lstm(
            args.dataset_root,
            args.split_manifest,
            args.output_dir,
            LstmConfig(
                hidden_sizes=(args.hidden_size,),
                layer_counts=(args.layer_count,),
                dropout_values=(args.dropout,),
                optimizer_names=(args.optimizer,),
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                early_stopping_patience=args.early_stopping_patience,
                scheduler_patience=args.scheduler_patience,
                scheduler_factor=args.scheduler_factor,
                min_epochs=args.min_epochs,
                minimum_recall=args.minimum_recall,
                feature=LstmFeatureConfig(
                    causal_window=args.causal_window,
                    window_stride=args.window_stride,
                ),
                random_state=args.seed,
            ),
            processed_dataset=args.processed_dataset,
        )
        print(f"Wrote model: {run.model_path}")
        print(f"Wrote audit: {run.audit_path}")
        print(f"Wrote run artifact: {run.artifact_path}")
        print(f"Wrote report: {run.report_path}")
        return 0

    if args.command == "analyze-lstm":
        run = analyze_lstm_runs(
            args.lstm_run,
            (args.cnn_run, args.improved_svm_run, args.random_forest_run),
            args.output_dir,
        )
        print(f"Wrote analysis artifact: {run.artifact_path}")
        print(f"Wrote report: {run.report_path}")
        return 0

    if args.command == "train-cnn":
        run = train_cnn(
            args.dataset_root,
            args.split_manifest,
            args.output_dir,
            processed_dataset=args.processed_dataset,
        )
        print(f"Wrote model: {run.model_path}")
        print(f"Wrote audit: {run.audit_path}")
        print(f"Wrote run artifact: {run.artifact_path}")
        print(f"Wrote report: {run.report_path}")
        return 0

    if args.command == "analyze-cnn":
        run = analyze_cnn_runs(
            args.cnn_run,
            (args.improved_svm_run, args.random_forest_run, args.ensemble_run),
            args.output_dir,
        )
        print(f"Wrote analysis artifact: {run.artifact_path}")
        print(f"Wrote report: {run.report_path}")
        return 0

    if args.command == "train-ensemble":
        run = train_ensemble(
            args.dataset_root,
            args.split_manifest,
            args.improved_svm_run,
            args.random_forest_run,
            args.output_dir,
        )
        print(f"Wrote run artifact: {run.artifact_path}")
        print(f"Wrote report: {run.report_path}")
        return 0

    run = train_random_forest(
        args.dataset_root,
        args.split_manifest,
        args.output_dir,
        processed_dataset=args.processed_dataset,
    )
    print(f"Wrote model: {run.model_path}")
    print(f"Wrote audit: {run.audit_path}")
    print(f"Wrote run artifact: {run.artifact_path}")
    print(f"Wrote report: {run.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
