"""Command-line interface for validating and splitting WiFall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ensemble import train_ensemble
from .improved_svm import train_improved_svm
from .loader import (
    SplitConfig,
    generate_split_manifest,
    save_split_manifest,
    summarize_dataset,
)
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

    random_forest = subparsers.add_parser(
        "train-random-forest", help="train and evaluate the Random Forest"
    )
    random_forest.add_argument("dataset_root", type=Path)
    random_forest.add_argument("split_manifest", type=Path)
    random_forest.add_argument("output_dir", type=Path)

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
        run = train_improved_svm(args.dataset_root, args.split_manifest, args.output_dir)
        print(f"Wrote model: {run.model_path}")
        print(f"Wrote audit: {run.audit_path}")
        print(f"Wrote run artifact: {run.artifact_path}")
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

    run = train_random_forest(args.dataset_root, args.split_manifest, args.output_dir)
    print(f"Wrote model: {run.model_path}")
    print(f"Wrote audit: {run.audit_path}")
    print(f"Wrote run artifact: {run.artifact_path}")
    print(f"Wrote report: {run.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
