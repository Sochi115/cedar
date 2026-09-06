"""Command-line interface for validating and splitting WiFall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .loader import (
    SplitConfig,
    generate_split_manifest,
    save_split_manifest,
    summarize_dataset,
)


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "summarize":
        summary = summarize_dataset(args.dataset_root)
        print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
        return 0

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


if __name__ == "__main__":
    raise SystemExit(main())
