"""Command-line interface for generic evaluation reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from .report import render_html_report_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a generic evaluation result as HTML")
    parser.add_argument("result", type=Path, help="evaluation result JSON")
    parser.add_argument("output", type=Path, help="static HTML output")
    args = parser.parse_args(argv)
    destination = render_html_report_file(args.result, args.output)
    print(f"Wrote HTML report: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
