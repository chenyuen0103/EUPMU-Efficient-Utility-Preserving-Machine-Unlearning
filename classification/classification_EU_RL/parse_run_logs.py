#!/usr/bin/env python3
"""Batch-parse run.log files into evaluation_result.json and training_log.json.

This helper walks one or more output roots, finds every run.log, and uses
parse_run_log.py to emit the JSON artifacts alongside the log.

It also writes train_log.json as a compatibility alias.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from parse_run_log import parse_run_log, write_json


def expand_roots(raw_roots: Iterable[Path]) -> List[Path]:
    roots: List[Path] = []
    for root in raw_roots:
        root = root.expanduser().resolve()
        if root.exists():
            roots.append(root)
            continue
        # Help with the common typo in this workspace.
        if root.name == "omd_tch_gd":
            alt = root.with_name("omd_tch_pgd")
            if alt.exists():
                roots.append(alt)
                continue
    return roots


def iter_run_logs(root: Path):
    yield from sorted(root.rglob("run.log"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="One or more roots to scan for run.log files.",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help=(
            "Only parse runs that are missing evaluation_result.json or "
            "training_log.json."
        ),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    roots = expand_roots(args.roots)
    if not roots:
        raise SystemExit("No valid roots found.")

    n_logs = 0
    n_skipped = 0
    for root in roots:
        for run_log in iter_run_logs(root):
            out_dir = run_log.parent
            evaluation_path = out_dir / "evaluation_result.json"
            training_path = out_dir / "training_log.json"
            if args.only_missing and evaluation_path.exists() and training_path.exists():
                n_skipped += 1
                continue

            evaluation_result, training_log = parse_run_log(run_log)
            write_json(evaluation_path, evaluation_result)
            write_json(training_path, training_log)
            write_json(out_dir / "train_log.json", training_log)
            n_logs += 1
            print(f"Parsed: {run_log}")

    if args.only_missing:
        print(
            f"Completed. Parsed {n_logs} run.log files; skipped {n_skipped} "
            "already-complete runs."
        )
    else:
        print(f"Completed. Parsed {n_logs} run.log files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())