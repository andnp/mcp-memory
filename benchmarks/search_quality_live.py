"""Run the opt-in search-quality corpus against the live daemon."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

try:
    from benchmarks.search_quality.runner import (
        LiveSearchQualityReport,
        ResultLabeler,
        run_live_daemon,
    )
except ModuleNotFoundError as exc:
    if exc.name != "benchmarks":
        raise
    from search_quality.runner import LiveSearchQualityReport, ResultLabeler, run_live_daemon


_FAILURE_STATUSES = frozenset(
    {"unavailable", "timeout", "malformed_response", "daemon_error", "error"}
)
_UNSCORED_EXIT_CODE = 2


def _label_map(value: str) -> dict[str, str]:
    try:
        with Path(value).open(encoding="utf-8") as stream:
            mapping = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise argparse.ArgumentTypeError(f"invalid label map: {exc}") from exc
    if not isinstance(mapping, dict):
        raise argparse.ArgumentTypeError("invalid label map: expected a JSON object")
    if any(
        not isinstance(memory_id, str)
        or not memory_id
        or not isinstance(label, str)
        or not label
        for memory_id, label in mapping.items()
    ):
        raise argparse.ArgumentTypeError(
            "invalid label map: keys and values must be non-empty strings"
        )
    return mapping


def _memory_id_labeler(label_map: Mapping[str, str]) -> ResultLabeler:
    def label_result(result: dict[str, object]) -> str | None:
        memory_id = result.get("memory_id")
        return label_map.get(memory_id) if isinstance(memory_id, str) else None

    return label_result


def _has_failures(report: LiveSearchQualityReport) -> bool:
    return any(case.status in _FAILURE_STATUSES for case in report.cases)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the live benchmark and print only its privacy-safe report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request daemon timeout in seconds (default: 60.0).",
    )
    parser.add_argument(
        "--workspace-root",
        help="Workspace root to use when querying the daemon.",
    )
    parser.add_argument(
        "--label-map",
        type=_label_map,
        help="Local JSON file mapping returned memory_id values to evaluation labels.",
    )
    args = parser.parse_args(argv)
    if args.label_map is not None:
        report = run_live_daemon(
            workspace_root=args.workspace_root,
            timeout_seconds=args.timeout,
            result_labeler=_memory_id_labeler(args.label_map),
        )
    else:
        report = run_live_daemon(
            workspace_root=args.workspace_root,
            timeout_seconds=args.timeout,
        )
    print(json.dumps(report.to_mapping(), indent=2, sort_keys=True))
    if _has_failures(report):
        return 1
    if not any(case.scored for case in report.cases):
        return _UNSCORED_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
