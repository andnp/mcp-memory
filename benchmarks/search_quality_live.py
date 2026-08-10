"""Run the opt-in search-quality corpus against the live daemon."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

try:
    from benchmarks.search_quality.runner import LiveSearchQualityReport, run_live_daemon
except ModuleNotFoundError as exc:
    if exc.name != "benchmarks":
        raise
    from search_quality.runner import LiveSearchQualityReport, run_live_daemon


_FAILURE_STATUSES = frozenset(
    {"unavailable", "timeout", "malformed_response", "daemon_error", "error"}
)


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
    args = parser.parse_args(argv)
    report = run_live_daemon(
        workspace_root=args.workspace_root,
        timeout_seconds=args.timeout,
    )
    print(json.dumps(report.to_mapping(), indent=2, sort_keys=True))
    return 1 if _has_failures(report) else 0


if __name__ == "__main__":
    raise SystemExit(main())
