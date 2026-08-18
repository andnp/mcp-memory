"""Write telemetry-derived relevance labels to a local artifact."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import psycopg

from mcp_memory.config import load_config

from .extract import DEFAULT_WINDOW_SECONDS, extract_labels

DEFAULT_OUTPUT = Path("benchmarks/replay_eval/artifacts/labels.jsonl")
DEFAULT_SINCE_DAYS = 90


def resolve_dsn() -> str:
    """Read the live store's DSN from configuration.

    The DSN carries a credential, so it is resolved at run time from the user's
    own configuration and never stored alongside the harness.
    """
    dsn = load_config().storage.postgres.dsn
    if not dsn:
        raise SystemExit("no postgres dsn configured; replay labels need the live store")
    return dsn


def main(argv: list[str] | None = None) -> int:
    """Extract labels and write them as JSON lines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since-days", type=float, default=DEFAULT_SINCE_DAYS)
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    since = time.time() - args.since_days * 86400.0
    with psycopg.connect(resolve_dsn()) as connection, connection.cursor() as cursor:
        result = extract_labels(
            cursor, since=since, window_seconds=args.window_seconds
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for label in result.labels:
            stream.write(
                json.dumps(
                    {
                        "query": label.query,
                        "workspace_id": label.workspace_id,
                        "memory_id": label.memory_id,
                        "logged_rank": label.logged_rank,
                        "observed_at": label.observed_at,
                        "dwell_ms": label.dwell_ms,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    queries = len({label.query for label in result.labels})
    print(
        f"wrote {len(result.labels)} labels ({queries} distinct queries) to {args.output}; "
        f"skipped {result.skipped} of {result.considered} attributed rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
