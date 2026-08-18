"""Sweep calibration thresholds against the pinned replay snapshot."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from .artifact import load_labels
from .labels import RelevanceLabel
from .replay import DEFAULT_LIMIT, SearchFn, replay_labels
from .report import compare_variants
from .run import DEFAULT_LABELS, SNAPSHOT_DSN, build_search, snapshot_config
from .variants import threshold_variant

DEFAULT_OUTPUT = Path("benchmarks/replay_eval/artifacts/threshold-sweep.json")


def sweep_thresholds(
    labels: Sequence[RelevanceLabel],
    *,
    baseline_threshold: float,
    challenger_thresholds: Sequence[float],
    search: SearchFn,
    limit: int = DEFAULT_LIMIT,
    seed: int,
) -> dict[str, object]:
    """Replay a baseline threshold and its challengers over the same labels."""
    variants = [threshold_variant(value) for value in (baseline_threshold, *challenger_thresholds)]
    replays = {
        variant.name: replay_labels(labels, variant, search, limit=limit)
        for variant in variants
    }
    return compare_variants(replays, baseline=variants[0].name, seed=seed).to_mapping()


def main(argv: list[str] | None = None) -> int:
    """Run a calibration-threshold sweep from the command line and write its report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-threshold", type=float, required=True)
    parser.add_argument(
        "--challenger-threshold",
        type=float,
        action="append",
        required=True,
        dest="challenger_thresholds",
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--dsn", default=SNAPSHOT_DSN)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    loaded = load_labels(args.labels)
    with TemporaryDirectory() as directory:
        search = build_search(snapshot_config(args.dsn), Path(directory))
        payload = sweep_thresholds(
            loaded.labels,
            baseline_threshold=args.baseline_threshold,
            challenger_thresholds=args.challenger_thresholds,
            search=search,
            limit=args.limit,
            seed=args.seed,
        )
    payload["duplicates_collapsed"] = loaded.duplicates_collapsed

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
