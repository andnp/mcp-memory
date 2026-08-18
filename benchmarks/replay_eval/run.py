"""Replay labeled queries against the pinned snapshot and report the result."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from mcp_memory.config import Config, load_config
from mcp_memory.embeddings import build_embedder
from mcp_memory.storage.postgres import build_postgres_runtime_components
from mcp_memory.storage.types import StorageBootstrapSpec

from .labels import RelevanceLabel
from .replay import DEFAULT_LIMIT, SearchFn, replay_labels
from .report import compare_variants
from .variants import Variant, default_variants

DEFAULT_LABELS = Path("benchmarks/replay_eval/artifacts/labels.jsonl")
DEFAULT_FINGERPRINT = Path("benchmarks/replay_eval/artifacts/corpus-snapshot.json")
DEFAULT_OUTPUT = Path("benchmarks/replay_eval/artifacts/comparison.json")
SNAPSHOT_DSN = "postgresql://replay:replay@127.0.0.1:5455/mcp_memory"


def load_labels(path: Path) -> list[RelevanceLabel]:
    """Read labels from the extractor's artifact."""
    labels: list[RelevanceLabel] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                labels.append(RelevanceLabel(**json.loads(line)))
    return labels


def sample_by_query(
    labels: Sequence[RelevanceLabel], *, queries: int, seed: int
) -> list[RelevanceLabel]:
    """Take every label belonging to a random subset of queries.

    Sampling whole queries rather than labels keeps each sampled query's
    evidence intact, which is the unit the comparison weights by.
    """
    keys = sorted({(label.query, label.workspace_id) for label in labels})
    if queries >= len(keys):
        return list(labels)
    chosen = set(random.Random(seed).sample(keys, queries))
    return [
        label for label in labels if (label.query, label.workspace_id) in chosen
    ]


def snapshot_config(dsn: str) -> Config:
    """Point a copy of the live configuration at the pinned snapshot."""
    config = load_config()
    postgres = replace(config.storage.postgres, dsn=dsn)
    storage = replace(config.storage, backend="postgres", postgres=postgres)
    return replace(config, storage=storage)


def build_search(config: Config, memory_path: Path) -> SearchFn:
    """Compose the production search path against the snapshot."""
    spec = StorageBootstrapSpec(
        memory_path=memory_path, config=config, workspace_id=None
    )
    resources = build_postgres_runtime_components(
        spec,
        embedder=build_embedder(config.embeddings),
        enable_background_repair_queue=False,
    )
    service: Any = resources.relational_search
    if service is None:
        raise SystemExit("snapshot produced no search service")

    def search(query: str, workspace_id: str | None, limit: int) -> Sequence[str]:
        results = service.search_memories(
            query,
            workspace_id=workspace_id,
            limit=limit,
            side_effect_free=True,
        )
        return [result.memory_id for result in results]

    return search


def main(argv: list[str] | None = None) -> int:
    """Replay every variant over the same labels and write the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dsn", default=SNAPSHOT_DSN)
    parser.add_argument("--queries", type=int, default=0, help="0 replays every query")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fingerprint", type=Path, default=DEFAULT_FINGERPRINT)
    args = parser.parse_args(argv)

    labels = load_labels(args.labels)
    if args.queries:
        labels = sample_by_query(labels, queries=args.queries, seed=args.seed)

    variants: tuple[Variant, ...] = default_variants()
    with TemporaryDirectory() as directory:
        search = build_search(snapshot_config(args.dsn), Path(directory))
        replays = {}
        for variant in variants:
            print(f"==> replaying {variant.name}", flush=True)
            replays[variant.name] = replay_labels(
                labels,
                variant,
                search,
                limit=args.limit,
                on_progress=lambda done, total: (
                    print(f"    {done}/{total}", flush=True) if done % 100 == 0 else None
                ),
            )

    report = compare_variants(
        replays,
        baseline=variants[0].name,
        controls=tuple(v.name for v in variants if v.is_control),
        seed=args.seed,
    )
    payload = report.to_mapping()
    if args.fingerprint.exists():
        payload["corpus"] = json.loads(args.fingerprint.read_text(encoding="utf-8"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
