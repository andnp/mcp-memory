"""Deterministic searchkernel workload backed by SQLite FTS5.

This workload keeps the application-owned relational repository and search
policy in the loop while measuring warmed keyword retrieval against the real
SQLite FTS5 index. It intentionally omits model-backed vector retrieval so
storage/index costs can be isolated from embedding-provider costs.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from benchmarks.searchkernel_ingest_search import (
    _BENCHMARK_QUERIES,
    SearchObservation,
    _summarize_latency,
    sample_memories,
    scaled_memories,
)
from mcp_memory.config import Config, SearchRankingConfig
from mcp_memory.core.ports.memory import MemoryCreateRequest, MemoryRecord
from mcp_memory.integrations.searchkernel_record_pipeline import (
    build_memory_record_pipeline,
)
from mcp_memory.relational.repository import SQLiteRelationalMemoryRepository
from mcp_memory.utils.db import DatabaseManager


@dataclass
class SQLiteSearchHarness:
    """Reusable search harness using the authoritative SQLite read path."""

    database_path: Path
    records: Sequence[MemoryRecord]

    def __post_init__(self) -> None:
        self.database = DatabaseManager(self.database_path)
        self.repository = SQLiteRelationalMemoryRepository(self.database)
        self.repository.create_memories(
            [
                MemoryCreateRequest(
                    title=record.title,
                    content=record.content,
                    workspace_ids=record.workspace_ids,
                    tags=record.tags,
                    summary=record.summary,
                    memory_type=record.type,
                    status=record.status,
                    memory_id=record.id,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                )
                for record in self.records
            ]
        )
        if "auth-current" in {record.id for record in self.records}:
            self.repository.add_link(
                "auth-current",
                "auth-legacy",
                "SUPERSEDES",
            )
        self.pipeline = build_memory_record_pipeline(
            self.repository,
            config=Config(
                search_ranking=SearchRankingConfig(
                    semantic_only_abstain_threshold=0.0,
                )
            ),
        )

    async def search(
        self,
        query: str,
        *,
        workspace_id: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
        limit: int = 5,
    ) -> SearchObservation:
        started = perf_counter()
        outcome = await self.pipeline.search(
            query,
            limit=limit,
            filters={
                "workspace_id": workspace_id,
                "status": status,
                "include_superseded": include_superseded,
            },
        )
        return SearchObservation(
            result_ids=tuple(result.record_id for result in outcome.results),
            elapsed_ms=(perf_counter() - started) * 1000.0,
            cache_diagnostics=outcome.cache_diagnostics,
            diagnostics=outcome.diagnostics,
            candidate_observations=(),
        )

    def close(self) -> None:
        self.database.close()


async def _run_benchmark_size(
    database_path: Path,
    *,
    record_count: int,
    warmups: int,
    repetitions: int,
    profiler: cProfile.Profile | None,
) -> dict[str, object]:
    started = perf_counter()
    harness = SQLiteSearchHarness(database_path, scaled_memories(record_count))
    setup_ms = (perf_counter() - started) * 1000.0
    try:
        cold_observations = {
            (query, workspace): await harness.search(
                query,
                workspace_id=workspace,
            )
            for query, workspace in _BENCHMARK_QUERIES
        }
        cold_query, cold_workspace = _BENCHMARK_QUERIES[0]
        cold_observation = cold_observations[(cold_query, cold_workspace)]
        for _ in range(warmups):
            for query, workspace in _BENCHMARK_QUERIES:
                await harness.search(query, workspace_id=workspace)

        samples_by_case: dict[tuple[str, str], list[float]] = {
            case: [] for case in _BENCHMARK_QUERIES
        }
        if profiler is not None:
            profiler.enable()
        try:
            for _ in range(repetitions):
                for query, workspace in _BENCHMARK_QUERIES:
                    observation = await harness.search(
                        query,
                        workspace_id=workspace,
                    )
                    samples_by_case[(query, workspace)].append(
                        observation.elapsed_ms
                    )
        finally:
            if profiler is not None:
                profiler.disable()

        return {
            "record_count": record_count,
            "setup": {
                "backend": "sqlite-fts5",
                "records_inserted": len(harness.records),
            },
            "cold": {
                "setup_ms": round(setup_ms, 3),
                "first_search_ms": round(cold_observation.elapsed_ms, 3),
            },
            "warm": {
                "repetitions": repetitions,
                "warmups": warmups,
                "overall": _summarize_latency(
                    [
                        sample
                        for samples in samples_by_case.values()
                        for sample in samples
                    ]
                ).to_mapping(),
                "by_case": {
                    f"{query}:{workspace}": _summarize_latency(
                        samples_by_case[(query, workspace)]
                    ).to_mapping()
                    for query, workspace in _BENCHMARK_QUERIES
                },
            },
            "search": {
                f"{query}:{workspace}": {
                    "result_ids": cold_observations[(query, workspace)].result_ids,
                    "elapsed_ms": round(
                        cold_observations[(query, workspace)].elapsed_ms,
                        3,
                    ),
                    "cache_diagnostics": cold_observations[
                        (query, workspace)
                    ].cache_diagnostics,
                    "diagnostics": cold_observations[
                        (query, workspace)
                    ].diagnostics,
                }
                for query, workspace in _BENCHMARK_QUERIES
            },
        }
    finally:
        harness.close()


async def _run_benchmark(
    database_path: Path,
    *,
    record_counts: Sequence[int] = (len(sample_memories()),),
    warmups: int = 3,
    repetitions: int = 10,
    profile_path: Path | None = None,
) -> dict[str, object]:
    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    profiler = cProfile.Profile() if profile_path is not None else None
    benchmarks = []
    for record_count in record_counts:
        generated_database_path = database_path.parent / (
            f"{database_path.stem}-{record_count}-{uuid4().hex}.db"
        )
        try:
            benchmarks.append(
                await _run_benchmark_size(
                    generated_database_path,
                    record_count=record_count,
                    warmups=warmups,
                    repetitions=repetitions,
                    profiler=profiler,
                )
            )
        finally:
            for suffix in ("", "-wal", "-shm"):
                generated_database_path.with_name(
                    generated_database_path.name + suffix
                ).unlink(missing_ok=True)
    if profiler is not None and profile_path is not None:
        profiler.dump_stats(profile_path)
    return {"benchmarks": benchmarks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="SQLite database prefix (defaults to a temporary directory).",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[len(sample_memories())],
        help="Corpus sizes to measure (default: 7).",
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    if args.database is None:
        with tempfile.TemporaryDirectory(prefix="searchkernel-sqlite-") as directory:
            result = asyncio.run(
                _run_benchmark(
                    Path(directory) / "memory.db",
                    record_counts=args.sizes,
                    warmups=args.warmups,
                    repetitions=args.repetitions,
                    profile_path=args.profile,
                )
            )
    else:
        result = asyncio.run(
            _run_benchmark(
                args.database,
                record_counts=args.sizes,
                warmups=args.warmups,
                repetitions=args.repetitions,
                profile_path=args.profile,
            )
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=list))


if __name__ == "__main__":
    main()
