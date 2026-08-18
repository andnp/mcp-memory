"""Deterministic searchkernel ingest-to-search harness.

The harness uses the real searchkernel ingestion boundary and mcp-memory's
record-pipeline composition, but keeps storage, embeddings, and data local to
the process.  It is suitable for focused tests and for collecting comparable
candidate/cache/latency observations without Docker, Postgres, or model
downloads.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from searchkernel.domain import Record, RecordHit
from searchkernel.indexing.embedding_cache import SQLiteEmbeddingCache
from searchkernel.ingestion import SemanticRecordIngestor
from searchkernel.ports.content_source import IngestionReceipt

from mcp_memory.config import Config, SearchRankingConfig
from mcp_memory.core.ports.memory import (
    MemoryLink,
    MemoryReadContext,
    MemoryRecord,
    MemoryRepositoryPort,
    RankedMemoryCandidate,
)
from mcp_memory.integrations.searchkernel_adapters import (
    MemoryRecordAdapter,
    MemoryVectorBackend,
)
from mcp_memory.integrations.searchkernel_record_pipeline import (
    build_memory_record_pipeline,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_EMBEDDING_AXES = {
    "authentication": 0,
    "credential": 0,
    "token": 0,
    "database": 1,
    "migration": 1,
    "storage": 1,
    "sqlite": 1,
    "postgres": 1,
    "deployment": 2,
    "daemon": 2,
    "search": 3,
    "retrieval": 3,
    "ranking": 3,
}


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(text.lower()))


def _timestamp(hour: int) -> str:
    return f"2026-08-01T{hour:02d}:00:00+00:00"


def _memory(
    memory_id: str,
    title: str,
    content: str,
    *,
    workspace_ids: list[str],
    status: str = "active",
) -> MemoryRecord:
    return MemoryRecord(
        id=memory_id,
        title=title,
        content=content,
        summary=content,
        type="fact",
        status=status,
        created_at=_timestamp(10),
        updated_at=_timestamp(11),
        read_count=0,
        access_score=0.0,
        last_accessed_at=None,
        last_surfaced_at=None,
        workspace_ids=workspace_ids,
        tags=[],
    )


def sample_memories() -> list[MemoryRecord]:
    """Return the fixed fixture used by the local tests and benchmark."""

    return [
        _memory(
            "auth-current",
            "Authentication policy",
            "Authentication credential rotation and token rules.",
            workspace_ids=["workspace-a"],
        ),
        _memory(
            "auth-legacy",
            "Legacy authentication policy",
            "Legacy authentication credential and token rules.",
            workspace_ids=["workspace-a"],
        ),
        _memory(
            "auth-archived",
            "Archived authentication policy",
            "Archived authentication credential and token rules.",
            workspace_ids=["workspace-a"],
            status="archived",
        ),
        _memory(
            "db-a",
            "Database migration A",
            "SQLite database migration and storage plan for workspace A.",
            workspace_ids=["workspace-a"],
        ),
        _memory(
            "db-b",
            "Database migration B",
            "Postgres database migration and storage plan for workspace B.",
            workspace_ids=["workspace-b"],
        ),
        _memory(
            "db-stale",
            "Stale database migration",
            "Stale database migration and storage plan.",
            workspace_ids=["workspace-a"],
            status="stale",
        ),
        _memory(
            "shared-workspace",
            "Shared workspace deployment",
            "Deployment note intentionally visible in both workspaces.",
            workspace_ids=["workspace-a", "workspace-b"],
        ),
    ]


def scaled_memories(record_count: int) -> list[MemoryRecord]:
    """Return the fixture expanded to a deterministic corpus size."""
    base = sample_memories()
    if record_count < len(base):
        raise ValueError(f"record_count must be at least {len(base)}")
    records = list(base)
    for index in range(len(base), record_count):
        template = base[index % len(base)]
        records.append(
            replace(
                template,
                id=f"{template.id}-{index}",
                title=f"{template.title} {index}",
            )
        )
    return records


class DeterministicEmbeddingProvider:
    """Small topic encoder with observable, repeatable calls."""

    model_name = "parity-harness"
    dim = 4

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in _tokens(text):
                axis = _EMBEDDING_AXES.get(token)
                if axis is not None:
                    vector[axis] += 1.0
            magnitude = math.sqrt(sum(value * value for value in vector))
            vectors.append(
                vector
                if magnitude == 0.0
                else [value / magnitude for value in vector]
            )
        return vectors


class _MemoryRepository:
    """Authoritative in-memory repository implementing the search read port."""

    def __init__(self, records: Sequence[MemoryRecord]) -> None:
        self.records = {record.id: record for record in records}
        self.links = [MemoryLink("auth-current", "auth-legacy", "SUPERSEDES", "")]
        self.keyword_queries: list[tuple[str, str | None, str | None]] = []
        self.ranking_candidate_calls = 0

    def search_keyword_memory_ids(
        self,
        query: str,
        *,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
        limit: int = 50,
    ) -> list[str]:
        self.keyword_queries.append((query, workspace_id, status))
        query_tokens = set(_tokens(query))
        ranked: list[tuple[int, str]] = []
        for record in self.records.values():
            if workspace_id is not None and workspace_id not in record.workspace_ids:
                continue
            if memory_type is not None and memory_type != record.type:
                continue
            if status is not None and status != record.status:
                continue
            if not include_superseded and self._is_superseded(record.id):
                continue
            if status is None and record.status == "archived":
                continue
            searchable = " ".join(
                (record.title, record.content, record.summary or "", *record.tags)
            )
            matched = len(query_tokens.intersection(_tokens(searchable)))
            if matched:
                ranked.append((matched, record.id))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [memory_id for _matched, memory_id in ranked[:limit]]

    def get_ranking_candidates(
        self,
        memory_ids: list[str],
        *,
        status: str | None = None,
        include_superseded: bool = False,
    ) -> list[RankedMemoryCandidate]:
        self.ranking_candidate_calls += 1
        candidates: list[RankedMemoryCandidate] = []
        for memory_id in memory_ids:
            record = self.records.get(memory_id)
            if record is None:
                continue
            incoming = self.get_links(memory_id, direction="incoming")
            counts: dict[str, int] = {}
            for link in incoming:
                counts[link.link_type] = counts.get(link.link_type, 0) + 1
            candidates.append(
                RankedMemoryCandidate(
                    record=record,
                    incoming_links_count=len(incoming),
                    has_incoming_supersedes=counts.get("SUPERSEDES", 0) > 0,
                    incoming_link_type_counts=counts,
                )
            )
        return candidates

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        return self.records.get(memory_id)

    def get_links(
        self,
        memory_id: str,
        direction: str = "outgoing",
        link_type: str | None = None,
    ) -> list[MemoryLink]:
        links = [
            link
            for link in self.links
            if (link.source_id if direction == "outgoing" else link.target_id)
            == memory_id
        ]
        if link_type is not None:
            links = [link for link in links if link.link_type == link_type]
        return links

    def peek_memory(self, memory_id: str) -> MemoryReadContext | None:
        record = self.get_memory(memory_id)
        if record is None:
            return None
        return MemoryReadContext(record, {"outgoing": self.get_links(memory_id)}, [])

    def list_memory_ids(
        self,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[str]:
        return [
            record.id
            for record in self.records.values()
            if (workspace_id is None or workspace_id in record.workspace_ids)
            and (memory_type is None or memory_type == record.type)
            and (status is None or status == record.status)
        ][:limit]

    def _is_superseded(self, memory_id: str) -> bool:
        return bool(self.get_links(memory_id, direction="incoming", link_type="SUPERSEDES"))


@dataclass(frozen=True)
class VectorSearchObservation:
    candidate_ids: tuple[str, ...] | None
    workspace_id: str | None
    limit: int


class _IndexedVectorStore:
    """In-memory vector index usable by both ingestion and memory adapters."""

    supports_candidate_filtering = True

    def __init__(self) -> None:
        self.vectors: dict[tuple[str, str | None], list[float]] = {}
        self.indexed_records: dict[str, Record] = {}
        self.searches: list[VectorSearchObservation] = []

    def index(self, records: list[Record]) -> None:
        self.indexed_records.update({record.storage_key: record for record in records})

    def upsert(
        self,
        records: list[Record] | None = None,
        model_name: str | None = None,
        dim: int | None = None,
        **kwargs: object,
    ) -> bool | None:
        if records is not None:
            for record in records:
                if record.embedding is not None:
                    self.vectors[(record.source_id, record.workspace_id)] = list(
                        record.embedding
                    )
                    self.indexed_records[record.storage_key] = record
            return None
        source_kind = kwargs.get("source_kind")
        if source_kind != "memory":
            raise ValueError(f"unexpected source kind: {source_kind!r}")
        source_id = kwargs.get("source_id")
        workspace_value = kwargs.get("workspace_id")
        workspace_id = workspace_value if isinstance(workspace_value, str) else None
        embedding = kwargs.get("embedding")
        if not isinstance(source_id, str) or not isinstance(embedding, list):
            raise TypeError("memory vector upsert requires source_id and embedding")
        self.vectors[(source_id, workspace_id)] = list(
            cast("list[float]", embedding)
        )
        return True

    def search(
        self,
        *,
        source_kind: str,
        model_name: str,
        query_embedding: list[float],
        candidate_ids: list[str] | None = None,
        diagnostics: dict[str, object] | None = None,
        workspace_id: str | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        del source_kind, model_name, diagnostics
        self.searches.append(
            VectorSearchObservation(
                candidate_ids=None if candidate_ids is None else tuple(candidate_ids),
                workspace_id=workspace_id,
                limit=limit,
            )
        )
        candidates = [
            (record_id, record_workspace, vector)
            for (record_id, record_workspace), vector in self.vectors.items()
            if (workspace_id is None or record_workspace == workspace_id)
            and (candidate_ids is None or record_id in candidate_ids)
        ]
        ranked = [
            (record_id, self._cosine(query_embedding, vector), record_workspace or "")
            for record_id, record_workspace, vector in candidates
        ]
        ranked.sort(key=lambda item: (-item[1], item[0], item[2]))
        return [(record_id, score) for record_id, score, _workspace in ranked[:limit]]

    def delete(self, **kwargs: object) -> int:
        source_id = kwargs.get("source_id")
        if not isinstance(source_id, str):
            return 0
        workspace_value = kwargs.get("workspace_id")
        workspace_id = workspace_value if isinstance(workspace_value, str) else None
        key: tuple[str, str | None] = (source_id, workspace_id)
        return int(self.vectors.pop(key, None) is not None)

    def identity_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.indexed_records))

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


class _KeywordIndex:
    def __init__(self) -> None:
        self.records: dict[str, Record] = {}

    def index(self, records: list[Record]) -> None:
        self.records.update({record.storage_key: record for record in records})

    def search(
        self,
        query: str,
        k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> list[RecordHit]:
        del query, k, filters
        return []


@dataclass(frozen=True)
class SearchObservation:
    result_ids: tuple[str, ...]
    elapsed_ms: float
    cache_diagnostics: tuple[str, ...]
    diagnostics: tuple[str, ...]
    candidate_observations: tuple[VectorSearchObservation, ...]


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Percentile summary for one warmed benchmark population."""

    count: int
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float

    def to_mapping(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
        }


def _percentile(samples: Sequence[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _summarize_latency(samples: Sequence[float]) -> LatencySummary:
    if not samples:
        raise ValueError("latency samples cannot be empty")
    return LatencySummary(
        count=len(samples),
        p50_ms=_percentile(samples, 0.50),
        p95_ms=_percentile(samples, 0.95),
        min_ms=min(samples),
        max_ms=max(samples),
        mean_ms=sum(samples) / len(samples),
    )


@dataclass
class IngestSearchHarness:
    """Reusable local ingest/search harness with structured observations."""

    cache_path: Path
    records: Sequence[MemoryRecord] = field(default_factory=sample_memories)

    def __post_init__(self) -> None:
        self.repository = _MemoryRepository(self.records)
        self.embedder = DeterministicEmbeddingProvider()
        self.vector_store = _IndexedVectorStore()
        self.keyword_store = _KeywordIndex()
        self.embedding_cache = SQLiteEmbeddingCache(
            self.cache_path,
            DeterministicEmbeddingProvider.model_name,
            DeterministicEmbeddingProvider.dim,
        )
        self.ingestor = SemanticRecordIngestor(
            embedding_provider=self.embedder,
            keyword_store=self.keyword_store,
            vector_store=cast(Any, self.vector_store),
            embedding_cache=self.embedding_cache,
        )
        self.pipeline = build_memory_record_pipeline(
            cast("MemoryRepositoryPort", self.repository),
            vector_store=cast("MemoryVectorBackend", self.vector_store),
            embedder=self.embedder,
            config=Config(
                search_ranking=SearchRankingConfig(
                    semantic_only_abstain_threshold=0.0,
                )
            ),
        )

    async def ingest(
        self,
        records: Sequence[MemoryRecord] | None = None,
        *,
        checkpoint: str | None = "fixture-cursor",
    ) -> IngestionReceipt:
        selected = tuple(records or self.records)
        kernel_records = [MemoryRecordAdapter.to_record(record) for record in selected]
        return await self.ingestor.index_records(kernel_records, checkpoint=checkpoint)

    async def ingest_kernel_records(
        self,
        records: Sequence[Record],
        *,
        checkpoint: str | None = "fixture-cursor",
    ) -> IngestionReceipt:
        return await self.ingestor.index_records(records, checkpoint=checkpoint)

    async def search(
        self,
        query: str,
        *,
        workspace_id: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
        limit: int = 5,
    ) -> SearchObservation:
        before = len(self.vector_store.searches)
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
        elapsed_ms = (perf_counter() - started) * 1000.0
        return SearchObservation(
            result_ids=tuple(result.record_id for result in outcome.results),
            elapsed_ms=elapsed_ms,
            cache_diagnostics=outcome.cache_diagnostics,
            diagnostics=outcome.diagnostics,
            candidate_observations=tuple(self.vector_store.searches[before:]),
        )

    def close(self) -> None:
        self.embedding_cache.close()


_BENCHMARK_QUERIES = (
    ("authentication policy", "workspace-a"),
    ("database migration", "workspace-a"),
    ("database migration", "workspace-b"),
)


async def _run_benchmark_size(
    cache_path: Path,
    *,
    record_count: int,
    warmups: int,
    repetitions: int,
    profiler: cProfile.Profile | None,
) -> dict[str, object]:
    started = perf_counter()
    harness = IngestSearchHarness(cache_path, records=scaled_memories(record_count))
    try:
        receipt = await harness.ingest()
        setup_ms = (perf_counter() - started) * 1000.0
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
                    observation = await harness.search(query, workspace_id=workspace)
                    samples_by_case[(query, workspace)].append(observation.elapsed_ms)
        finally:
            if profiler is not None:
                profiler.disable()

        search = {
            f"{query}:{workspace}": {
                "result_ids": cold_observations[(query, workspace)].result_ids,
                "elapsed_ms": round(cold_observations[(query, workspace)].elapsed_ms, 3),
                "cache_diagnostics": cold_observations[(query, workspace)].cache_diagnostics,
                "diagnostics": cold_observations[(query, workspace)].diagnostics,
                "candidate_observations": [
                    {
                        "candidate_ids": item.candidate_ids,
                        "workspace_id": item.workspace_id,
                        "limit": item.limit,
                    }
                    for item in cold_observations[(query, workspace)].candidate_observations
                ],
            }
            for query, workspace in _BENCHMARK_QUERIES
        }
        return {
            "record_count": record_count,
            "ingest": {
                "attempted": receipt.attempted,
                "committed": receipt.committed,
                "failed": receipt.failed,
                "checkpoint": receipt.checkpoint,
            },
            "cold": {
                "startup_ms": round(setup_ms, 3),
                "first_search_ms": round(cold_observation.elapsed_ms, 3),
            },
            "warm": {
                "repetitions": repetitions,
                "warmups": warmups,
                "overall": _summarize_latency(
                    [sample for samples in samples_by_case.values() for sample in samples]
                ).to_mapping(),
                "by_case": {
                    f"{query}:{workspace}": _summarize_latency(
                        samples_by_case[(query, workspace)]
                    ).to_mapping()
                    for query, workspace in _BENCHMARK_QUERIES
                },
            },
            "search": search,
        }
    finally:
        harness.close()


async def _run_benchmark(
    cache_path: Path,
    *,
    record_counts: Sequence[int] = (7,),
    warmups: int = 3,
    repetitions: int = 10,
    profile_path: Path | None = None,
) -> dict[str, object]:
    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    profiler = cProfile.Profile() if profile_path is not None else None
    benchmarks = [
        await _run_benchmark_size(
            cache_path,
            record_count=record_count,
            warmups=warmups,
            repetitions=repetitions,
            profiler=profiler,
        )
        for record_count in record_counts
    ]
    if profiler is not None and profile_path is not None:
        profiler.dump_stats(profile_path)
    first = benchmarks[0]
    return {
        "benchmarks": benchmarks,
        "ingest": first["ingest"],
        "search": first["search"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="SQLite embedding cache path (defaults to a temporary file)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[len(sample_memories())],
        help="Corpus sizes to measure (default: 7).",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=3,
        help="Unmeasured searches per query before sampling (default: 3).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        help="Measured rounds per query (default: 10).",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help="Write a cProfile stats file for warmed searches.",
    )
    args = parser.parse_args()
    if args.cache is None:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="searchkernel-parity-") as directory:
            result = asyncio.run(
                _run_benchmark(
                    Path(directory) / "embeddings.db",
                    record_counts=args.sizes,
                    warmups=args.warmups,
                    repetitions=args.repetitions,
                    profile_path=args.profile,
                )
            )
    else:
        result = asyncio.run(
            _run_benchmark(
                args.cache,
                record_counts=args.sizes,
                warmups=args.warmups,
                repetitions=args.repetitions,
                profile_path=args.profile,
            )
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=list))


if __name__ == "__main__":
    main()
