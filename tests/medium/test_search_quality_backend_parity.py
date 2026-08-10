"""Backend parity coverage for the deterministic search-quality corpus."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

import pytest

from benchmarks.search_quality import (
    SearchQualityCase,
    SearchObservation,
    SearchQualityMetrics,
    TopicEmbedder,
    load_corpus,
    run_in_process,
    run_search_quality,
    seed_search_quality_records,
)
from mcp_memory.config import Config
from mcp_memory.integrations.searchkernel_ingestion import MemoryRecordIngestor
from mcp_memory.integrations.searchkernel_record_pipeline import (
    MemoryRecordSearchPipeline,
    build_memory_record_pipeline,
)
from mcp_memory.storage.postgres import ensure_postgres_schema
from mcp_memory.storage.postgres_connection import PostgresConnectionManager
from mcp_memory.storage.postgres_repository import PostgresRelationalMemoryRepository
from mcp_memory.storage.postgres_vector_store import PostgresVectorStore


pytestmark = pytest.mark.medium


@dataclass
class _PostgresQualityContext:
    manager: PostgresConnectionManager
    repository: PostgresRelationalMemoryRepository
    vector_store: PostgresVectorStore
    embedder: TopicEmbedder
    label_to_id: dict[str, str]
    pipeline: MemoryRecordSearchPipeline


@pytest.fixture
def postgres_quality_context(
    postgres_storage_config,
) -> Iterator[_PostgresQualityContext]:
    """Provide a shared-Postgres corpus with the Python vector fallback."""
    ensure_postgres_schema(postgres_storage_config)
    with PostgresConnectionManager(postgres_storage_config) as manager:
        repository = PostgresRelationalMemoryRepository(manager)
        vector_store = PostgresVectorStore(manager)
        embedder = TopicEmbedder()
        label_to_id = seed_search_quality_records(repository)
        other = repository.create_memory(
            "Authentication policy elsewhere",
            "Authentication policy in another workspace.",
            ["other-workspace"],
            summary="Authentication policy in another workspace.",
            tags=["auth"],
            memory_type="fact",
        )
        assert other is not None
        label_to_id["other-workspace"] = other.id
        records = [
            record
            for memory_id in label_to_id.values()
            if (record := repository.get_memory(memory_id)) is not None
        ]
        receipt = asyncio.run(
            MemoryRecordIngestor(repository, vector_store, embedder).index_records(records)
        )
        assert receipt.failed == 0
        pipeline = build_memory_record_pipeline(
            repository,
            embedder=embedder,
            vector_store=vector_store,
            config=Config(),
        )
        yield _PostgresQualityContext(
            manager,
            repository,
            vector_store,
            embedder,
            label_to_id,
            pipeline,
        )


def test_search_quality_sqlite_baseline_is_corpus_ordered() -> None:
    """Run the versioned corpus through the real local SQLite search path."""
    corpus = load_corpus()
    metrics = run_in_process(corpus)

    assert metrics.corpus_version == corpus.version
    assert [case.evaluation_label for case in metrics.cases] == [
        entry.evaluation_label for entry in corpus.entries
    ]
    assert metrics.hit_at_1 == 1.0
    assert metrics.hit_at_5 == 1.0


def _vector_corpus_metrics(
    context: _PostgresQualityContext,
) -> SearchQualityMetrics:
    corpus = load_corpus()
    labels_by_id = {memory_id: label for label, memory_id in context.label_to_id.items()}

    def search_case(case: SearchQualityCase) -> SearchObservation:
        diagnostics: dict[str, object] = {}
        query_vector = context.embedder.embed([case.query])[0]
        hits = context.vector_store.search(
            source_kind="memory",
            model_name=context.embedder.model_name,
            query_embedding=query_vector,
            diagnostics=diagnostics,
            workspace_id=(None if case.workspace == "global" else case.workspace),
            limit=case.acceptable_top_k,
        )
        return SearchObservation(
            tuple(labels_by_id.get(memory_id, "unknown") for memory_id, _score in hits),
        )

    return run_search_quality(corpus, search_case)


def test_postgres_fallback_matches_corpus_order_and_workspace_scope(
    postgres_quality_context: _PostgresQualityContext,
) -> None:
    """Compare corpus ordering while proving another workspace stays excluded."""
    context = postgres_quality_context
    metrics = _vector_corpus_metrics(context)
    query_vector = context.embedder.embed(["auth policy"])[0]
    diagnostics: dict[str, object] = {}
    hits = context.vector_store.search(
        source_kind="memory",
        model_name=context.embedder.model_name,
        query_embedding=query_vector,
        diagnostics=diagnostics,
        workspace_id="workspace",
        limit=5,
    )

    assert metrics.hit_at_5 == 1.0
    assert context.label_to_id["other-workspace"] not in {
        memory_id for memory_id, _score in hits
    }
    assert diagnostics["search_mode"] == "client_python_fallback"
    assert diagnostics["server_side_vector_search_available"] is False


def test_global_search_includes_workspaces_and_ranks_active_project(
    postgres_quality_context: _PostgresQualityContext,
) -> None:
    """Verify global retrieval includes both projects with active-project preference."""
    context = postgres_quality_context
    outcome = asyncio.run(
        context.pipeline.search(
            "shared deployment guidance",
            limit=5,
            filters={"_ranking_workspace_id": "workspace"},
        )
    )
    labels_by_id = {memory_id: label for label, memory_id in context.label_to_id.items()}
    labels = [labels_by_id[result.record_id] for result in outcome.results]

    assert labels[:2] == ["global-project-ranking", "global-other-project"]


def test_postgres_fallback_reports_abstention_and_degradation(
    postgres_quality_context: _PostgresQualityContext,
) -> None:
    """Cover semantic-only abstention plus malformed-row degradation evidence."""
    context = postgres_quality_context
    outcome = asyncio.run(
        context.pipeline.search(
            "daemon graph",
            limit=5,
            filters={
                "workspace_id": "workspace",
                "retrieval_mode": "semantic_only",
            },
        )
    )

    assert outcome.results == ()
    assert outcome.degraded is False
    assert outcome.failures == ()

    with context.manager.open_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO embeddings (
                    source_kind, source_id, workspace_id, model_name,
                    embedding_json, updated_at
                ) VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                """,
                (
                    "memory",
                    "malformed-row",
                    "workspace",
                    context.embedder.model_name,
                    json.dumps({"not": "an-array"}),
                    time.time(),
                ),
            )
        connection.commit()

    diagnostics: dict[str, object] = {}
    context.vector_store.search(
        source_kind="memory",
        model_name=context.embedder.model_name,
        query_embedding=context.embedder.embed(["semantic retrieval"])[0],
        diagnostics=diagnostics,
        workspace_id="workspace",
        limit=5,
    )

    assert diagnostics["search_mode"] == "client_python_fallback"
    assert cast(int, diagnostics["skipped_invalid_embedding_count"]) >= 1


def test_postgres_fallback_reports_row_cap(
    postgres_quality_context: _PostgresQualityContext,
) -> None:
    """Verify fallback scans remain bounded and explain the bound in diagnostics."""
    context = postgres_quality_context
    bounded_store = PostgresVectorStore(context.manager, fallback_row_cap=2)
    diagnostics: dict[str, object] = {}
    hits = bounded_store.search(
        source_kind="memory",
        model_name=context.embedder.model_name,
        query_embedding=context.embedder.embed(["semantic retrieval"])[0],
        diagnostics=diagnostics,
        workspace_id="workspace",
        limit=5,
    )

    assert len(hits) <= 2
    assert diagnostics["row_count"] == 2
    assert diagnostics["fallback_row_cap"] == 2
    assert diagnostics["fallback_row_cap_applied"] is True
