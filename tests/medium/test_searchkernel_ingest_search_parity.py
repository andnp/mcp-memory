"""Deterministic searchkernel ingest-to-search parity coverage."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from searchkernel.domain import Record
from searchkernel.runtime.query_embedding_cache import clear_query_embedding_cache

from benchmarks.searchkernel_ingest_search import (
    IngestSearchHarness,
    sample_memories,
)
from mcp_memory.integrations.searchkernel_adapters import MemoryRecordAdapter
from mcp_memory.integrations.searchkernel_ingestion import MemoryRecordIngestor


pytestmark = pytest.mark.medium


@pytest.fixture
def harness(tmp_path: Path):
    clear_query_embedding_cache()
    value = IngestSearchHarness(tmp_path / "embedding-cache.db")
    try:
        yield value
    finally:
        value.close()
        clear_query_embedding_cache()


@pytest.mark.asyncio
async def test_ingest_receipt_and_search_are_connected(harness: IngestSearchHarness) -> None:
    receipt = await harness.ingest()

    assert receipt.attempted == len(sample_memories())
    assert receipt.committed == len(sample_memories())
    assert receipt.failed == 0
    assert receipt.checkpoint == "fixture-cursor"

    observation = await harness.search("authentication policy", workspace_id="workspace-a")

    assert observation.result_ids == ("auth-current",)
    assert observation.elapsed_ms >= 0.0
    assert observation.candidate_observations


@pytest.mark.asyncio
async def test_memory_ingestor_round_trips_into_kernel_search(
    harness: IngestSearchHarness,
) -> None:
    ingestor = MemoryRecordIngestor(
        object(),
        cast(Any, harness.vector_store),
        harness.embedder,
    )

    receipt = await ingestor.index_records(sample_memories())

    assert receipt.committed == len(sample_memories())
    observation = await harness.search("authentication policy", workspace_id="workspace-a")

    assert observation.result_ids == ("auth-current",)


@pytest.mark.asyncio
async def test_workspace_filters_isolate_search_and_keep_shared_membership(
    harness: IngestSearchHarness,
) -> None:
    await harness.ingest()

    workspace_a = await harness.search("database migration", workspace_id="workspace-a")
    workspace_b = await harness.search("database migration", workspace_id="workspace-b")

    assert "db-a" in workspace_a.result_ids
    assert "db-b" not in workspace_a.result_ids
    assert "db-b" in workspace_b.result_ids
    assert "db-a" not in workspace_b.result_ids
    assert all(
        item.workspace_id == "workspace-a"
        for item in workspace_a.candidate_observations
    )
    assert all(
        item.workspace_id == "workspace-b"
        for item in workspace_b.candidate_observations
    )

    shared_a = await harness.search("deployment", workspace_id="workspace-a")
    shared_b = await harness.search("deployment", workspace_id="workspace-b")
    assert shared_a.result_ids == shared_b.result_ids == ("shared-workspace",)


@pytest.mark.asyncio
async def test_supersession_and_status_filters_match_memory_policy(
    harness: IngestSearchHarness,
) -> None:
    await harness.ingest()

    default = await harness.search("authentication", workspace_id="workspace-a")
    with_superseded = await harness.search(
        "authentication",
        workspace_id="workspace-a",
        include_superseded=True,
    )
    archived = await harness.search(
        "authentication",
        workspace_id="workspace-a",
        status="archived",
    )
    stale = await harness.search(
        "database migration",
        workspace_id="workspace-a",
        status="stale",
    )

    assert "auth-legacy" not in default.result_ids
    assert "auth-archived" not in default.result_ids
    assert "auth-legacy" in with_superseded.result_ids
    assert archived.result_ids == ("auth-archived",)
    assert stale.result_ids == ("db-stale",)


@pytest.mark.asyncio
async def test_multi_workspace_identity_is_collision_safe(harness: IngestSearchHarness) -> None:
    timestamp = datetime(2026, 8, 1, 12, tzinfo=UTC)
    first = Record(
        source_kind="memory",
        source_id="same-id",
        title="Workspace one",
        body="workspace one deployment",
        created_at=timestamp,
        updated_at=timestamp,
        workspace_id="workspace-a",
    )
    second = Record(
        source_kind="memory",
        source_id="same-id",
        title="Workspace two",
        body="workspace two deployment",
        created_at=timestamp,
        updated_at=timestamp,
        workspace_id="workspace-b",
    )

    assert first.storage_key != second.storage_key
    receipt = await harness.ingest_kernel_records([first, second])

    assert receipt.committed == 2
    assert harness.vector_store.identity_keys() == (
        first.storage_key,
        second.storage_key,
    )

    memory_first = MemoryRecordAdapter.to_record(sample_memories()[0])
    assert memory_first.storage_key.startswith('record:["workspace-a","memory"')


@pytest.mark.asyncio
async def test_cache_diagnostics_and_candidate_observability_are_repeatable(
    harness: IngestSearchHarness,
) -> None:
    await harness.ingest()

    first = await harness.search("authentication policy", workspace_id="workspace-a")
    second = await harness.search("authentication policy", workspace_id="workspace-a")

    assert first.cache_diagnostics == second.cache_diagnostics == (
        "candidate_cache:bypass:unstable_policy",
    )
    assert first.candidate_observations
    assert second.candidate_observations
    assert first.candidate_observations[-1].candidate_ids == ("auth-current",)
    assert second.candidate_observations[-1].candidate_ids == ("auth-current",)
    assert first.result_ids == second.result_ids == ("auth-current",)
    assert harness.embedder.calls.count(["authentication policy"]) == 1
    assert first.elapsed_ms >= 0.0
    assert second.elapsed_ms >= 0.0
