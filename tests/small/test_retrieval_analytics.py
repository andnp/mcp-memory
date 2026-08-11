from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import mcp_memory.application.memory_use_cases as memory_use_cases
from mcp_memory.application.memory_use_cases import (
    ReadMemoryRecordUseCase,
    SearchMemoryRecordsUseCase,
)
from mcp_memory.application.ports import (
    MemoryReadDependencies,
    MemorySearchPort,
    RetrievalTelemetryPort,
)
from mcp_memory.core.ports import MemoryIDResolutionPort
from mcp_memory.management.analytics_reporting import build_retrieval_analytics
from mcp_memory.management.analytics_reporting import build_nerd_metrics
from mcp_memory.management.reporting_rows import MemoryToolEventRow, ScopedMemoryRow
from mcp_memory.mcp.internal_services import internal_search_memory_records_service
from mcp_memory.mcp.runtime import create_runtime
from mcp_memory.mcp.services import read_memory_record_service, search_memory_records_service
from mcp_memory.provider_usage_store import ProviderUsageRepository
from mcp_memory.relational.search import SearchExecutionDiagnostics
from mcp_memory.utils.db import DatabaseManager


pytestmark = pytest.mark.small


def _raw_search_result(
    memory_id: str = "memory-1",
    *,
    title: str = "Search result",
    summary: str = "Search summary",
) -> SimpleNamespace:
    record = SimpleNamespace(
        source_id=memory_id,
        title=title,
        body="Search summary",
        metadata={
            "memory_ref": 1,
            "summary": summary,
            "memory_type": "fact",
            "memory_status": "active",
            "tags": [],
            "workspace_ids": [],
        },
        status=SimpleNamespace(value="active"),
        storage_key=f"memory:{memory_id}",
    )
    provenance = SimpleNamespace(
        strategies=("keyword",),
        to_dict=lambda: {"strategies": ["keyword"]},
    )
    return SimpleNamespace(record=record, score=0.9, provenance=provenance)


def test_debug_search_reports_bounded_application_phase_timings(monkeypatch) -> None:
    """Debug search exposes deterministic bounded phase timings."""
    clock_values = iter((0.0, 0.001, 0.004, 0.006, 100.006, 100.007, 100.008, 100.009))
    monkeypatch.setattr(memory_use_cases, "perf_counter", lambda: next(clock_values))
    health = SimpleNamespace(repair_wait_count=0, last_repair_wait_seconds=0.007)
    diagnostics = SearchExecutionDiagnostics(timing_ms={"total": 2.0, "search": 1.5})

    def search_sync_with_diagnostics(_request, *, debug):
        assert debug is True
        health.repair_wait_count += 1
        return SimpleNamespace(results=(_raw_search_result(),)), diagnostics

    typed_health = SimpleNamespace(get_health=lambda: health)
    legacy_search = SimpleNamespace(
        get_health=lambda: pytest.fail("legacy search health should not be called")
    )
    ctx = cast(
        MemoryReadDependencies,
        SimpleNamespace(
            workspace_id="workspace-1",
            relational_search=cast(MemorySearchPort, legacy_search),
            search_health=typed_health,
            memory_retrieval=SimpleNamespace(
                search_sync_with_diagnostics=search_sync_with_diagnostics,
            ),
            config=None,
            repository=None,
            read_cache=None,
            read_cache_validation=None,
            memory_id_resolution=None,
            vector_store=None,
            embedder=None,
            embedding_maintenance=None,
            surface_tracker=None,
        ),
    )
    telemetry = cast(
        RetrievalTelemetryPort,
        SimpleNamespace(record_search=lambda **_kwargs: None),
    )

    payload = SearchMemoryRecordsUseCase(ctx, telemetry).execute(
        {
            "query": "timing query",
            "workspace_id": None,
            "limit": 5,
            "adaptive_limit": False,
            "memory_type": None,
            "status": None,
            "tags": (),
            "include_superseded": False,
            "debug": True,
        }
    )

    timing_ms = payload["timing_ms"]
    assert timing_ms["fresh_cache_lookup"] == 3.0
    assert timing_ms["authoritative_search"] == 60_000.0
    assert timing_ms["validation_projection_warming"] == 1.0
    assert timing_ms["embedding_repair_wait"] == 7.0
    assert timing_ms["coalesced_wait"] == 0.0
    phase_keys = {
        "fresh_cache_lookup",
        "coalesced_wait",
        "authoritative_search",
        "validation_projection_warming",
        "embedding_repair_wait",
    }
    assert all(isinstance(timing_ms[key], (int, float)) for key in phase_keys)
    assert all(0.0 <= timing_ms[key] <= 60_000.0 for key in phase_keys)
    assert payload["search_diagnostics"]["timing_ms"] == timing_ms


def test_read_uses_typed_memory_id_resolution_before_legacy_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the typed resolver for telemetry while preserving the read payload."""
    resolver_calls: list[str] = []
    resolver = SimpleNamespace(
        resolve_memory_id=lambda memory_id: resolver_calls.append(memory_id)
        or "resolved-by-capability"
    )
    legacy_search = SimpleNamespace(
        resolve_memory_id=lambda _memory_id: pytest.fail(
            "legacy resolver should not be called"
        )
    )
    record = SimpleNamespace(
        id="resolved-by-capability",
        memory_ref=1,
        title="Typed resolver result",
        content="content",
        metadata={},
        workspace_ids=[],
    )
    monkeypatch.setattr(
        memory_use_cases.ReadMemoryRecordOperation,
        "execute",
        lambda _operation, _memory_id: SimpleNamespace(
            record=record,
            relationships={},
            superseded=[],
        ),
    )
    telemetry_calls: list[dict[str, object]] = []
    telemetry = cast(
        RetrievalTelemetryPort,
        SimpleNamespace(record_read=lambda **kwargs: telemetry_calls.append(kwargs)),
    )
    ctx = MemoryReadDependencies(
        relational_search=cast(MemorySearchPort, legacy_search),
        memory_id_resolution=cast(MemoryIDResolutionPort, resolver),
    )

    payload = ReadMemoryRecordUseCase(ctx, telemetry).execute(
        {
            "memory_id": "mem-1",
            "include_relationships": False,
            "include_superseded": False,
            "include_metadata": False,
        }
    )

    assert payload["status"] == "ok"
    assert resolver_calls == ["mem-1"]
    assert telemetry_calls[0]["memory_id"] == "resolved-by-capability"


def test_non_debug_search_does_not_add_phase_timings() -> None:
    """Non-debug search retains its compact response contract."""
    retrieval = SimpleNamespace(
        search_sync=lambda _request: SimpleNamespace(
                results=(_raw_search_result(title="Compact result", summary="Compact summary"),),
        )
    )
    ctx = MemoryReadDependencies(
        workspace_id="workspace-1",
        relational_search=cast(MemorySearchPort, SimpleNamespace()),
        memory_retrieval=retrieval,
    )
    telemetry = cast(
        RetrievalTelemetryPort,
        SimpleNamespace(record_search=lambda **_kwargs: None),
    )

    payload = SearchMemoryRecordsUseCase(ctx, telemetry).execute(
        {
            "query": "compact query",
            "workspace_id": None,
            "limit": 5,
            "adaptive_limit": False,
            "memory_type": None,
            "status": None,
            "tags": (),
            "include_superseded": False,
            "debug": False,
        }
    )

    assert payload == {
        "status": "ok",
        "results": [
            {
                "memory_ref": "mem-1",
                "title": "Compact result",
                "summary": "Compact summary",
                "status": "active",
                "created_at": None,
                "updated_at": None,
            }
        ],
    }


def test_open_connection_timeout_bounds_sqlite_busy_lock(tmp_path: Path) -> None:
    manager = DatabaseManager(tmp_path / "memory.sqlite3")
    blocker = manager.get_connection()
    bounded = manager.open_connection(timeout_seconds=0.1)
    overridden = manager.open_connection(timeout_seconds=0.1, busy_timeout_milliseconds=2500)

    try:
        assert manager.get_connection().execute("PRAGMA busy_timeout").fetchone()[0] == 30_000
        assert bounded.execute("PRAGMA busy_timeout").fetchone()[0] == 100
        assert overridden.execute("PRAGMA busy_timeout").fetchone()[0] == 2500

        blocker.execute("BEGIN IMMEDIATE")
        started_at = time.monotonic()
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            bounded.execute("UPDATE schema_metadata SET value = value WHERE key = 'schema_version'")
        assert time.monotonic() - started_at < 1.0
    finally:
        blocker.rollback()
        bounded.close()
        overridden.close()
        manager.close()


def test_database_manager_reopens_connection_after_close(tmp_path: Path) -> None:
    manager = DatabaseManager(tmp_path / "memory.sqlite3")
    connection = manager.get_connection()

    manager.close()

    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")

    reopened = manager.get_connection()
    try:
        assert reopened is not connection
        assert reopened.execute("SELECT 1").fetchone()[0] == 1
    finally:
        manager.close()


def test_build_retrieval_analytics_rolls_up_direct_search_and_read_rows() -> None:
    payload = build_retrieval_analytics(
        [
            ScopedMemoryRow(
                id="memory-a",
                title="Alpha retrieval memory",
                memory_type="fact",
                status="active",
                tags=["alpha-tag", "common-tag"],
            ),
            ScopedMemoryRow(
                id="memory-b",
                title="Beta retrieval memory",
                memory_type="plan",
                status="stale",
                tags=["beta-tag", "common-tag"],
            ),
        ],
        retrieval_rows=[
            MemoryToolEventRow(
                event_kind="search",
                invocation_id="search-1",
                created_at=100.0,
                caller_kind="external",
                query_text="Alpha phrase",
                result_count=2,
                memory_id="memory-a",
            ),
            MemoryToolEventRow(
                event_kind="search",
                invocation_id="search-1",
                created_at=100.0,
                caller_kind="external",
                query_text="Alpha phrase",
                result_count=2,
                memory_id="memory-b",
                graph_provenance={"path": "graph"},
            ),
            MemoryToolEventRow(
                event_kind="read",
                invocation_id="read-1",
                created_at=130.0,
                caller_kind="external",
                memory_id="memory-a",
            ),
            MemoryToolEventRow(
                event_kind="search",
                invocation_id="search-2",
                created_at=160.0,
                caller_kind="internal",
                query_text="Missing phrase",
                result_count=0,
            ),
            MemoryToolEventRow(
                event_kind="search",
                invocation_id="search-3",
                created_at=170.0,
                caller_kind="external",
                query_text="  alpha   phrase  ",
                result_count=1,
                memory_id="missing-memory",
            ),
        ],
        cutoff=0.0,
        generated_at=200.0,
        bucket_seconds=60,
    )

    assert payload.summary.search_invocations == 3
    assert payload.summary.search_hits == 2
    assert payload.summary.zero_result_searches == 1
    assert payload.summary.read_events == 1
    assert payload.summary.unique_search_memories == 2
    assert payload.summary.unique_read_memories == 1
    assert payload.funnel.search_hits == 2
    assert payload.funnel.converted_search_hits == 1
    assert payload.funnel.conversion_rate == 0.5
    evidence = {(row.memory_id, row.evidence_kind): row for row in payload.engagement_evidence}
    assert evidence[("memory-a", "search_to_read")].strength == "strong"
    assert evidence[("memory-b", "skipped_co_result")].co_result_read is True
    assert evidence[("memory-b", "skipped_co_result")].graph_provenance == {"path": "graph"}
    assert not any(row.memory_id == "missing-memory" for row in payload.engagement_evidence)

    by_caller_kind = {row.key: row for row in payload.by_caller_kind}
    assert by_caller_kind["external"].search_invocations == 2
    assert by_caller_kind["external"].search_hits == 2
    assert by_caller_kind["external"].zero_result_searches == 0
    assert by_caller_kind["external"].read_events == 1
    assert by_caller_kind["internal"].search_invocations == 1
    assert by_caller_kind["internal"].search_hits == 0
    assert by_caller_kind["internal"].zero_result_searches == 1

    query_families = {row.key: row for row in payload.top_query_families}
    assert query_families["alpha phrase"].search_invocations == 2
    assert query_families["alpha phrase"].search_hits == 2
    assert query_families["alpha phrase"].unique_search_memories == 2
    assert query_families["alpha phrase"].converted_search_hits == 1
    assert query_families["alpha phrase"].conversion_rate == 0.5
    assert query_families["missing phrase"].search_invocations == 1
    assert query_families["missing phrase"].search_hits == 0
    assert query_families["missing phrase"].zero_result_searches == 1
    assert [row.key for row in payload.top_zero_result_query_families] == ["missing phrase"]

    assert [row.memory_id for row in payload.top_read_memories] == ["memory-a"]
    assert [row.memory_id for row in payload.top_search_memories] == ["memory-a", "memory-b"]
    assert [row.memory_id for row in payload.low_conversion_memories] == ["memory-b", "memory-a"]

    top_tags = {row.key: row for row in payload.top_tags}
    assert top_tags["common-tag"].read_count == 1
    assert top_tags["common-tag"].search_count == 2
    assert top_tags["alpha-tag"].search_count == 1
    assert top_tags["beta-tag"].search_count == 1

    timelines = {row.key: row for row in payload.tag_timelines}
    assert sum(bucket.count for bucket in timelines["common-tag"].read_buckets) == 1
    assert sum(bucket.count for bucket in timelines["common-tag"].search_buckets) == 2


def test_build_nerd_metrics_includes_retrieval_analytics(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    try:
        assert runtime.repository is not None
        assert runtime.workspace_id is not None

        alpha = runtime.repository.create_memory(
            title="Alpha retrieval memory",
            content="alpha phrase for retrieval analytics",
            workspace_ids=[runtime.workspace_id],
            memory_type="fact",
            tags=["alpha-tag", "common-tag"],
        )
        beta = runtime.repository.create_memory(
            title="Beta retrieval memory",
            content="beta phrase for retrieval analytics",
            workspace_ids=[runtime.workspace_id],
            memory_type="plan",
            status="stale",
            tags=["beta-tag", "common-tag"],
        )
        gamma = runtime.repository.create_memory(
            title="Gamma retrieval memory",
            content="gamma phrase for retrieval analytics",
            workspace_ids=[runtime.workspace_id],
            memory_type="reflection",
            tags=["gamma-tag", "common-tag"],
        )
        assert alpha is not None
        assert beta is not None
        assert gamma is not None

        alpha_search = search_memory_records_service(runtime, {"query": "alpha phrase", "limit": 5, "status": "active", "memory_type": "fact"})
        beta_search = search_memory_records_service(runtime, {"query": "beta phrase", "limit": 5, "status": "stale"})
        gamma_search = search_memory_records_service(runtime, {"query": "gamma phrase", "limit": 5, "status": "active", "memory_type": "reflection"})
        missing_search = search_memory_records_service(runtime, {"query": "alpha phrase", "limit": 5, "status": "archived", "memory_type": "fact"})
        internal_beta_search = internal_search_memory_records_service(runtime, {"query": "beta phrase", "limit": 5, "status": "stale"})
        alpha_read = read_memory_record_service(runtime, {"memory_id": alpha.id})
        second_alpha_read = read_memory_record_service(runtime, {"memory_id": alpha.id})
        beta_read = read_memory_record_service(runtime, {"memory_id": beta.id})

        assert alpha_search["status"] == "ok"
        assert beta_search["status"] == "ok"
        assert gamma_search["status"] == "ok"
        assert missing_search["status"] == "ok"
        assert internal_beta_search["status"] == "ok"
        assert alpha_read["status"] == "ok"
        assert second_alpha_read["status"] == "ok"
        assert beta_read["status"] == "ok"

        def resolved_result_ids(search_payload: dict) -> set[str]:
            return {
                runtime.repository.resolve_memory_id(row["memory_ref"]) or row["memory_ref"]
                for row in search_payload["results"]
                if isinstance(row.get("memory_ref"), str)
            }

        alpha_result_ids = resolved_result_ids(alpha_search)
        beta_result_ids = resolved_result_ids(beta_search)
        gamma_result_ids = resolved_result_ids(gamma_search)
        missing_result_ids = resolved_result_ids(missing_search)
        internal_beta_result_ids = resolved_result_ids(internal_beta_search)
        expected_search_hits = sum(
            len(result_ids)
            for result_ids in (
                alpha_result_ids,
                beta_result_ids,
                gamma_result_ids,
                missing_result_ids,
                internal_beta_result_ids,
            )
        )
        expected_unique_search_memories = len(
            alpha_result_ids
            | beta_result_ids
            | gamma_result_ids
            | missing_result_ids
            | internal_beta_result_ids
        )
        expected_search_counts_by_memory = {
            memory_id: sum(
                1
                for result_ids in (
                    alpha_result_ids,
                    beta_result_ids,
                    gamma_result_ids,
                    missing_result_ids,
                    internal_beta_result_ids,
                )
                if memory_id in result_ids
            )
            for memory_id in (
                alpha_result_ids
                | beta_result_ids
                | gamma_result_ids
                | missing_result_ids
                | internal_beta_result_ids
            )
        }
        converted_memory_ids = {alpha.id, beta.id}
        def rounded_rate(numerator: int, denominator: int) -> float:
            if denominator <= 0:
                return 0.0
            return round(numerator / denominator, 4)

        expected_converted_search_hits = sum(
            len(result_ids & converted_memory_ids)
            for result_ids in (
                alpha_result_ids,
                beta_result_ids,
                gamma_result_ids,
                missing_result_ids,
                internal_beta_result_ids,
            )
        )

        payload = build_nerd_metrics(
            db_manager=runtime.db_manager,
            workspace_id=runtime.workspace_id,
            task_queue=runtime.task_queue,
            provider_usage_repo=ProviderUsageRepository(runtime.db_manager, workspace_id=runtime.workspace_id),
            config=runtime.config,
            ai_json_provider=runtime.ai_json_provider,
            ai_agent_provider=runtime.ai_agent_provider,
            ai_provider_registry=runtime.ai_provider_registry,
            relational_search=runtime.relational_search,
            window_hours=24,
            bucket_minutes=60,
            now=time.time() + 1.0,
        )

        assert payload.retrieval.summary.search_invocations == 5
        assert payload.retrieval.summary.search_hits == expected_search_hits
        assert payload.retrieval.summary.zero_result_searches == 1
        assert payload.retrieval.summary.read_events == 3
        assert payload.retrieval.summary.unique_search_memories == expected_unique_search_memories
        assert payload.retrieval.summary.unique_read_memories == 2
        assert payload.retrieval.funnel.search_hits == expected_search_hits
        assert payload.retrieval.funnel.converted_search_hits == expected_converted_search_hits
        assert payload.retrieval.funnel.conversion_rate == pytest.approx(
            rounded_rate(expected_converted_search_hits, expected_search_hits)
        )

        caller_kind_counts = {row.key: row for row in payload.retrieval.by_caller_kind}
        assert caller_kind_counts["external"].search_invocations == 4
        assert caller_kind_counts["external"].search_hits == (
            len(alpha_result_ids)
            + len(beta_result_ids)
            + len(gamma_result_ids)
            + len(missing_result_ids)
        )
        assert caller_kind_counts["external"].read_events == 3
        assert caller_kind_counts["external"].zero_result_searches == 1
        assert caller_kind_counts["internal"].search_invocations == 1
        assert caller_kind_counts["internal"].search_hits == len(internal_beta_result_ids)
        assert caller_kind_counts["internal"].read_events == 0

        query_family_counts = {row.key: row for row in payload.retrieval.top_query_families}
        assert query_family_counts["alpha phrase"].search_invocations == 2
        assert query_family_counts["alpha phrase"].search_hits == len(alpha_result_ids) + len(missing_result_ids)
        assert query_family_counts["alpha phrase"].zero_result_searches == 1
        assert query_family_counts["alpha phrase"].unique_search_memories == len(alpha_result_ids | missing_result_ids)
        assert query_family_counts["alpha phrase"].converted_search_hits == len((alpha_result_ids | missing_result_ids) & converted_memory_ids)
        assert query_family_counts["alpha phrase"].conversion_rate == pytest.approx(
            rounded_rate(
                query_family_counts["alpha phrase"].converted_search_hits,
                query_family_counts["alpha phrase"].search_hits,
            )
        )
        assert query_family_counts["beta phrase"].search_invocations == 2
        assert query_family_counts["beta phrase"].search_hits == len(beta_result_ids) + len(internal_beta_result_ids)
        assert query_family_counts["beta phrase"].zero_result_searches == 0
        assert query_family_counts["beta phrase"].unique_search_memories == len(beta_result_ids | internal_beta_result_ids)
        assert query_family_counts["beta phrase"].converted_search_hits == len(beta_result_ids & converted_memory_ids) + len(internal_beta_result_ids & converted_memory_ids)
        assert query_family_counts["beta phrase"].conversion_rate == pytest.approx(
            rounded_rate(
                query_family_counts["beta phrase"].converted_search_hits,
                query_family_counts["beta phrase"].search_hits,
            )
        )
        assert query_family_counts["gamma phrase"].search_invocations == 1
        assert query_family_counts["gamma phrase"].search_hits == len(gamma_result_ids)
        assert query_family_counts["gamma phrase"].zero_result_searches == 0
        assert query_family_counts["gamma phrase"].unique_search_memories == len(gamma_result_ids)
        assert query_family_counts["gamma phrase"].converted_search_hits == len(gamma_result_ids & converted_memory_ids)
        assert query_family_counts["gamma phrase"].conversion_rate == pytest.approx(
            rounded_rate(
                query_family_counts["gamma phrase"].converted_search_hits,
                query_family_counts["gamma phrase"].search_hits,
            )
        )

        assert [row.key for row in payload.retrieval.top_zero_result_query_families] == ["alpha phrase"]
        assert payload.retrieval.top_zero_result_query_families[0].zero_result_searches == 1

        low_conversion_ids = [row.memory_id for row in payload.retrieval.low_conversion_memories]
        assert gamma.id in low_conversion_ids
        gamma_conversion = next(row for row in payload.retrieval.low_conversion_memories if row.memory_id == gamma.id)
        assert gamma_conversion.converted_search_count == 0
        assert gamma_conversion.conversion_rate == pytest.approx(0.0)

        assert [row.memory_id for row in payload.retrieval.top_read_memories[:2]] == [alpha.id, beta.id]
        assert [row.read_count for row in payload.retrieval.top_read_memories[:2]] == [2, 1]
        top_search_counts = {row.memory_id: row.search_count for row in payload.retrieval.top_search_memories}
        assert top_search_counts[alpha.id] == expected_search_counts_by_memory[alpha.id]
        assert top_search_counts[beta.id] == expected_search_counts_by_memory[beta.id]
        assert top_search_counts[gamma.id] == expected_search_counts_by_memory[gamma.id]

        top_tag_keys = {row.key for row in payload.retrieval.top_tags}
        assert {"alpha-tag", "beta-tag", "gamma-tag", "common-tag"}.issubset(top_tag_keys)
        common_tag = next(row for row in payload.retrieval.top_tags if row.key == "common-tag")
        assert common_tag.read_count == 3
        assert common_tag.search_count == expected_search_hits
        assert common_tag.total_count == expected_search_hits + 3

        timeline_keys = {series.key for series in payload.retrieval.tag_timelines}
        assert {"alpha-tag", "beta-tag", "gamma-tag", "common-tag"}.issubset(timeline_keys)
        common_timeline = next(series for series in payload.retrieval.tag_timelines if series.key == "common-tag")
        assert sum(bucket.count for bucket in common_timeline.read_buckets) == 3
        assert sum(bucket.count for bucket in common_timeline.search_buckets) == expected_search_hits
    finally:
        runtime.close()


def test_search_memory_records_service_returns_results_while_sqlite_write_lock_is_held(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    try:
        assert runtime.repository is not None
        assert runtime.db_manager is not None
        assert runtime.workspace_id is not None

        record = runtime.repository.create_memory(
            title="Parallel search reliability",
            content="Parallel searches should still return results under telemetry write lock contention.",
            workspace_ids=[runtime.workspace_id],
            memory_type="fact",
            tags=["search", "reliability"],
        )
        assert record is not None
        assert runtime.embedding_maintenance is not None
        monkeypatch.setattr(
            runtime.embedding_maintenance,
            "ensure_searchable_memory_embeddings",
            lambda _candidates: None,
        )

        lock_conn = sqlite3.connect(str(runtime.db_manager.db_path), timeout=0.1)
        try:
            lock_conn.execute("PRAGMA journal_mode=WAL;")
            lock_conn.execute("BEGIN IMMEDIATE")

            payload = search_memory_records_service(runtime, {"query": "parallel search reliability", "limit": 5})

            assert payload["status"] == "ok"
            assert [row["memory_ref"] for row in payload["results"]] == [
                f"mem-{record.memory_ref}"
            ]
        finally:
            lock_conn.rollback()
            lock_conn.close()
    finally:
        runtime.close()
