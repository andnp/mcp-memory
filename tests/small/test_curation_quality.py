from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from mcp_memory.core.curation_models import CurationRunOutcome
from mcp_memory.core.curation_quality import CurationQualitySampler
from mcp_memory.curation_quality_store import SQLiteCurationQualityStore
from mcp_memory.curation_store import (
    CurationActionReceipt,
    CurationReceiptState,
    CurationRun,
    CurationRunState,
    SQLiteCurationStore,
)
from mcp_memory.management.analytics_curation import build_curation_metrics


pytestmark = pytest.mark.small


@dataclass
class _Search:
    contexts: list[object]
    calls: list[str] | None = None

    def search_memories_for_maintenance(self, query: str, *, limit: int = 50) -> list[object]:
        if self.calls is None:
            self.calls = []
        self.calls.append(query)
        return self.contexts[:limit]


def _run(run_id=None, *, created_at: datetime | None = None) -> CurationRun:
    return CurationRun(
        run_id=run_id or uuid4(),
        frontier_key="frontier",
        context_fingerprint="context",
        state=CurationRunState.TERMINAL,
        outcome=CurationRunOutcome.APPLIED,
        created_at=created_at or datetime.now(UTC),
    )


def _receipt(run_id, *, status=CurationReceiptState.VERIFIED, event_id=None, applied_at=None):
    return CurationActionReceipt(
        run_id=run_id,
        action_id=uuid4(),
        operation="rewrite_memory",
        affected_ids=[uuid4()],
        status=status,
        mutation_event_id=event_id,
        intent_hash="intent",
        applied_at=applied_at,
    )


def _context(memory_id: str, *, content: str = "after") -> object:
    return SimpleNamespace(
        record=SimpleNamespace(
            id=memory_id,
            title="title",
            summary="summary",
            content=content,
            tags=["tag"],
        )
    )


def test_quality_sampler_persists_no_query_without_positive_quality(db_manager) -> None:
    run = _run()
    SQLiteCurationStore(db_manager).create_run(run)
    receipt = _receipt(run.run_id, event_id=uuid4(), applied_at=datetime.now(UTC))
    search = _Search([])
    sampler = CurationQualitySampler(
        db_manager=db_manager,
        search=search,
        repository=SQLiteCurationQualityStore(db_manager),
        sample_rate=1.0,
    )

    evidence = sampler.evaluate(run=run, receipts=[receipt])

    assert evidence[0].status == "no_query"
    assert evidence[0].useful_work is None
    assert search.calls in (None, [])


def test_quality_sampler_replays_before_after_without_instrumenting_reads(db_manager) -> None:
    now = datetime.now(UTC)
    run = _run(created_at=now)
    SQLiteCurationStore(db_manager).create_run(run)
    event_id = uuid4()
    memory_id = uuid4()
    receipt = CurationActionReceipt(
        run_id=run.run_id,
        action_id=uuid4(),
        operation="rewrite_memory",
        affected_ids=[memory_id],
        status=CurationReceiptState.VERIFIED,
        mutation_event_id=event_id,
        intent_hash="intent",
        applied_at=now,
    )
    connection = db_manager.get_connection()
    noise_id = uuid4()
    for value in (memory_id, noise_id):
        connection.execute(
            """
            INSERT INTO memories (
                id, title, content, type, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (str(value), "title", "content", "observation", "active", now.isoformat(), now.isoformat()),
        )
    connection.execute(
        """
        INSERT INTO memory_mutation_events (
            id, operation, actor_kind, curation_run_id, status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (str(event_id), "rewrite_memory", "maintenance", str(run.run_id), "applied", now.isoformat()),
    )
    connection.execute(
        """
        INSERT INTO memory_tool_events (
            invocation_id, caller_kind, event_kind, memory_id, query_text, result_rank,
            result_count, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("query-1", "maintenance", "search", str(memory_id), "important", 2, 2, (now - timedelta(minutes=1)).isoformat()),
    )
    connection.execute(
        """
        INSERT INTO memory_tool_events (
            invocation_id, caller_kind, event_kind, memory_id, query_text, result_rank,
            result_count, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("query-1", "maintenance", "search", str(noise_id), "important", 1, 2, (now - timedelta(minutes=1)).isoformat()),
    )
    connection.execute(
        """
        INSERT INTO memory_record_revisions (
            event_id, memory_id, role, before_exists, before_snapshot,
            after_exists, after_snapshot
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (str(event_id), str(memory_id), "target", 1, json.dumps({"content": "before"}), 1, "{}"),
    )
    connection.commit()
    search = _Search([_context(str(memory_id), content="after")])
    sampler = CurationQualitySampler(
        db_manager=db_manager,
        search=search,
        repository=SQLiteCurationQualityStore(db_manager),
        sample_rate=1.0,
    )
    before_event_count = connection.execute("SELECT COUNT(*) FROM memory_tool_events").fetchone()[0]

    evidence = sampler.evaluate(run=run, receipts=[receipt])

    assert evidence[0].status == "evaluated"
    assert evidence[0].query_id == "query-1"
    assert evidence[0].before_ranked_memory_ids == [noise_id, memory_id]
    assert evidence[0].after_ranked_memory_ids == [memory_id]
    assert search.calls == ["important"]
    assert connection.execute("SELECT COUNT(*) FROM memory_tool_events").fetchone()[0] == before_event_count


def test_quality_sampler_skips_non_applied_receipts(db_manager) -> None:
    run = _run()
    SQLiteCurationStore(db_manager).create_run(run)
    search = _Search([])
    sampler = CurationQualitySampler(
        db_manager=db_manager,
        search=search,
        repository=SQLiteCurationQualityStore(db_manager),
        sample_rate=1.0,
    )

    evidence = sampler.evaluate(
        run=run,
        receipts=[_receipt(run.run_id, status=CurationReceiptState.REJECTED)],
    )

    assert evidence == ()
    assert search.calls in (None, [])


def test_quality_metrics_project_sampled_evidence(db_manager) -> None:
    now = datetime.now(UTC)
    run = _run(created_at=now)
    SQLiteCurationStore(db_manager).create_run(run)
    event_id = uuid4()
    receipt = _receipt(run.run_id, event_id=event_id, applied_at=now)
    from mcp_memory.core.curation_quality import CurationQualityEvidence

    SQLiteCurationQualityStore(db_manager).put_quality_evidence(
        CurationQualityEvidence(
            run_id=run.run_id,
            action_id=receipt.action_id,
            operation=receipt.operation,
            affected_memory_ids=receipt.affected_ids,
            policy_version=run.policy_version,
            status="no_query",
            created_at=now,
        )
    )

    metrics = build_curation_metrics(db_manager, window_hours=24, now=now.timestamp())

    assert metrics.verified_yield == 0.0
    assert metrics.retrieval_quality.sampled_action_count == 1
    assert metrics.retrieval_quality.no_query_action_count == 1
