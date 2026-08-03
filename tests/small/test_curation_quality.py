from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, Self, cast
from uuid import uuid4

import pytest

from mcp_memory.core.curation_models import (
    CampaignHypothesis,
    CampaignTargetMode,
    CurationRunOutcome,
)
from mcp_memory.core.curation_evaluation import ReplayCase, ReplayResult, ReplaySnapshot, evaluate_query_replay
from mcp_memory.core.curation_quality import (
    CurationQualityEvidence,
    CurationQualitySampler,
    _acceptance_met,
)
from mcp_memory.core.curation_work_items import CurationWorkItemService, WorkItemAction
from mcp_memory.curation_quality_store import (
    PostgresCurationQualityStore,
    SQLiteCurationQualityStore,
)
from mcp_memory.curation_store import (
    CandidateDisposition,
    CurationActionReceipt,
    CurationCandidateState,
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


class _PostgresQualityCursor:
    def __init__(self, connection: _PostgresQualityConnection) -> None:
        self._cursor = connection.database.cursor()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self._cursor.close()
        return False

    def execute(self, query: str, params: tuple[object, ...] = ()) -> None:
        self._cursor.execute(query.replace("%s::jsonb", "?").replace("%s", "?"), params)

    def fetchall(self) -> list[tuple[object, ...]]:
        return self._cursor.fetchall()


class _PostgresQualityConnection:
    def __init__(self) -> None:
        self.database = sqlite3.connect(":memory:")
        self.database.execute(
            """
            CREATE TABLE curation_quality_evidence (
                run_id TEXT NOT NULL,
                action_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                affected_memory_ids_json TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                query_id TEXT,
                status TEXT NOT NULL,
                before_ranked_ids_json TEXT NOT NULL,
                after_ranked_ids_json TEXT NOT NULL,
                retrieval_regression_count INTEGER,
                zero_result_change INTEGER,
                payload_size_change INTEGER,
                useful_work INTEGER,
                created_at TEXT NOT NULL,
                retrieval_utility_delta REAL,
                acceptance_met INTEGER,
                neutral_reason TEXT,
                content_quality_score_before REAL,
                content_quality_score_after REAL,
                content_quality_delta REAL,
                content_quality_improved INTEGER,
                PRIMARY KEY (run_id, action_id)
            )
            """
        )
        self.committed = False

    def cursor(self) -> _PostgresQualityCursor:
        return _PostgresQualityCursor(self)

    def commit(self) -> None:
        self.database.commit()
        self.committed = True

    def rollback(self) -> None:
        self.database.rollback()
        self.committed = False


class _PostgresQualityLease:
    def __init__(self, connection: _PostgresQualityConnection) -> None:
        self.connection = connection
        self.connection.committed = False

    def __enter__(self) -> _PostgresQualityConnection:
        return self.connection

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if not self.connection.committed:
            self.connection.rollback()
        return False


class _PostgresQualitySession:
    def __init__(self) -> None:
        self.connection = _PostgresQualityConnection()

    def open_connection(self) -> _PostgresQualityLease:
        return _PostgresQualityLease(self.connection)


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


def test_postgres_quality_store_commits_evidence() -> None:
    session = _PostgresQualitySession()
    store = PostgresCurationQualityStore(cast(Any, session))
    evidence = CurationQualityEvidence(
        run_id=uuid4(),
        action_id=uuid4(),
        operation="create_link",
        affected_memory_ids=[uuid4()],
        policy_version="policy",
        status="no_query",
        retrieval_utility_delta=0.75,
        acceptance_met=True,
        neutral_reason="trusted_query",
        created_at=datetime.now(UTC),
    )

    store.put_quality_evidence(evidence)
    store.put_quality_evidence(evidence)

    stored = store.list_quality_evidence()
    assert len(stored) == 1
    assert stored[0].action_id == evidence.action_id
    assert stored[0].retrieval_utility_delta == 0.75
    assert stored[0].acceptance_met is True
    assert stored[0].neutral_reason == "trusted_query"


def test_sqlite_quality_store_persists_acceptance_evidence(db_manager) -> None:
    run = _run()
    SQLiteCurationStore(db_manager).create_run(run)
    evidence = CurationQualityEvidence(
        run_id=run.run_id,
        action_id=uuid4(),
        operation="rewrite_memory",
        affected_memory_ids=[uuid4()],
        policy_version="policy",
        status="evaluated",
        retrieval_utility_delta=0.5,
        acceptance_met=False,
        neutral_reason=None,
        created_at=datetime.now(UTC),
    )
    store = SQLiteCurationQualityStore(db_manager)

    store.put_quality_evidence(evidence)
    store.put_quality_evidence(evidence)

    stored = store.list_quality_evidence()
    assert len(stored) == 1
    assert stored[0].retrieval_utility_delta == 0.5
    assert stored[0].acceptance_met is False
    assert stored[0].neutral_reason is None


def test_quality_hydration_defaults_missing_acceptance_columns() -> None:
    from mcp_memory.curation_quality_store import _from_postgres_row, _from_row

    run_id, action_id = uuid4(), uuid4()
    row = (
        str(run_id),
        str(action_id),
        "rewrite_memory",
        [],
        "policy",
        None,
        "no_query",
        [],
        [],
        None,
        None,
        None,
        None,
        datetime.now(UTC).isoformat(),
    )

    evidence = _from_postgres_row(row)

    assert evidence.retrieval_utility_delta is None
    assert evidence.acceptance_met is None
    assert evidence.neutral_reason is None

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE curation_quality_evidence (
            run_id TEXT, action_id TEXT, operation TEXT,
            affected_memory_ids_json TEXT, policy_version TEXT, query_id TEXT,
            status TEXT, before_ranked_ids_json TEXT, after_ranked_ids_json TEXT,
            retrieval_regression_count INTEGER, zero_result_change INTEGER,
            payload_size_change INTEGER, useful_work INTEGER, created_at TEXT
        )
        """
    )
    connection.execute(
        "INSERT INTO curation_quality_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            str(run_id),
            str(action_id),
            "rewrite_memory",
            "[]",
            "policy",
            None,
            "no_query",
            "[]",
            "[]",
            None,
            None,
            None,
            None,
            row[13],
        ),
    )

    sqlite_evidence = _from_row(
        connection.execute("SELECT * FROM curation_quality_evidence").fetchone()
    )

    assert sqlite_evidence.run_id == run_id
    assert sqlite_evidence.retrieval_utility_delta is None
    assert sqlite_evidence.acceptance_met is None
    assert sqlite_evidence.neutral_reason is None


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


def test_quality_sampler_accepts_content_improvement_without_search_query(db_manager) -> None:
    now = datetime.now(UTC)
    run = _run(created_at=now)
    SQLiteCurationStore(db_manager).create_run(run)
    memory_id = uuid4()
    event_id = uuid4()
    connection = db_manager.get_connection()
    connection.execute(
        """
        INSERT INTO memory_mutation_events (id, operation, actor_kind, status, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (str(event_id), "rewrite_memory", "curator", "applied", now.isoformat()),
    )
    connection.execute(
        """
        INSERT INTO memory_record_revisions (
            event_id, memory_id, role, before_exists, before_snapshot,
            after_exists, after_snapshot
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(event_id),
            str(memory_id),
            "target",
            1,
            json.dumps(
                {
                    "record": {
                        "title": "Auth",
                        "summary": "Added auth.",
                        "content": "Short note.",
                    }
                }
            ),
            1,
            json.dumps(
                {
                    "record": {
                        "title": "JWT client authentication requires coverage",
                        "summary": "JWT client authentication requires coverage in the API test suite.",
                        "content": "JWT coverage is required in src/auth.py; error ENG-741 confirmed the missing path.",
                    }
                }
            ),
        ),
    )
    connection.commit()
    receipt = _receipt(
        run.run_id,
        event_id=event_id,
        applied_at=now,
    ).model_copy(update={"affected_ids": [memory_id]})
    evidence = CurationQualitySampler(
        db_manager=db_manager,
        search=_Search([]),
        repository=SQLiteCurationQualityStore(db_manager),
        sample_rate=1.0,
    ).evaluate(run=run, receipts=[receipt])

    assert evidence[0].status == "content_evaluated"
    assert evidence[0].content_quality_improved is True
    assert (evidence[0].content_quality_delta or 0.0) > 0
    assert evidence[0].useful_work is True
    assert evidence[0].wave_status == "accepted"
    assert evidence[0].productive_mutation_count == 1


def test_quality_sampler_rejects_content_regression_without_search_query(db_manager) -> None:
    now = datetime.now(UTC)
    run = _run(created_at=now)
    SQLiteCurationStore(db_manager).create_run(run)
    memory_id = uuid4()
    event_id = uuid4()
    connection = db_manager.get_connection()
    connection.execute(
        """
        INSERT INTO memory_mutation_events (id, operation, actor_kind, status, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (str(event_id), "rewrite_memory", "curator", "applied", now.isoformat()),
    )
    connection.execute(
        """
        INSERT INTO memory_record_revisions (
            event_id, memory_id, role, before_exists, before_snapshot,
            after_exists, after_snapshot
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(event_id),
            str(memory_id),
            "target",
            1,
            json.dumps(
                {
                    "record": {
                        "title": "JWT client authentication requires coverage",
                        "summary": "JWT client authentication requires coverage in the API test suite.",
                        "content": "JWT coverage is required in src/auth.py; error ENG-741 confirmed the missing path.",
                    }
                }
            ),
            1,
            json.dumps(
                {
                    "record": {
                        "title": "Auth",
                        "summary": "Added auth.",
                        "content": "Short note.",
                    }
                }
            ),
        ),
    )
    connection.commit()
    receipt = _receipt(
        run.run_id,
        event_id=event_id,
        applied_at=now,
    ).model_copy(update={"affected_ids": [memory_id]})
    evidence = CurationQualitySampler(
        db_manager=db_manager,
        search=_Search([]),
        repository=SQLiteCurationQualityStore(db_manager),
        sample_rate=1.0,
    ).evaluate(run=run, receipts=[receipt])

    assert evidence[0].content_quality_improved is False
    assert evidence[0].useful_work is False
    assert evidence[0].wave_status == "rejected"
    assert evidence[0].productive_mutation_count == 0


def test_quality_sampler_accepts_coherent_multi_action_wave(db_manager) -> None:
    now = datetime.now(UTC)
    run = _run(created_at=now)
    SQLiteCurationStore(db_manager).create_run(run)
    memory_ids = [uuid4(), uuid4()]
    goal_id = uuid4()
    connection = db_manager.get_connection()
    for index, memory_id in enumerate(memory_ids, start=1):
        connection.execute(
            """
            INSERT INTO memories (id, title, content, type, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (str(memory_id), "title", "content", "fact", now.isoformat(), now.isoformat()),
        )
        connection.execute(
            """
            INSERT INTO memory_tool_events (
                invocation_id, caller_kind, event_kind, memory_id, query_text,
                result_rank, result_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "wave-query",
                "user",
                "search",
                str(memory_id),
                "wave retrieval",
                index,
                2,
                (now - timedelta(minutes=1)).isoformat(),
            ),
        )
    connection.commit()
    receipts = [
        _receipt(
            run.run_id,
            event_id=uuid4(),
            applied_at=now,
        ).model_copy(update={"affected_ids": [memory_id]})
        for memory_id in memory_ids
    ]
    sampler = CurationQualitySampler(
        db_manager=db_manager,
        search=_Search([_context(str(goal_id)), _context(str(memory_ids[0]))]),
        repository=SQLiteCurationQualityStore(db_manager),
        sample_rate=1.0,
    )

    evidence = sampler.evaluate(
        run=run,
        receipts=receipts,
        campaign_hypothesis=CampaignHypothesis(
            query="wave retrieval",
            expected_memory_ids=[goal_id],
            target_mode=CampaignTargetMode.TOP_K,
            minimum_improvement=0.1,
        ),
    )

    assert len(evidence) == 2
    assert {item.wave_id for item in evidence} != {None}
    assert {item.wave_status for item in evidence} == {"accepted"}
    assert {item.productive_mutation_count for item in evidence} == {2}
    assert evidence[0].wave_action_ids == evidence[1].wave_action_ids


def test_quality_sampler_rejects_wave_when_one_action_regresses(db_manager) -> None:
    now = datetime.now(UTC)
    run = _run(created_at=now)
    SQLiteCurationStore(db_manager).create_run(run)
    target, collateral = uuid4(), uuid4()
    connection = db_manager.get_connection()
    for memory_id in (target, collateral):
        connection.execute(
            """
            INSERT INTO memories (id, title, content, type, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (str(memory_id), "title", "content", "fact", now.isoformat(), now.isoformat()),
        )
        connection.execute(
            """
            INSERT INTO memory_tool_events (
                invocation_id, caller_kind, event_kind, memory_id, query_text,
                result_rank, result_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"query-{memory_id}",
                "user",
                "search",
                str(memory_id),
                "wave regression",
                1,
                1,
                (now - timedelta(minutes=1)).isoformat(),
            ),
        )
    connection.commit()
    receipts = [
        _receipt(run.run_id, event_id=uuid4(), applied_at=now).model_copy(
            update={"affected_ids": [memory_id]}
        )
        for memory_id in (target, collateral)
    ]
    sampler = CurationQualitySampler(
        db_manager=db_manager,
        search=_Search([_context(str(collateral))]),
        repository=SQLiteCurationQualityStore(db_manager),
        sample_rate=1.0,
    )

    evidence = sampler.evaluate(run=run, receipts=receipts)

    assert evidence
    assert {item.wave_status for item in evidence} == {"rejected"}
    assert {item.productive_mutation_count for item in evidence} == {0}


def test_quality_sampler_keeps_neutral_quality_evidence_non_escalating(db_manager) -> None:
    now = datetime.now(UTC)
    run = _run(created_at=now)
    curation_store = SQLiteCurationStore(db_manager)
    curation_store.create_run(run)
    memory_id = uuid4()
    receipt = _receipt(run.run_id, event_id=uuid4(), applied_at=now).model_copy(
        update={"affected_ids": [memory_id]}
    )
    initial = CurationCandidateState(
        memory_id=memory_id,
        disposition=CandidateDisposition.ACTIONED,
        last_disposition_reason="verified_receipt",
        escalation_count=2,
    )
    curation_store.put_candidate_state(initial)
    sampler = CurationQualitySampler(
        db_manager=db_manager,
        search=_Search([]),
        repository=SQLiteCurationQualityStore(db_manager),
        candidate_repository=curation_store,
        sample_rate=1.0,
    )

    evidence = sampler.evaluate(
        run=run,
        receipts=[receipt],
        campaign_hypothesis=CampaignHypothesis(query="important", minimum_improvement=0.5),
    )

    assert evidence[0].status == "no_query"
    assert evidence[0].acceptance_met is None
    assert curation_store.get_candidate_state(memory_id) == initial


def test_explicit_acceptance_failure_defers_verified_work() -> None:
    decision = CurationWorkItemService().decide(
        CurationRunOutcome.APPLIED,
        "applied",
        None,
        quality_evidence=[
            SimpleNamespace(acceptance_met=False),
        ],
        campaign_hypothesis=CampaignHypothesis(query="important", minimum_improvement=0.5),
    )

    assert decision.action is WorkItemAction.DEFER
    assert decision.reason_code == "quality_acceptance_failed"


@pytest.mark.parametrize(
    ("mode", "before", "after", "minimum", "expected"),
    [
        (CampaignTargetMode.RANK, ("noise", "wanted"), ("wanted",), 0.5, True),
        (CampaignTargetMode.RANK, ("wanted",), ("noise", "wanted"), 0.1, False),
        (CampaignTargetMode.TOP_K, ("noise", "noise-2", "wanted"), ("wanted",), 1.0, True),
        (CampaignTargetMode.TOP_K, ("wanted",), ("noise",), 0.1, False),
        (CampaignTargetMode.ZERO_RESULTS, (), ("wanted",), 1.0, True),
        (CampaignTargetMode.ZERO_RESULTS, ("wanted",), (), 0.1, False),
        (CampaignTargetMode.HEURISTIC, ("noise", "wanted"), ("wanted",), 0.5, True),
        (CampaignTargetMode.HEURISTIC, ("wanted",), ("noise", "wanted"), 0.1, False),
    ],
)
def test_campaign_acceptance_target_modes(
    mode: CampaignTargetMode,
    before: tuple[str, ...],
    after: tuple[str, ...],
    minimum: float,
    expected: bool,
) -> None:
    case = evaluate_query_replay(
        [
            ReplayCase(
                query_id="acceptance",
                intended_memory_ids=("wanted",),
                before=ReplaySnapshot(tuple(ReplayResult(value) for value in before)),
                after=ReplaySnapshot(tuple(ReplayResult(value) for value in after)),
                top_k=2,
            )
        ]
    ).cases[0]

    assert _acceptance_met(
        case,
        CampaignHypothesis(
            expected_memory_ids=[],
            target_mode=mode,
            minimum_improvement=minimum,
        ),
    ) is expected


def test_legacy_campaign_accepts_verified_work_without_quality_evidence() -> None:
    decision = CurationWorkItemService().decide(
        CurationRunOutcome.APPLIED,
        "applied",
        None,
    )

    assert decision.action is WorkItemAction.COMPLETE


def test_quality_sampler_ignores_maintenance_searches(db_manager) -> None:
    now = datetime.now(UTC)
    run = _run(created_at=now)
    SQLiteCurationStore(db_manager).create_run(run)
    memory_id = uuid4()
    connection = db_manager.get_connection()
    connection.execute(
        """
        INSERT INTO memories (
            id, title, content, type, status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (str(memory_id), "title", "content", "observation", "active", now.isoformat(), now.isoformat()),
    )
    connection.execute(
        """
        INSERT INTO memory_tool_events (
            invocation_id, caller_kind, event_kind, memory_id, query_text,
            result_rank, result_count, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "maintenance-query",
            "internal",
            "search",
            str(memory_id),
            "maintenance-only query",
            1,
            1,
            (now - timedelta(minutes=1)).isoformat(),
        ),
    )
    connection.commit()
    sampler = CurationQualitySampler(
        db_manager=db_manager,
        search=_Search([]),
        repository=SQLiteCurationQualityStore(db_manager),
        sample_rate=1.0,
    )

    evidence = sampler.evaluate(
        run=run,
        receipts=[
            _receipt(
                run.run_id,
                event_id=uuid4(),
                applied_at=now,
            ).model_copy(update={"affected_ids": [memory_id]}),
        ],
    )

    assert evidence[0].status == "no_query"


def test_quality_sampler_marks_link_work_structural_only(db_manager) -> None:
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
        receipts=[
            _receipt(
                run.run_id,
                event_id=uuid4(),
            ).model_copy(update={"operation": "create_link"})
        ],
    )

    assert evidence[0].status == "structural_only"
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
    noise_ids = [uuid4() for _ in range(6)]
    for value in (memory_id, *noise_ids):
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
    connection.executemany(
        """
        INSERT INTO memory_tool_events (
            invocation_id, caller_kind, event_kind, memory_id, query_text, result_rank,
            result_count, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "query-1",
                "external",
                "search",
                str(noise_id),
                "important",
                index + 1,
                7,
                (now - timedelta(minutes=1)).isoformat(),
            )
            for index, noise_id in enumerate(noise_ids)
        ]
        + [
            (
                "query-1",
                "external",
                "search",
                str(memory_id),
                "important",
                7,
                7,
                (now - timedelta(minutes=1)).isoformat(),
            )
        ],
    )
    connection.executemany(
        """
        INSERT INTO memory_tool_events (
            invocation_id, caller_kind, event_kind, memory_id, query_text, result_rank,
            result_count, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                f"noise-{index}",
                "maintenance",
                "search",
                str(noise_ids[0]),
                "unrelated",
                1,
                1,
                now.isoformat(),
            )
            for index in range(1001)
        ],
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

    evidence = sampler.evaluate(
        run=run,
        receipts=[receipt],
        campaign_hypothesis=CampaignHypothesis(
            query="important",
            expected_memory_ids=[memory_id],
            minimum_improvement=0.5,
        ),
    )

    assert evidence[0].status == "evaluated"
    assert evidence[0].query_id == "query-1"
    assert evidence[0].before_ranked_memory_ids == [*noise_ids, memory_id]
    assert evidence[0].after_ranked_memory_ids == [memory_id]
    assert evidence[0].useful_work is True
    assert evidence[0].acceptance_met is True
    assert (evidence[0].retrieval_utility_delta or 0.0) >= 0.5
    assert evidence[0].payload_size_change is None
    assert search.calls == ["important"]
    assert connection.execute("SELECT COUNT(*) FROM memory_tool_events").fetchone()[0] == before_event_count


def test_quality_sampler_recovers_explicit_zero_result_goal(db_manager) -> None:
    now = datetime.now(UTC)
    run = _run(created_at=now)
    SQLiteCurationStore(db_manager).create_run(run)
    memory_id = uuid4()
    receipt = _receipt(run.run_id, event_id=uuid4(), applied_at=now).model_copy(
        update={"affected_ids": [memory_id]}
    )
    connection = db_manager.get_connection()
    connection.execute(
        """
        INSERT INTO memory_tool_events (
            invocation_id, caller_kind, event_kind, memory_id, query_text,
            result_rank, result_count, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "zero-result-goal",
            "external",
            "search",
            None,
            "find the goal",
            None,
            0,
            (now - timedelta(minutes=1)).isoformat(),
        ),
    )
    connection.commit()
    search = _Search([_context(str(memory_id))])
    sampler = CurationQualitySampler(
        db_manager=db_manager,
        search=search,
        repository=SQLiteCurationQualityStore(db_manager),
        sample_rate=1.0,
    )

    evidence = sampler.evaluate(
        run=run,
        receipts=[receipt],
        campaign_hypothesis=CampaignHypothesis(
            query="find the goal",
            expected_memory_ids=[memory_id],
            target_mode=CampaignTargetMode.ZERO_RESULTS,
            minimum_improvement=1.0,
        ),
    )

    assert evidence[0].status == "evaluated"
    assert evidence[0].query_id == "zero-result-goal"
    assert evidence[0].zero_result_change == -1
    assert evidence[0].acceptance_met is True
    assert search.calls == ["find the goal"]


def test_quality_sampler_excludes_unrelated_explicit_zero_result_searches(db_manager) -> None:
    now = datetime.now(UTC)
    run = _run(created_at=now)
    SQLiteCurationStore(db_manager).create_run(run)
    memory_id = uuid4()
    receipt = _receipt(run.run_id, event_id=uuid4(), applied_at=now).model_copy(
        update={"affected_ids": [memory_id]}
    )
    connection = db_manager.get_connection()
    connection.execute(
        """
        INSERT INTO memory_tool_events (
            invocation_id, caller_kind, event_kind, memory_id, query_text,
            result_rank, result_count, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "unrelated-zero-result",
            "external",
            "search",
            None,
            "unrelated goal",
            None,
            0,
            (now - timedelta(minutes=1)).isoformat(),
        ),
    )
    connection.commit()
    search = _Search([_context(str(memory_id))])
    sampler = CurationQualitySampler(
        db_manager=db_manager,
        search=search,
        repository=SQLiteCurationQualityStore(db_manager),
        sample_rate=1.0,
    )

    evidence = sampler.evaluate(
        run=run,
        receipts=[receipt],
        campaign_hypothesis=CampaignHypothesis(
            query="find the goal",
            expected_memory_ids=[memory_id],
            target_mode=CampaignTargetMode.ZERO_RESULTS,
            minimum_improvement=1.0,
        ),
    )

    assert evidence[0].status == "no_query"
    assert evidence[0].neutral_reason == "no_trusted_query"
    assert search.calls in (None, [])


@pytest.mark.parametrize("mode", [CampaignTargetMode.RANK, CampaignTargetMode.TOP_K])
def test_quality_sampler_resolves_absent_explicit_goal_from_exact_query(
    db_manager,
    mode: CampaignTargetMode,
) -> None:
    now = datetime.now(UTC)
    run = _run(created_at=now)
    SQLiteCurationStore(db_manager).create_run(run)
    event_id = uuid4()
    memory_id = uuid4()
    noise_id = uuid4()
    receipt = _receipt(run.run_id, event_id=event_id, applied_at=now).model_copy(
        update={"affected_ids": [memory_id]}
    )
    connection = db_manager.get_connection()
    connection.executemany(
        """
        INSERT INTO memories (
            id, title, content, type, status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                str(value),
                "title",
                "content",
                "observation",
                "active",
                now.isoformat(),
                now.isoformat(),
            )
            for value in (memory_id, noise_id)
        ],
    )
    connection.executemany(
        """
        INSERT INTO memory_tool_events (
            invocation_id, caller_kind, event_kind, memory_id, query_text,
            result_rank, result_count, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "goal-query",
                "external",
                "search",
                str(noise_id),
                "find target",
                1,
                1,
                (now - timedelta(minutes=1)).isoformat(),
            ),
            (
                "unrelated-query",
                "external",
                "search",
                str(memory_id),
                "other query",
                1,
                1,
                (now - timedelta(minutes=1)).isoformat(),
            ),
        ],
    )
    connection.commit()
    search = _Search([_context(str(memory_id))])
    sampler = CurationQualitySampler(
        db_manager=db_manager,
        search=search,
        repository=SQLiteCurationQualityStore(db_manager),
        sample_rate=1.0,
    )

    evidence = sampler.evaluate(
        run=run,
        receipts=[receipt],
        campaign_hypothesis=CampaignHypothesis(
            query="find target",
            expected_memory_ids=[memory_id],
            target_mode=mode,
            top_k=1,
            minimum_improvement=1.0,
        ),
    )

    assert evidence[0].status == "evaluated"
    assert evidence[0].query_id == "goal-query"
    assert evidence[0].before_ranked_memory_ids == [noise_id]
    assert evidence[0].after_ranked_memory_ids == [memory_id]
    assert evidence[0].useful_work is True
    assert evidence[0].acceptance_met is True
    assert search.calls == ["find target"]


def test_quality_sampler_escalates_retrieval_regression(db_manager) -> None:
    now = datetime.now(UTC)
    run = _run(created_at=now)
    curation_store = SQLiteCurationStore(db_manager)
    curation_store.create_run(run)
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
    connection.execute(
        """
        INSERT INTO memories (
            id, title, content, type, status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (str(memory_id), "title", "content", "observation", "active", now.isoformat(), now.isoformat()),
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
        ("query-regression", "external", "search", str(memory_id), "important", 1, 1, (now - timedelta(minutes=1)).isoformat()),
    )
    connection.execute(
        """
        INSERT INTO memory_record_revisions (
            event_id, memory_id, role, before_exists, before_snapshot,
            after_exists, after_snapshot
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(event_id),
            str(memory_id),
            "target",
            1,
            json.dumps({"content": "before"}),
            1,
            json.dumps({"record": {"status": "active"}}),
        ),
    )
    connection.commit()
    sampler = CurationQualitySampler(
        db_manager=db_manager,
        search=_Search([]),
        repository=SQLiteCurationQualityStore(db_manager),
        candidate_repository=curation_store,
        sample_rate=1.0,
    )

    evidence = sampler.evaluate(run=run, receipts=[receipt])

    assert evidence[0].retrieval_regression_count == 1
    candidate = curation_store.get_candidate_state(memory_id)
    assert candidate is not None
    assert candidate.disposition.value == "escalated"
    assert candidate.last_disposition_reason == "retrieval_regression"


def test_quality_sampler_escalates_explicit_acceptance_failure(db_manager) -> None:
    now = datetime.now(UTC)
    run = _run(created_at=now)
    curation_store = SQLiteCurationStore(db_manager)
    curation_store.create_run(run)
    memory_id = uuid4()
    receipt = _receipt(run.run_id, event_id=uuid4(), applied_at=now).model_copy(
        update={"affected_ids": [memory_id]}
    )
    connection = db_manager.get_connection()
    connection.execute(
        """
        INSERT INTO memories (
            id, title, content, type, status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(memory_id),
            "Acceptance target",
            "Durable acceptance target.",
            "observation",
            "active",
            now.isoformat(),
            now.isoformat(),
        ),
    )
    connection.execute(
        """
        INSERT INTO memory_tool_events (
            invocation_id, caller_kind, event_kind, memory_id, query_text,
            result_rank, result_count, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "acceptance-query",
            "external",
            "search",
            str(memory_id),
            "important",
            1,
            1,
            (now - timedelta(minutes=1)).isoformat(),
        ),
    )
    connection.commit()
    sampler = CurationQualitySampler(
        db_manager=db_manager,
        search=_Search([_context(str(memory_id))]),
        repository=SQLiteCurationQualityStore(db_manager),
        candidate_repository=curation_store,
        sample_rate=1.0,
    )

    evidence = sampler.evaluate(
        run=run,
        receipts=[receipt],
        campaign_hypothesis=CampaignHypothesis(
            query="important",
            expected_memory_ids=[memory_id],
            minimum_improvement=0.5,
        ),
    )

    assert evidence[0].status == "evaluated"
    assert evidence[0].acceptance_met is False
    assert evidence[0].retrieval_regression_count == 0
    candidate = curation_store.get_candidate_state(memory_id)
    assert candidate is not None
    assert candidate.disposition.value == "escalated"
    assert candidate.last_disposition_reason == "acceptance_not_met"
    assert candidate.coverage_evidence_json["quality_regression"]["acceptance_met"] is False
    assert candidate.coverage_evidence_json["quality_regression"]["neutral_reason"] is None


def test_quality_sampler_excludes_archived_merge_sources(db_manager) -> None:
    now = datetime.now(UTC)
    run = _run(created_at=now)
    SQLiteCurationStore(db_manager).create_run(run)
    event_id = uuid4()
    active_id = uuid4()
    archived_id = uuid4()
    connection = db_manager.get_connection()
    connection.executemany(
        """
        INSERT INTO memories (
            id, title, content, type, status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                str(memory_id),
                "title",
                "content",
                "observation",
                "active",
                now.isoformat(),
                now.isoformat(),
            )
            for memory_id in (active_id, archived_id)
        ],
    )
    connection.execute(
        """
        INSERT INTO memory_mutation_events (
            id, operation, actor_kind, curation_run_id, status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (str(event_id), "merge_memories", "maintenance", str(run.run_id), "applied", now.isoformat()),
    )
    connection.executemany(
        """
        INSERT INTO memory_record_revisions (
            event_id, memory_id, role, before_exists, before_snapshot,
            after_exists, after_snapshot
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                str(event_id),
                str(active_id),
                "target",
                1,
                "{}",
                1,
                json.dumps({"record": {"status": "active"}}),
            ),
            (
                str(event_id),
                str(archived_id),
                "target",
                1,
                "{}",
                1,
                json.dumps({"record": {"status": "archived"}}),
            ),
        ],
    )
    connection.commit()
    receipt = CurationActionReceipt(
        run_id=run.run_id,
        action_id=uuid4(),
        operation="merge_memories",
        affected_ids=[active_id, archived_id],
        status=CurationReceiptState.VERIFIED,
        mutation_event_id=event_id,
        intent_hash="intent",
        applied_at=now,
    )
    sampler = CurationQualitySampler(
        db_manager=db_manager,
        search=_Search([]),
        repository=SQLiteCurationQualityStore(db_manager),
        sample_rate=1.0,
    )

    assert sampler._intended_memory_ids(receipt) == [active_id]


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
    SQLiteCurationQualityStore(db_manager).put_quality_evidence(
        CurationQualityEvidence(
            run_id=run.run_id,
            action_id=uuid4(),
            operation="create_link",
            affected_memory_ids=receipt.affected_ids,
            policy_version=run.policy_version,
            status="structural_only",
            created_at=now,
        )
    )

    metrics = build_curation_metrics(db_manager, window_hours=24, now=now.timestamp())

    assert metrics.verified_yield == 0.0
    assert metrics.retrieval_quality.sampled_action_count == 2
    assert metrics.retrieval_quality.no_query_action_count == 1
    assert metrics.retrieval_quality.structural_only_action_count == 1
