from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest

from mcp_memory.core.curation_quality import CurationQualitySampler
from mcp_memory.core.curation_quality_clusters import evaluate_cluster_utility
from mcp_memory.core.curation_quality_consistency import assess_replay_consistency
from mcp_memory.core.curation_quality_inputs import (
    CurationQualityMutation,
    direct_action_id,
    mutations_from_direct_evidence,
)
from mcp_memory.core.curation_quality_policy import (
    QualityOutcome,
    classify_quality_outcome,
    should_escalate_quality,
)
from mcp_memory.core.curation_quality_structural import evaluate_structural_mutation
from mcp_memory.core.curation_shadow import (
    _direct_quality_run,
    _evaluate_direct_quality,
    _persist_direct_quality_run,
)
from mcp_memory.core.ports.tasks import TaskRecord
from mcp_memory.context import ApplicationContext
from mcp_memory.curation_quality_store import SQLiteCurationQualityStore
from mcp_memory.curation_store import SQLiteCurationStore


pytestmark = pytest.mark.small


class _Search:
    def __init__(self, records=(), epochs=None) -> None:
        self.records = list(records)
        self.epochs = dict(epochs or {})
        self.calls: list[str] = []

    def search_memories_for_maintenance(self, query: str, *, limit: int = 50):
        self.calls.append(query)
        return self.records[:limit]

    def get_search_epochs(self):
        return self.epochs


class _EvidenceRepository:
    def __init__(self) -> None:
        self.values = []

    def put_quality_evidence(self, evidence):
        self.values.append(evidence)
        return evidence


class _RacingCurationRepository:
    def __init__(self) -> None:
        self.run = None

    def get_run(self, run_id):
        if self.run is not None and self.run.run_id == run_id:
            return self.run
        return None

    def create_run(self, run):
        self.run = run
        raise RuntimeError("simulated concurrent insert")


def _run():
    return SimpleNamespace(
        run_id=uuid4(),
        policy_version="test",
        frontier_key="direct:test",
        selector_strategy="direct",
    )


def _task() -> TaskRecord:
    return TaskRecord(
        id="task-1",
        task_name="memory_curator",
        data={},
        workspace_id=None,
        status="running",
        priority=100,
        retries_count=0,
        max_retries=3,
        created_at=0.0,
        updated_at=0.0,
        available_at=0.0,
        claimed_at=None,
        started_at=None,
        completed_at=None,
        last_error=None,
        execution_epoch=2,
    )


def _direct_evidence() -> SimpleNamespace:
    memory_id = uuid4()
    return SimpleNamespace(
        evidence_id="evidence-1",
        operation="rewrite_memory",
        outcome="applied_verified",
        completed_at=datetime.now(UTC),
        payload={},
        deltas=(
            SimpleNamespace(
                kind="record",
                entity_id=str(memory_id),
                before_exists=True,
                after_exists=True,
                transition="updated",
                before_revision="before",
                after_revision="after",
                snapshot={"id": str(memory_id)},
            ),
        ),
    )


def test_direct_quality_uses_durable_repository(db_manager) -> None:
    """Direct quality observations survive through the SQLite repository."""
    quality_store = SQLiteCurationQualityStore(db_manager)
    ctx = ApplicationContext(
        db_manager=db_manager,
        relational_search=cast(Any, _Search()),
        curation=SQLiteCurationStore(db_manager),
        curation_quality=quality_store,
    )
    task = _task()
    run = _direct_quality_run(task)

    evaluation = _evaluate_direct_quality(
        ctx,
        task,
        [_direct_evidence()],
        campaign_hypothesis=None,
    )

    assert evaluation.status == "recorded"
    assert len(evaluation.evidence) == 1
    persisted = SQLiteCurationStore(db_manager).get_run(run.run_id)
    assert persisted is not None
    assert persisted.state.value == "terminal"
    stored = quality_store.list_quality_evidence(run_id=_direct_quality_run(task).run_id)
    assert len(stored) == 1
    assert stored[0].action_id == evaluation.evidence[0].action_id


def test_direct_quality_reports_missing_repository(db_manager) -> None:
    """Missing quality storage is explicit instead of empty success."""
    ctx = ApplicationContext(db_manager=db_manager, relational_search=cast(Any, _Search()))

    evaluation = _evaluate_direct_quality(
        ctx,
        _task(),
        [_direct_evidence()],
        campaign_hypothesis=None,
    )

    assert evaluation.evidence == ()
    assert evaluation.status == "unavailable"
    assert evaluation.reason == "quality_repository_unavailable"


def test_direct_quality_run_recovers_from_concurrent_insert() -> None:
    repository = _RacingCurationRepository()
    run = _direct_quality_run(_task())
    context = ApplicationContext(curation=repository)

    persisted = _persist_direct_quality_run(context, run)

    assert persisted == run


def test_direct_evidence_bridges_to_stable_action_identity() -> None:
    """Derive one stable action ID from the persisted direct evidence ID."""
    mutation = mutations_from_direct_evidence(_direct_evidence())

    assert mutation.action_identity_required is True
    assert mutation.evidence_id == "evidence-1"
    assert mutation.mutation_id == direct_action_id("evidence-1")


def test_action_identity_mismatch_is_unobserved(db_manager) -> None:
    """Reject quality credit when an action claims the wrong direct evidence ID."""
    memory_id = uuid4()
    mutation = CurationQualityMutation(
        mutation_id=uuid4(),
        operation="rewrite_memory",
        affected_memory_ids=(memory_id,),
        applied=True,
        verified=True,
        evidence_id="evidence-1",
        action_identity_required=True,
    )
    repository = _EvidenceRepository()
    evidence = CurationQualitySampler(
        db_manager=db_manager,
        search=_Search(),
        repository=cast(Any, repository),
        sample_rate=1.0,
    ).evaluate(run=cast(Any, _run()), mutations=[mutation])

    assert evidence[0].status == QualityOutcome.UNOBSERVED.value
    assert evidence[0].neutral_reason == "action_identity_mismatch"
    assert evidence[0].productive_mutation_count == 0


def test_missing_action_identity_is_unobserved(db_manager) -> None:
    """Reject quality credit when direct evidence has no durable identity."""
    memory_id = uuid4()
    mutation = CurationQualityMutation(
        mutation_id=uuid4(),
        operation="rewrite_memory",
        affected_memory_ids=(memory_id,),
        applied=True,
        verified=True,
        action_identity_required=True,
    )
    repository = _EvidenceRepository()
    evidence = CurationQualitySampler(
        db_manager=db_manager,
        search=_Search(),
        repository=cast(Any, repository),
        sample_rate=1.0,
    ).evaluate(run=cast(Any, _run()), mutations=[mutation])

    assert evidence[0].status == QualityOutcome.UNOBSERVED.value
    assert evidence[0].neutral_reason == "action_identity_missing"
    assert evidence[0].productive_mutation_count == 0


def _mutation(*, outcome: str = "applied_verified", operation: str = "rewrite_memory"):
    memory_id = uuid4()
    return CurationQualityMutation(
        mutation_id=uuid4(),
        operation=operation,
        affected_memory_ids=(memory_id,),
        applied=True,
        verified=outcome == "applied_verified",
        evidence_id="direct-evidence",
        applied_at=datetime.now(UTC),
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"index_lag": True}, "index_lag"),
        ({"unrelated_write_count": 1}, "unrelated_writes"),
        ({"write_pollution": True}, "write_pollution"),
        ({"replay_complete": False}, "incomplete_replay"),
    ],
)
def test_consistency_faults_are_explicit(kwargs, expected: str) -> None:
    parameters = {
        "before_context": {"status": "active"},
        "after_context": {"status": "active"},
        "before_epochs": {"keyword": 4},
        "after_epochs": {"keyword": 3},
        "replay_complete": True,
    }
    parameters.update(kwargs)
    assessment = assess_replay_consistency(
        **parameters,
    )

    assert assessment.verified is False
    assert expected in assessment.flags


def test_cluster_utility_rewards_duplicate_removal() -> None:
    utility = evaluate_cluster_utility(
        before_ids=("canonical", "duplicate", "other"),
        after_ids=("canonical", "other"),
        clusters=(("canonical", "duplicate"),),
        top_k=3,
    )

    assert utility.duplicate_density_after < utility.duplicate_density_before
    assert utility.delta > 0


@pytest.mark.parametrize(
    ("operation", "deltas", "verified"),
    [
        ("create_link", ({"kind": "link", "after_exists": True},), True),
        ("remove_link", ({"kind": "link", "after_exists": False},), True),
        ("archive_memory", ({"kind": "record", "transition": "archived", "after_exists": True},), True),
        ("delete_memory", ({"kind": "record", "after_exists": False},), True),
        ("reconcile_dangling_links", ({"kind": "link", "after_exists": False},), True),
        ("delete_memory", ({"kind": "record", "after_exists": True},), False),
    ],
)
def test_structural_postconditions(operation, deltas, verified) -> None:
    assert evaluate_structural_mutation(operation, deltas).verified is verified


def test_missing_trusted_query_is_neutral_without_productive_credit(db_manager) -> None:
    mutation = _mutation()
    repository = _EvidenceRepository()
    sampler = CurationQualitySampler(
        db_manager=db_manager,
        search=_Search(),
        repository=cast(Any, repository),
        sample_rate=1.0,
    )

    evidence = sampler.evaluate(run=cast(Any, _run()), mutations=[mutation])

    assert evidence[0].status == QualityOutcome.UNOBSERVED.value
    assert evidence[0].neutral_reason in {"incomplete_replay", "no_trusted_query"}
    assert evidence[0].productive_mutation_count == 0
    assert repository.values == list(evidence)


def test_unverified_direct_evidence_cannot_escalate(db_manager) -> None:
    mutation = _mutation(outcome="applied_unverified")
    candidate_repository = SimpleNamespace(
        get_candidate_state=lambda _memory_id: None,
        put_candidate_state=lambda _state: pytest.fail("unverified evidence escalated"),
    )
    sampler = CurationQualitySampler(
        db_manager=db_manager,
        search=_Search(),
        repository=cast(Any, _EvidenceRepository()),
        candidate_repository=cast(Any, candidate_repository),
        sample_rate=1.0,
    )

    evidence = sampler.evaluate(run=cast(Any, _run()), mutations=[mutation])

    assert evidence[0].status == QualityOutcome.UNVERIFIED.value
    assert evidence[0].productive_mutation_count == 0
    assert should_escalate_quality(QualityOutcome.UNVERIFIED, mutation_verified=False) is False


def test_synthetic_incomplete_replay_is_unobserved_without_retrieval_credit(db_manager) -> None:
    """Keep synthetic probes from turning incomplete replay into productive quality."""
    memory_id = uuid4()
    mutation = CurationQualityMutation(
        mutation_id=uuid4(),
        operation="rewrite_memory",
        affected_memory_ids=(memory_id,),
        verified=True,
        evidence_id="direct-evidence",
        after_entities={
            str(memory_id): {
                "title": "Synthetic probe target",
                "summary": "A replay-free quality target.",
            }
        },
    )
    repository = _EvidenceRepository()
    evidence = CurationQualitySampler(
        db_manager=db_manager,
        search=_Search(),
        repository=cast(Any, repository),
        sample_rate=1.0,
    ).evaluate(run=cast(Any, _run()), mutations=[mutation])

    assert evidence[0].status == QualityOutcome.UNOBSERVED.value
    assert evidence[0].neutral_reason == "incomplete_replay"
    assert evidence[0].retrieval_utility_delta is None
    assert evidence[0].productive_mutation_count == 0


@pytest.mark.parametrize(
    ("content_delta", "expected"),
    [
        (0.25, QualityOutcome.VERIFIED_ONLY),
        (0.0, QualityOutcome.NEUTRAL),
        (-0.25, QualityOutcome.REGRESSED),
        (None, QualityOutcome.UNOBSERVED),
    ],
)
def test_content_quality_without_trusted_replay_has_no_productive_retrieval_credit(
    content_delta: float | None,
    expected: QualityOutcome,
) -> None:
    """Apply the explicit disposition policy when retrieval observation is absent."""
    assert classify_quality_outcome(
        mutation_verified=True,
        query_trusted=False,
        consistency_verified=True,
        structural_only=False,
        utility_delta=None,
        quality_observed=False,
        content_quality_delta=content_delta,
    ) is expected


def test_trusted_direct_query_can_be_productive(db_manager) -> None:
    now = datetime.now(UTC)
    memory_id = uuid4()
    other_id = uuid4()
    connection = db_manager.get_connection()
    connection.executemany(
        """
        INSERT INTO memories (id, title, content, type, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (str(memory_id), "target", "target", "fact", now.isoformat(), now.isoformat()),
            (str(other_id), "other", "other", "fact", now.isoformat(), now.isoformat()),
        ],
    )
    connection.execute(
        """
        INSERT INTO memory_tool_events (
            invocation_id, caller_kind, event_kind, memory_id, query_text,
            result_rank, result_count, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "direct-query",
            "operator",
            "search",
            str(memory_id),
            "durable target",
            2,
            2,
            (now - timedelta(minutes=1)).timestamp(),
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
            "direct-query",
            "operator",
            "search",
            str(other_id),
            "durable target",
            1,
            2,
            (now - timedelta(minutes=1)).timestamp(),
        ),
    )
    connection.commit()
    mutation = CurationQualityMutation(
        mutation_id=uuid4(),
        operation="rewrite_memory",
        affected_memory_ids=(memory_id,),
        applied=True,
        verified=True,
        evidence_id="direct-evidence",
        applied_at=now,
    )
    repository = _EvidenceRepository()
    sampler = CurationQualitySampler(
        db_manager=db_manager,
        search=_Search([SimpleNamespace(record=SimpleNamespace(id=memory_id))]),
        repository=cast(Any, repository),
        sample_rate=1.0,
    )

    evidence = sampler.evaluate(run=cast(Any, _run()), mutations=[mutation])

    assert evidence[0].status == QualityOutcome.PRODUCTIVE.value
    assert evidence[0].engagement_evidence["query_provenance"] == "real_user_search"
    assert evidence[0].retrieval_utility_delta is not None
