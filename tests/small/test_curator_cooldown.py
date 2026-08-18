from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.core.task_handlers.curator_support import (
    _quality_feedback_candidates,
    curator_candidate_revision_token,
    curator_quality_feedback,
    filter_curator_candidates,
    record_curator_no_op_dispositions,
)
from mcp_memory.curation_store import CandidateDisposition, CurationCandidateState, SQLiteCurationStore
from mcp_memory.mutation_history import MutationActorKind

pytestmark = pytest.mark.small


def _record(memory_id: UUID) -> SimpleNamespace:
    return SimpleNamespace(
        id=str(memory_id),
        title="A durable title",
        content="A durable memory.",
        summary="A durable summary.",
        type="fact",
        status="active",
        tags=["durable"],
        workspace_ids=["global"],
        metadata={},
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        read_count=0,
        last_surfaced_at=None,
    )


class _Repository:
    def __init__(self) -> None:
        self.links: dict[str, list[SimpleNamespace]] = {}

    def get_links(self, memory_id: str, *, direction: str, link_type: str | None = None):
        del link_type
        if direction == "outgoing":
            return self.links.get(memory_id, [])
        return [
            link
            for links in self.links.values()
            for link in links
            if link.target_id == memory_id
        ]


def _context(db_manager, repository: _Repository, history=None) -> ApplicationContext:
    return ApplicationContext(
        repository=repository,
        curation=SQLiteCurationStore(db_manager),
        mutation_history=history,
    )


def test_noop_cooldown_is_per_record_not_frontier_composition(db_manager) -> None:
    repository = _Repository()
    first = _record(uuid4())
    second = _record(uuid4())
    ctx = _context(db_manager, repository)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    ctx.curation.put_candidate_state(
        CurationCandidateState(
            memory_id=UUID(first.id),
            last_observed_revision_token=curator_candidate_revision_token(ctx, first),
            disposition=CandidateDisposition.COOLDOWN,
            consecutive_no_op_count=2,
            cooldown_until=now + timedelta(hours=1),
        )
    )

    assert filter_curator_candidates(ctx, [first, second], now=now) == [second]
    assert filter_curator_candidates(ctx, [second, first], now=now) == [second]


def test_noop_cooldown_is_bounded_and_exact_expiry_is_eligible(db_manager) -> None:
    """Allow a no-op candidate to re-enter selection at exact expiry.

    Repeated no-ops back off without making a candidate permanently ineligible.
    """
    repository = _Repository()
    record = _record(uuid4())
    ctx = _context(db_manager, repository)
    now = datetime(2026, 1, 1, tzinfo=UTC)

    record_curator_no_op_dispositions(ctx, [record], now=now)
    first = ctx.curation.get_candidate_state(UUID(record.id))
    assert first is not None
    assert first.disposition is CandidateDisposition.COOLDOWN
    assert first.cooldown_until == now + timedelta(hours=1)

    assert filter_curator_candidates(ctx, [record], now=first.cooldown_until) == [record]

    record_curator_no_op_dispositions(ctx, [record], now=now + timedelta(hours=1))
    second = ctx.curation.get_candidate_state(UUID(record.id))
    assert second is not None
    assert second.consecutive_no_op_count == 2
    assert second.cooldown_until == now + timedelta(hours=3)


@pytest.mark.parametrize("change", ["revision", "adjacency"])
def test_revision_or_adjacency_change_resets_noop_eligibility(db_manager, change: str) -> None:
    repository = _Repository()
    record = _record(uuid4())
    ctx = _context(db_manager, repository)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    ctx.curation.put_candidate_state(
        CurationCandidateState(
            memory_id=UUID(record.id),
            last_observed_revision_token=curator_candidate_revision_token(ctx, record),
            disposition=CandidateDisposition.COOLDOWN,
            consecutive_no_op_count=3,
            cooldown_until=now + timedelta(hours=1),
        )
    )
    if change == "revision":
        record.summary = "A newly supported summary."
    else:
        adjacent = _record(uuid4())
        repository.links[record.id] = [
            SimpleNamespace(
                source_id=record.id,
                target_id=adjacent.id,
                link_type="DEPENDS_ON",
                context=None,
            )
        ]

    assert filter_curator_candidates(ctx, [record], now=now) == [record]
    state = ctx.curation.get_candidate_state(UUID(record.id))
    assert state is not None
    assert state.disposition is CandidateDisposition.PENDING
    assert state.consecutive_no_op_count == 0
    assert state.cooldown_until is None


def test_candidate_cas_loss_uses_authoritative_winner(db_manager, monkeypatch) -> None:
    repository = _Repository()
    record = _record(uuid4())
    ctx = _context(db_manager, repository)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    original = CurationCandidateState(
        memory_id=UUID(record.id),
        last_observed_revision_token="stale-token",
        disposition=CandidateDisposition.COOLDOWN,
        cooldown_until=now + timedelta(hours=1),
    )
    winner = original.model_copy(
        update={"last_observed_revision_token": curator_candidate_revision_token(ctx, record)}
    )
    ctx.curation.put_candidate_state(original)

    def lose_race(memory_id, expected_token, state):
        del expected_token, state
        ctx.curation.put_candidate_state(winner)
        return None

    monkeypatch.setattr(ctx.curation, "compare_and_set_candidate_state", lose_race)

    assert filter_curator_candidates(ctx, [record], now=now) == []


def test_quality_feedback_escalation_survives_revision_change(db_manager) -> None:
    repository = _Repository()
    record = _record(uuid4())
    ctx = _context(db_manager, repository)
    state = CurationCandidateState(
        memory_id=UUID(record.id),
        last_observed_revision_token="stale-token",
        disposition=CandidateDisposition.ESCALATED,
        last_disposition_reason="retrieval_regression",
        escalation_count=1,
    )
    ctx.curation.put_candidate_state(state)

    assert filter_curator_candidates(ctx, [record]) == [record]
    updated = ctx.curation.get_candidate_state(UUID(record.id))
    assert updated is not None
    assert updated.disposition is CandidateDisposition.ESCALATED
    assert updated.last_disposition_reason == "retrieval_regression"
    assert updated.last_observed_revision_token == curator_candidate_revision_token(ctx, record)


def test_quality_feedback_candidates_are_prioritized(db_manager) -> None:
    repository = _Repository()
    first = _record(uuid4())
    second = _record(uuid4())
    ctx = _context(db_manager, repository)
    ctx.curation.put_candidate_state(
        CurationCandidateState(
            memory_id=UUID(second.id),
            disposition=CandidateDisposition.ESCALATED,
            last_disposition_reason="retrieval_regression",
            escalation_count=2,
        )
    )

    assert _quality_feedback_candidates(ctx, [first, second]) == [second]
    feedback = curator_quality_feedback(ctx, second)
    assert feedback is not None
    assert feedback["reason"] == "retrieval_regression"
    assert feedback["escalation_count"] == 2


def test_failed_quality_repairs_cool_down_until_quality_route(db_manager) -> None:
    repository = _Repository()
    record = _record(uuid4())
    ctx = _context(db_manager, repository)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    ctx.curation.put_candidate_state(
        CurationCandidateState(
            memory_id=UUID(record.id),
            last_observed_revision_token=curator_candidate_revision_token(ctx, record),
            disposition=CandidateDisposition.ESCALATED,
            cooldown_until=now + timedelta(hours=6),
            last_disposition_reason="retrieval_regression",
        )
    )

    assert filter_curator_candidates(ctx, [record], now=now) == []
    assert filter_curator_candidates(
        ctx,
        [record],
        now=now,
        allow_quality_feedback=True,
    ) == [record]


def test_acceptance_failure_is_quality_feedback_for_retry(db_manager) -> None:
    record = _record(uuid4())
    ctx = _context(db_manager, _Repository())
    ctx.curation.put_candidate_state(
        CurationCandidateState(
            memory_id=UUID(record.id),
            disposition=CandidateDisposition.ESCALATED,
            last_disposition_reason="acceptance_not_met",
            escalation_count=1,
            coverage_evidence_json={
                "quality_regression": {
                    "acceptance_met": False,
                    "neutral_reason": None,
                }
            },
        )
    )

    assert _quality_feedback_candidates(ctx, [record]) == [record]
    feedback = curator_quality_feedback(ctx, record)
    assert feedback is not None
    assert feedback["reason"] == "acceptance_not_met"
    assert feedback["acceptance_met"] is False
    assert feedback["neutral_reason"] is None


@pytest.mark.parametrize("neutral_reason", ["no_query", "irrelevant_query", "inconclusive"])
def test_neutral_quality_evidence_is_not_feedback(db_manager, neutral_reason: str) -> None:
    record = _record(uuid4())
    ctx = _context(db_manager, _Repository())
    ctx.curation.put_candidate_state(
        CurationCandidateState(
            memory_id=UUID(record.id),
            disposition=CandidateDisposition.ESCALATED,
            last_disposition_reason="acceptance_not_met",
            coverage_evidence_json={
                "quality_regression": {
                    "acceptance_met": None,
                    "neutral_reason": neutral_reason,
                }
            },
        )
    )

    assert curator_quality_feedback(ctx, record) is None


def test_recent_human_and_cross_family_edits_stabilize_candidates(db_manager) -> None:
    repository = _Repository()
    record = _record(uuid4())
    now = datetime(2026, 1, 1, tzinfo=UTC)
    history = SimpleNamespace(
        list_events=lambda *, memory_id, limit: [
            SimpleNamespace(
                created_at=now - timedelta(minutes=5),
                actor_kind=MutationActorKind.USER,
                family="curator",
            ),
            SimpleNamespace(
                created_at=now - timedelta(minutes=10),
                actor_kind=MutationActorKind.MAINTENANCE,
                family="taxonomist",
            ),
        ]
    )
    ctx = _context(db_manager, repository, history)

    assert filter_curator_candidates(ctx, [record], now=now) == []
    assert filter_curator_candidates(
        ctx,
        [record],
        now=now + timedelta(seconds=3601),
    ) == [record]
