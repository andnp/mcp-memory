from uuid import uuid4

import pytest

from mcp_memory.core.curation_feedback_controller import select_curator_feedback_strategy
from mcp_memory.curation_store import CandidateDisposition, CurationCandidateState

pytestmark = pytest.mark.small


class _CandidateLedger:
    def __init__(self, states: list[CurationCandidateState]) -> None:
        self.states = states

    def list_candidate_states(self, **_kwargs: object) -> list[CurationCandidateState]:
        return self.states


def _escalated_state(reason: str) -> CurationCandidateState:
    return CurationCandidateState(
        memory_id=uuid4(),
        disposition=CandidateDisposition.ESCALATED,
        last_disposition_reason=reason,
    )


def test_explicit_strategy_precedes_persisted_quality_feedback() -> None:
    """An operator-requested strategy remains unchanged despite quality evidence."""
    decision = select_curator_feedback_strategy(
        _CandidateLedger([_escalated_state("retrieval_regression")]),
        "semantic",
    )

    assert decision.strategy == "semantic"
    assert decision.reason == "explicit_strategy_preserved"


def test_quality_feedback_escalations_select_quality_signal() -> None:
    """Persisted quality escalations advise quality-signal for the next acquisition."""
    decision = select_curator_feedback_strategy(
        _CandidateLedger(
            [
                _escalated_state("retrieval_regression"),
                _escalated_state("acceptance_not_met"),
            ]
        ),
        None,
    )

    assert decision.strategy == "quality-signal"
    assert decision.reason == "quality_feedback_escalations_found"
    assert decision.escalation_count == 2
    assert decision.evidence_reasons == ("acceptance_not_met", "retrieval_regression")


def test_unavailable_candidate_ledger_leaves_strategy_unset() -> None:
    """Missing candidate-ledger access produces an advisory no-op."""
    decision = select_curator_feedback_strategy(None, None)

    assert decision.strategy is None
    assert decision.reason == "candidate_ledger_unavailable"
    assert decision.escalation_count == 0
