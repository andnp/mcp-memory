"""Advisory strategy selection from persisted curator feedback."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from mcp_memory.core.ports.curation import CandidateDisposition, CurationCandidateState
from mcp_memory.core.sampling import QUALITY_SIGNAL_STRATEGY

CuratorFeedbackReason = Literal[
    "explicit_strategy_preserved",
    "quality_feedback_escalations_found",
    "candidate_ledger_unavailable",
    "no_quality_feedback_evidence",
]

_QUALITY_FEEDBACK_REASONS = frozenset(
    {"retrieval_regression", "zero_result_regression", "acceptance_not_met"}
)


class CandidateStateReader(Protocol):
    def list_candidate_states(
        self,
        *,
        disposition: CandidateDisposition | None = None,
        limit: int = 100,
    ) -> Sequence[CurationCandidateState]: ...


@dataclass(frozen=True, slots=True)
class CuratorFeedbackDecision:
    """Bounded advisory output for the next curator candidate acquisition."""

    strategy: str | None
    reason: CuratorFeedbackReason
    escalation_count: int = 0
    evidence_reasons: tuple[str, ...] = ()


def select_curator_feedback_strategy(
    curation: CandidateStateReader | None,
    requested_strategy: str | None,
) -> CuratorFeedbackDecision:
    """Choose quality-signal only when persisted escalation evidence supports it."""
    if requested_strategy:
        return CuratorFeedbackDecision(
            strategy=requested_strategy,
            reason="explicit_strategy_preserved",
        )

    if curation is None:
        return CuratorFeedbackDecision(strategy=None, reason="candidate_ledger_unavailable")

    try:
        states = curation.list_candidate_states(
            disposition=CandidateDisposition.ESCALATED,
            limit=100,
        )
    except Exception:
        return CuratorFeedbackDecision(strategy=None, reason="candidate_ledger_unavailable")

    feedback_states = [
        state
        for state in states
        if state.last_disposition_reason in _QUALITY_FEEDBACK_REASONS
    ]
    if not feedback_states:
        return CuratorFeedbackDecision(strategy=None, reason="no_quality_feedback_evidence")

    evidence_reasons = tuple(
        sorted({state.last_disposition_reason for state in feedback_states if state.last_disposition_reason})
    )
    return CuratorFeedbackDecision(
        strategy=QUALITY_SIGNAL_STRATEGY,
        reason="quality_feedback_escalations_found",
        escalation_count=len(feedback_states),
        evidence_reasons=evidence_reasons,
    )
