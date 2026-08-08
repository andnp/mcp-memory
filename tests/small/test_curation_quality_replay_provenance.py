from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest

from mcp_memory.core.curation_quality import CurationQualitySampler
from mcp_memory.core.curation_quality_inputs import CurationQualityMutation
from mcp_memory.core.curation_quality_provenance import QualityQuery, QueryProvenance
from mcp_memory.curation_store import CurationRun, CurationRunOutcome, CurationRunState


pytestmark = pytest.mark.small


class _Search:
    def __init__(self, memory_id: str) -> None:
        self._memory_id = memory_id

    def search_memories_for_maintenance(self, query: str, *, limit: int = 50) -> list[object]:
        return [SimpleNamespace(record=SimpleNamespace(id=self._memory_id))]

    def get_search_epochs(self) -> dict[str, int]:
        return {"records": 2}


def _run() -> CurationRun:
    return CurationRun(
        run_id=uuid4(),
        frontier_key="replay-provenance",
        context_fingerprint="test",
        state=CurationRunState.TERMINAL,
        outcome=CurationRunOutcome.APPLIED,
    )


def _mutation(memory_id, *, after_context: dict[str, object] | None = None) -> CurationQualityMutation:
    consistency: dict[str, object] = {}
    if after_context is not None:
        consistency["after_query_context"] = after_context
    return CurationQualityMutation(
        mutation_id=uuid4(),
        operation="rewrite_memory",
        affected_memory_ids=(memory_id,),
        verified=True,
        evidence_id="direct-evidence-1",
        payload={"quality_consistency": consistency},
    )


def _sampler(db_manager, memory_id, query: QualityQuery) -> CurationQualitySampler:
    sampler = CurationQualitySampler(
        db_manager=db_manager,
        search=_Search(str(memory_id)),
        repository=cast(Any, SimpleNamespace(put_quality_evidence=lambda evidence: evidence)),
        sample_rate=1.0,
    )
    sampler._find_historical_query = cast(Any, lambda mutation, hypothesis: query)
    return sampler


def test_changed_after_context_is_unverified_with_provenance(db_manager) -> None:
    """Reject replay when captured after-context differs from the query snapshot."""
    memory_id = uuid4()
    query = QualityQuery(
        query_id="query-1",
        query_text="important",
        before_memory_ids=(str(memory_id),),
        replay_complete=True,
        provenance=QueryProvenance.REAL_USER_SEARCH,
        query_context={"workspace": "before"},
        before_search_epochs={"records": 1},
    )

    evidence = _sampler(db_manager, memory_id, query)._evaluate_action(
        _run(),
        _mutation(memory_id, after_context={"workspace": "after"}),
        None,
    )

    assert evidence.status == "unverified"
    assert evidence.neutral_reason == "query_context_changed"
    assert evidence.engagement_evidence["before_query_context"] == {"workspace": "before"}
    assert evidence.engagement_evidence["after_query_context"] == {"workspace": "after"}


def test_missing_after_context_preserves_before_context_behavior(db_manager) -> None:
    """Use the before context when no captured after context is supplied."""
    memory_id = uuid4()
    query = QualityQuery(
        query_id="query-2",
        query_text="important",
        before_memory_ids=(str(memory_id),),
        replay_complete=True,
        provenance=QueryProvenance.REAL_USER_SEARCH,
        query_context={"workspace": "same"},
        before_search_epochs={"records": 1},
    )

    evidence = _sampler(db_manager, memory_id, query)._evaluate_action(
        _run(), _mutation(memory_id), None
    )

    assert evidence.status != "unverified"
    assert evidence.engagement_evidence["after_query_context"] == {"workspace": "same"}


def test_successful_replay_persists_context_and_epoch_snapshots(db_manager) -> None:
    """Persist both query contexts and search epochs with evaluated replay evidence."""
    memory_id = uuid4()
    query = QualityQuery(
        query_id="query-3",
        query_text="important",
        before_memory_ids=(str(memory_id),),
        replay_complete=True,
        provenance=QueryProvenance.REAL_USER_SEARCH,
        query_context={"workspace": "same"},
        before_search_epochs={"records": 1},
    )

    evidence = _sampler(db_manager, memory_id, query)._evaluate_action(
        _run(), _mutation(memory_id, after_context={"workspace": "same"}), None
    )

    assert evidence.engagement_evidence["before_search_epochs"] == {"records": 1}
    assert evidence.engagement_evidence["after_search_epochs"] == {"records": 2}
