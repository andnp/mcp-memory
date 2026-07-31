from datetime import datetime, timezone

import pytest

from mcp_memory.config import Config
from mcp_memory.core.ports.memory import MemoryRecord, RankedMemoryCandidate
from mcp_memory.core.search_ranking import RankingEngine, RankingSignals


pytestmark = pytest.mark.small


def _record(memory_id: str, workspace_ids: list[str]) -> MemoryRecord:
    now = datetime.now(timezone.utc).isoformat()
    return MemoryRecord(
        id=memory_id,
        title=memory_id,
        content="search content",
        summary="summary",
        type="fact",
        status="active",
        created_at=now,
        updated_at=now,
        read_count=0,
        access_score=0.0,
        last_accessed_at=None,
        last_surfaced_at=None,
        workspace_ids=workspace_ids,
    )


def test_pure_ranking_uses_explicit_candidate_authority_without_storage() -> None:
    engine = RankingEngine(Config())
    supported = _record("supported", ["workspace-alpha"])
    unsupported = _record("unsupported", ["workspace-alpha"])
    candidates = [
        RankedMemoryCandidate(
            record=supported,
            incoming_links_count=2,
            incoming_link_type_counts={"DEPENDS_ON": 2},
        ),
        RankedMemoryCandidate(record=unsupported, incoming_links_count=0),
    ]

    ranked = engine.rank_records(
        candidates,
        {"supported": 0.04, "unsupported": 0.04},
        workspace_id="workspace-alpha",
        ranking_signals={
            "supported": RankingSignals(matched_by_keyword=True, keyword_token_coverage=1.0),
            "unsupported": RankingSignals(matched_by_keyword=True, keyword_token_coverage=1.0),
        },
        keyword_candidates_present=True,
    )

    assert ranked[0][0].id == "supported"
    assert ranked[0][1] > ranked[1][1]
