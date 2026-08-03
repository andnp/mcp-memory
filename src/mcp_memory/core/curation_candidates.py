"""Application boundary for acquiring curator candidates."""

from __future__ import annotations

from dataclasses import dataclass

from mcp_memory.context import ApplicationContext
from mcp_memory.core.curation_models import CampaignHypothesis
from mcp_memory.core.sampling import SamplingBatch


@dataclass(frozen=True, slots=True)
class CuratorCandidateRequest:
    task_id: str
    workspace_id: str | None = None
    requested_strategy: str | None = None
    limit: int | None = None
    exclude_memory_ids: frozenset[str] = frozenset()
    campaign_hypothesis: CampaignHypothesis | None = None


@dataclass(frozen=True, slots=True)
class CuratorSamplingContext:
    task_id: str
    requested_strategy: str | None
    workspace_id: str | None = None
    task_name: str = "memory-curator"


def acquire_curator_candidates(
    ctx: ApplicationContext,
    request: CuratorCandidateRequest,
) -> SamplingBatch:
    from mcp_memory.core.task_handlers.curator_support import _acquire_curator_candidates

    return _acquire_curator_candidates(ctx, request)
