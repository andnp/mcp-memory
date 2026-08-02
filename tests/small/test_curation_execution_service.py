from __future__ import annotations

from uuid import UUID

import pytest

from mcp_memory.core.curation_context import AcceptedMaintenanceRead, build_context_packet
from mcp_memory.core.curation_disclosure import ProviderTrust, ProviderTrustClass
from mcp_memory.core.curation_execution_service import _hydrate_record_tokens
from mcp_memory.core.curation_models import (
    ActionPreconditions,
    ClaimManifest,
    MergeMemoriesAction,
    NormalizeMemoryAction,
)


pytestmark = pytest.mark.small

CANONICAL_ID = UUID("00000000-0000-0000-0000-000000000001")
SOURCE_ID = UUID("00000000-0000-0000-0000-000000000002")


def _record(memory_id: UUID) -> dict[str, object]:
    return {
        "id": memory_id,
        "title": "Title",
        "content": "Content",
        "summary": "Summary",
        "type": "observation",
        "status": "active",
        "tags": [],
        "workspace_ids": [],
    }


def _context():
    return build_context_packet(
        family="curator",
        strategy="recent",
        seed_reads=[
            AcceptedMaintenanceRead(_record(CANONICAL_ID)),
            AcceptedMaintenanceRead(_record(SOURCE_ID)),
        ],
        provider=ProviderTrust(ProviderTrustClass.LOCAL),
    )


def test_hydrates_missing_tokens_for_every_affected_memory() -> None:
    context = _context()
    action = MergeMemoriesAction(
        action_id=UUID("00000000-0000-0000-0000-000000000003"),
        canonical_id=CANONICAL_ID,
        source_ids=[SOURCE_ID],
        confidence=1,
        rationale="merge duplicate records",
        content="Merged content",
        claim_manifest=ClaimManifest(preserved_claims=["claim"]),
    )

    hydrated = _hydrate_record_tokens(action, context)

    assert hydrated.preconditions.record_tokens == {
        CANONICAL_ID: context.record_tokens[str(CANONICAL_ID)],
        SOURCE_ID: context.record_tokens[str(SOURCE_ID)],
    }


def test_preserves_provider_supplied_token_mismatch() -> None:
    context = _context()
    action = NormalizeMemoryAction(
        action_id=UUID("00000000-0000-0000-0000-000000000004"),
        target_id=CANONICAL_ID,
        confidence=1,
        rationale="normalize metadata",
        title="Specific title",
        preconditions=ActionPreconditions(record_tokens={CANONICAL_ID: "stale-provider-token"}),
    )

    hydrated = _hydrate_record_tokens(action, context)

    assert hydrated.preconditions.record_tokens[CANONICAL_ID] == "stale-provider-token"
