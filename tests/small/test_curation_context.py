from __future__ import annotations

from uuid import UUID

import pytest

from mcp_memory.core.curation_context import (
    AcceptedMaintenanceRead,
    CurationBudgetExhausted,
    CurationReadBudget,
    build_context_packet,
)
from mcp_memory.core.curation_disclosure import ProviderTrust, ProviderTrustClass
from mcp_memory.mutation_history import ProtectionMode


SEED_ID = UUID("00000000-0000-0000-0000-000000000001")
SUPPORT_ID = UUID("00000000-0000-0000-0000-000000000002")


def _record(memory_id: UUID, content: str) -> dict[str, object]:
    return {
        "id": memory_id,
        "title": "Title",
        "content": content,
        "summary": "Summary",
        "type": "observation",
        "status": "active",
        "tags": ["tag"],
        "workspace_ids": ["workspace"],
    }


def test_context_is_immutable_bounded_and_attaches_record_and_graph_tokens() -> None:
    packet = build_context_packet(
        family="curator",
        strategy="recent",
        seed_reads=[
            AcceptedMaintenanceRead(
                _record(SEED_ID, "seed content"),
                edges=({"source_id": SEED_ID, "target_id": SUPPORT_ID, "type": "RELATES", "context": "near"},),
            )
        ],
        support_reads=[AcceptedMaintenanceRead(_record(SUPPORT_ID, "support content"))],
        provider=ProviderTrust(ProviderTrustClass.LOCAL),
    )

    assert packet.seeds[0]["content"] == "seed content"
    assert "content" not in packet.support[0]
    assert str(SEED_ID) in packet.record_tokens
    assert str(SEED_ID) in packet.graph_tokens
    assert packet.context_fingerprint.startswith("v1:")
    with pytest.raises(TypeError):
        packet.seeds[0]["content"] = "changed"  # type: ignore[index]


def test_denied_content_is_omitted_before_provider_packet_construction() -> None:
    packet = build_context_packet(
        family="curator",
        strategy="recent",
        seed_reads=[AcceptedMaintenanceRead(_record(SEED_ID, "secret denied content"))],
        provider=ProviderTrust(ProviderTrustClass.EXTERNAL),
        protections_by_memory={SEED_ID: {ProtectionMode.NO_EXTERNAL_PROVIDER_DISCLOSURE}},
    )

    assert packet.seeds == ()
    assert packet.record_tokens == {}
    assert all("secret denied content" not in str(item) for item in packet.as_dict()["omissions"])
    assert packet.disclosure[0]["decision"] == "deny"


def test_equivalent_packet_maps_have_same_fingerprint() -> None:
    first = build_context_packet(
        family="curator",
        strategy="recent",
        seed_reads=[AcceptedMaintenanceRead(_record(SEED_ID, "same"))],
        provider=ProviderTrust(ProviderTrustClass.LOCAL),
    )
    second = build_context_packet(
        family="curator",
        strategy="recent",
        seed_reads=[AcceptedMaintenanceRead(dict(reversed(list(_record(SEED_ID, "same").items()))))],
        provider=ProviderTrust(ProviderTrustClass.LOCAL),
    )
    assert first.context_fingerprint == second.context_fingerprint
    assert first.frontier_fingerprint == second.frontier_fingerprint


def test_budget_exhaustion_is_typed_and_deterministic() -> None:
    reads = [
        AcceptedMaintenanceRead(_record(SEED_ID, "one")),
        AcceptedMaintenanceRead(_record(SUPPORT_ID, "two")),
    ]
    budget = CurationReadBudget(max_seed_records=1, max_support_records=1, max_read_tool_calls=1)
    with pytest.raises(CurationBudgetExhausted) as first:
        build_context_packet(
            family="curator",
            strategy="recent",
            seed_reads=reads,
            provider=ProviderTrust(ProviderTrustClass.LOCAL),
            budget=budget,
        )
    with pytest.raises(CurationBudgetExhausted) as second:
        build_context_packet(
            family="curator",
            strategy="recent",
            seed_reads=reads,
            provider=ProviderTrust(ProviderTrustClass.LOCAL),
            budget=budget,
        )
    assert first.value.dimension.value == "read_tool_calls"
    assert (first.value.used, first.value.requested, first.value.limit) == (
        1,
        1,
        1,
    )
    assert str(first.value) == str(second.value)


def test_time_budget_uses_injected_clock() -> None:
    ticks = iter((0.0, 2.0))
    with pytest.raises(CurationBudgetExhausted) as error:
        build_context_packet(
            family="curator",
            strategy="recent",
            seed_reads=[AcceptedMaintenanceRead(_record(SEED_ID, "content"))],
            provider=ProviderTrust(ProviderTrustClass.LOCAL),
            budget=CurationReadBudget(max_wall_clock_seconds=1.0),
            clock=lambda: next(ticks),
        )
    assert error.value.dimension.value == "wall_clock_seconds"
