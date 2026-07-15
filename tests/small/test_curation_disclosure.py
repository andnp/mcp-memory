from uuid import uuid4

from mcp_memory.core.curation_disclosure import (
    DisclosureDecision,
    ProtectionMode,
    ProviderTrust,
    ProviderTrustClass,
    build_provider_packet,
    decide_record_disclosure,
)


def test_strict_protection_denies_external_disclosure_and_explains_local_execution() -> None:
    memory_id = uuid4()
    result = decide_record_disclosure(
        {"memory_id": memory_id, "content": "private"},
        provider=ProviderTrust(ProviderTrustClass.EXTERNAL),
        protections={ProtectionMode.LOCAL_PROVIDER_ONLY},
    )
    assert result.decision is DisclosureDecision.DENY
    assert result.local_execution_required
    assert "local" in result.reason


def test_no_external_disclosure_wins_over_other_protections() -> None:
    memory_id = uuid4()
    result = decide_record_disclosure(
        {"memory_id": memory_id, "content": "private"},
        provider=ProviderTrust(ProviderTrustClass.EXTERNAL),
        protections={ProtectionMode.MANUAL_REVIEW_REQUIRED, ProtectionMode.NO_EXTERNAL_PROVIDER_DISCLOSURE},
    )
    assert result.decision is DisclosureDecision.DENY
    assert "prohibited" in result.reason


def test_bounded_and_explicitly_sensitive_fields_are_redacted_without_scanning() -> None:
    memory_id = uuid4()
    result = decide_record_disclosure(
        {"memory_id": memory_id, "content": "abcdef", "token": "do-not-scan"},
        provider=ProviderTrust(ProviderTrustClass.EXTERNAL),
        sensitive_fields={"token"},
        max_characters=3,
    )
    assert result.decision is DisclosureDecision.REDACT
    assert result.disclosed_fields == {"memory_id": memory_id, "content": "abc", "token": "[REDACTED]"}


def test_denied_records_never_enter_provider_packet() -> None:
    denied, allowed = uuid4(), uuid4()
    packet = build_provider_packet(
        [{"memory_id": denied, "content": "no"}, {"memory_id": allowed, "content": "yes"}],
        provider=ProviderTrust(ProviderTrustClass.EXTERNAL),
        protections_by_memory={denied: {ProtectionMode.NO_EXTERNAL_PROVIDER_DISCLOSURE}},
    )
    assert [record["memory_id"] for record in packet.records] == [allowed]
    assert packet.decisions[0].decision is DisclosureDecision.DENY
