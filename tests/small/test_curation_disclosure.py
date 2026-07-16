from uuid import uuid4

from mcp_memory.core.curation_disclosure import (
    DisclosureDecision,
    ProtectionMode,
    ProviderTrust,
    ProviderTrustClass,
    build_provider_packet,
    decide_record_disclosure,
)
from mcp_memory.core.curation_context import build_context_packet, disclosure_audit_manifest


def test_strict_protection_denies_external_disclosure_and_explains_local_execution() -> None:
    memory_id = uuid4()
    result = decide_record_disclosure(
        {"memory_id": memory_id, "content": "private"},
        provider=ProviderTrust(ProviderTrustClass.EXTERNAL, allowlisted=True),
        protections={ProtectionMode.LOCAL_PROVIDER_ONLY},
    )
    assert result.decision is DisclosureDecision.DENY
    assert result.local_execution_required
    assert "local" in result.reason


def test_local_provider_is_allowed_by_local_only_protection() -> None:
    result = decide_record_disclosure(
        {"memory_id": uuid4(), "content": "private"},
        provider=ProviderTrust(ProviderTrustClass.LOCAL),
        protections={ProtectionMode.LOCAL_PROVIDER_ONLY},
    )
    assert result.decision is DisclosureDecision.ALLOW


def test_no_external_disclosure_wins_over_other_protections() -> None:
    memory_id = uuid4()
    result = decide_record_disclosure(
        {"memory_id": memory_id, "content": "private"},
        provider=ProviderTrust(ProviderTrustClass.EXTERNAL, allowlisted=True),
        protections={ProtectionMode.MANUAL_REVIEW_REQUIRED, ProtectionMode.NO_EXTERNAL_PROVIDER_DISCLOSURE},
    )
    assert result.decision is DisclosureDecision.DENY
    assert "prohibited" in result.reason


def test_bounded_and_explicitly_sensitive_fields_are_redacted_without_scanning() -> None:
    memory_id = uuid4()
    result = decide_record_disclosure(
        {"memory_id": memory_id, "content": "abcdef", "token": "do-not-scan"},
        provider=ProviderTrust(ProviderTrustClass.EXTERNAL, allowlisted=True),
        sensitive_fields={"token"},
        max_characters=3,
    )
    assert result.decision is DisclosureDecision.REDACT
    assert result.disclosed_fields == {"memory_id": memory_id, "content": "abc", "token": "[REDACTED]"}


def test_denied_records_never_enter_provider_packet() -> None:
    denied, allowed = uuid4(), uuid4()
    packet = build_provider_packet(
        [{"memory_id": denied, "content": "no"}, {"memory_id": allowed, "content": "yes"}],
        provider=ProviderTrust(ProviderTrustClass.EXTERNAL, allowlisted=True),
        protections_by_memory={denied: {ProtectionMode.NO_EXTERNAL_PROVIDER_DISCLOSURE}},
    )
    assert [record["memory_id"] for record in packet.records] == [allowed]
    assert packet.decisions[0].decision is DisclosureDecision.DENY


def test_external_context_denies_when_authoritative_policy_is_unavailable() -> None:
    memory_id = uuid4()
    context = build_context_packet(
        family="curator",
        strategy="focused",
        seed_reads=[{"id": memory_id, "content": "private"}],
        provider=ProviderTrust(ProviderTrustClass.EXTERNAL, allowlisted=True),
        require_authoritative_disclosure_context=True,
    )
    assert context.seeds == ()
    assert context.disclosure[0]["decision"] == DisclosureDecision.DENY
    assert context.disclosure[0]["memory_id"] == str(memory_id)


def test_external_context_persists_only_redaction_metadata_for_authorized_records() -> None:
    denied, redacted = uuid4(), uuid4()
    context = build_context_packet(
        family="curator",
        strategy="focused",
        seed_reads=[
            {"id": denied, "content": "must not leave the store"},
            {"id": redacted, "content": "sensitive content"},
        ],
        provider=ProviderTrust(ProviderTrustClass.EXTERNAL, allowlisted=True),
        protections_by_memory={denied: {ProtectionMode.NO_EXTERNAL_PROVIDER_DISCLOSURE}, redacted: set()},
        sensitive_fields_by_memory={denied: set(), redacted: {"content"}},
        require_authoritative_disclosure_context=True,
    )
    assert [record["memory_id"] for record in context.seeds] == [str(redacted)]
    assert context.seeds[0]["content"] == "[REDACTED]"
    assert "must not leave the store" not in str(context.as_dict())
    assert context.disclosure[0]["decision"] == DisclosureDecision.DENY
    assert next(field for field in context.disclosure[1]["fields"] if field["field"] == "content")["decision"] == DisclosureDecision.REDACT


def test_relationships_filter_endpoints_and_persist_content_free_audit_metadata() -> None:
    source, target = uuid4(), uuid4()
    context = build_context_packet(
        family="curator",
        strategy="focused",
        seed_reads=[
            {
                "id": source,
                "content": "source",
                "edges": [
                    {
                        "source_id": source,
                        "target_id": target,
                        "type": "RELATES",
                        "context": "private relationship context",
                    }
                ],
            }
        ],
        provider=ProviderTrust(ProviderTrustClass.TRUSTED_EXTERNAL, allowlisted=True),
        protections_by_memory={source: set(), target: set()},
        sensitive_fields_by_memory={source: {"context"}, target: set()},
        require_authoritative_disclosure_context=True,
    )

    relationship = context.seeds[0]["relationships"][0]
    assert relationship["source_id"] == str(source)
    assert relationship["target_id"] == str(target)
    assert relationship["context"] == "[REDACTED]"

    manifest = disclosure_audit_manifest(context, ProviderTrust(ProviderTrustClass.TRUSTED_EXTERNAL, allowlisted=True))
    relationship_audit = next(item for item in manifest["records"] if item["kind"] == "relationship")
    assert relationship_audit["source_id"] == str(source)
    assert relationship_audit["target_id"] == str(target)
    assert relationship_audit["type"] == "RELATES"
    assert any(field["field"] == "context" for field in relationship_audit["fields"])
    assert "private relationship context" not in str(manifest)


def test_relationships_fail_closed_without_endpoint_policy() -> None:
    source, target = uuid4(), uuid4()
    context = build_context_packet(
        family="curator",
        strategy="focused",
        seed_reads=[
            {
                "id": source,
                "content": "source",
                "edges": [{"source_id": source, "target_id": target, "type": "RELATES"}],
            }
        ],
        provider=ProviderTrust(ProviderTrustClass.TRUSTED_EXTERNAL, allowlisted=True),
        protections_by_memory={source: set()},
        sensitive_fields_by_memory={source: set()},
        require_authoritative_disclosure_context=True,
    )

    assert "relationships" not in context.seeds[0]
    assert any(
        item["kind"] == "relationship" and "policy is unavailable" in item["reason"]
        for item in context.disclosure
    )
