"""Pure provider-disclosure decisions for curation context packets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping
from uuid import UUID

from mcp_memory.mutation_history import ProtectionMode


class ProviderTrustClass(StrEnum):
    LOCAL = "local"
    TRUSTED_EXTERNAL = "trusted_external"
    EXTERNAL = "external"


class DisclosureDecision(StrEnum):
    ALLOW = "allow"
    REDACT = "redact"
    DENY = "deny"


@dataclass(frozen=True)
class ProviderTrust:
    """The caller-supplied trust facts for one provider route."""

    trust_class: ProviderTrustClass
    allowlisted: bool = True


@dataclass(frozen=True)
class FieldDisclosure:
    field: str
    value: object
    decision: DisclosureDecision
    reason: str | None = None


@dataclass(frozen=True)
class DisclosureResult:
    memory_id: UUID
    decision: DisclosureDecision
    fields: tuple[FieldDisclosure, ...] = ()
    reason: str = ""
    local_execution_required: bool = False
    review_required: bool = False

    @property
    def disclosed_fields(self) -> dict[str, object]:
        return {
            item.field: item.value
            for item in self.fields
            if item.decision is not DisclosureDecision.DENY
        }


@dataclass(frozen=True)
class ProviderPacket:
    """A provider-ready packet containing only non-denied records."""

    records: tuple[dict[str, object], ...] = ()
    decisions: tuple[DisclosureResult, ...] = ()


def decide_record_disclosure(
    record: Mapping[str, object],
    *,
    provider: ProviderTrust,
    protections: set[ProtectionMode] | frozenset[ProtectionMode] = frozenset(),
    sensitive_fields: set[str] | frozenset[str] = frozenset(),
    fields: tuple[str, ...] | None = None,
    max_characters: int | None = None,
) -> DisclosureResult:
    """Decide disclosure using only explicit caller-supplied sensitivity facts.

    This function deliberately does not inspect content for secrets.  Callers
    that have an independently established sensitive field may request it be
    redacted here.
    """
    memory_id = record.get("memory_id")
    if not isinstance(memory_id, UUID):
        raise ValueError("record must contain a UUID memory_id")

    external = provider.trust_class is not ProviderTrustClass.LOCAL
    if external and ProtectionMode.NO_EXTERNAL_PROVIDER_DISCLOSURE in protections:
        return _denied(memory_id, "external provider disclosure is prohibited")
    if external and ProtectionMode.LOCAL_PROVIDER_ONLY in protections:
        return _denied(memory_id, "local provider execution is required", local=True)
    if not provider.allowlisted:
        return _denied(memory_id, "provider is not operator-allowlisted", review=True)

    selected = tuple(record.items() if fields is None else ((name, record[name]) for name in fields if name in record))
    disclosed: list[FieldDisclosure] = []
    redacted = False
    remaining = max_characters
    for name, value in selected:
        if name in sensitive_fields:
            disclosed.append(FieldDisclosure(name, "[REDACTED]", DisclosureDecision.REDACT, "sensitive field"))
            redacted = True
            continue
        if remaining is not None and isinstance(value, str) and len(value) > remaining:
            disclosed.append(FieldDisclosure(name, value[:remaining], DisclosureDecision.REDACT, "field character bound"))
            redacted = True
            remaining = 0
            continue
        disclosed.append(FieldDisclosure(name, value, DisclosureDecision.ALLOW))
        if remaining is not None and isinstance(value, str):
            remaining = max(0, remaining - len(value))

    return DisclosureResult(
        memory_id=memory_id,
        decision=DisclosureDecision.REDACT if redacted else DisclosureDecision.ALLOW,
        fields=tuple(disclosed),
        reason="bounded or sensitive fields were redacted" if redacted else "provider is eligible",
        local_execution_required=False,
        review_required=ProtectionMode.MANUAL_REVIEW_REQUIRED in protections,
    )


def build_provider_packet(
    records: tuple[Mapping[str, object], ...] | list[Mapping[str, object]],
    *,
    provider: ProviderTrust,
    protections_by_memory: Mapping[UUID, set[ProtectionMode] | frozenset[ProtectionMode]] | None = None,
    sensitive_fields_by_memory: Mapping[UUID, set[str] | frozenset[str]] | None = None,
    fields: tuple[str, ...] | None = None,
    max_characters: int | None = None,
) -> ProviderPacket:
    """Build a packet while guaranteeing denied records are not included."""
    packet_records: list[dict[str, object]] = []
    decisions: list[DisclosureResult] = []
    for record in records:
        memory_id = record.get("memory_id")
        if not isinstance(memory_id, UUID):
            raise ValueError("record must contain a UUID memory_id")
        result = decide_record_disclosure(
            record,
            provider=provider,
            protections=(protections_by_memory or {}).get(memory_id, frozenset()),
            sensitive_fields=(sensitive_fields_by_memory or {}).get(memory_id, frozenset()),
            fields=fields,
            max_characters=max_characters,
        )
        decisions.append(result)
        if result.decision is not DisclosureDecision.DENY:
            packet_records.append({"memory_id": memory_id, **result.disclosed_fields})
    return ProviderPacket(records=tuple(packet_records), decisions=tuple(decisions))


def _denied(memory_id: UUID, reason: str, *, local: bool = False, review: bool = False) -> DisclosureResult:
    return DisclosureResult(
        memory_id=memory_id,
        decision=DisclosureDecision.DENY,
        reason=reason,
        local_execution_required=local,
        review_required=review,
    )
