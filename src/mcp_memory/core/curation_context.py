"""Provider-free construction of bounded curation context packets.

The context builder accepts only authoritative maintenance-read results.  It
does not perform reads or invoke a provider; callers hand it already accepted
read results and receive an immutable, disclosure-filtered packet.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from time import monotonic
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence
from uuid import UUID

from mcp_memory.core.curation_disclosure import (
    DisclosureDecision,
    ProviderTrust,
    decide_record_disclosure,
)
from mcp_memory.core.curation_identity import (
    context_fingerprint,
    frontier_fingerprint,
    graph_token,
    record_token,
)
from mcp_memory.mutation_history import ProtectionMode


class BudgetDimension(StrEnum):
    """The dimensions enforced while constructing a context packet."""

    SEED_RECORDS = "seed_records"
    SUPPORT_RECORDS = "support_records"
    CONTEXT_CHARACTERS = "context_characters"
    READ_TOOL_CALLS = "read_tool_calls"
    RECORDS_RETURNED = "records_returned"
    WALL_CLOCK_SECONDS = "wall_clock_seconds"


class CurationBudgetExhausted(Exception):
    """A deterministic, typed failure raised before a packet can be returned."""

    reason_code = "budget_exhausted"

    def __init__(
        self,
        dimension: BudgetDimension,
        *,
        used: int | float,
        requested: int | float,
        limit: int | float,
    ) -> None:
        self.dimension = dimension
        self.used = used
        self.requested = requested
        self.limit = limit
        super().__init__(
            f"curation context budget exhausted: {dimension.value} "
            f"used={used!r} requested={requested!r} limit={limit!r}"
        )


# These aliases keep the error discoverable without creating multiple error
# types for callers that use "exceeded" terminology.
CurationBudgetExceeded = CurationBudgetExhausted
ContextBudgetExceeded = CurationBudgetExhausted


@dataclass(frozen=True, slots=True)
class CurationReadBudget:
    """Typed limits for one in-memory context construction."""

    max_seed_records: int = 16
    max_support_records: int = 8
    max_context_characters: int = 12_000
    max_read_tool_calls: int = 24
    max_records_returned: int = 32
    max_wall_clock_seconds: float = 30.0

    def __post_init__(self) -> None:
        for name in (
            "max_seed_records",
            "max_support_records",
            "max_context_characters",
            "max_read_tool_calls",
            "max_records_returned",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.max_wall_clock_seconds < 0:
            raise ValueError("max_wall_clock_seconds must be non-negative")


CurationContextBudget = CurationReadBudget


@dataclass(frozen=True, slots=True)
class CurationReadCounters:
    """Deterministic usage accumulated by accepted maintenance reads."""

    seed_records: int = 0
    support_records: int = 0
    context_characters: int = 0
    read_tool_calls: int = 0
    records_returned: int = 0
    wall_clock_seconds: float = 0.0

    def as_dict(self) -> dict[str, int | float]:
        return {
            "seed_records": self.seed_records,
            "support_records": self.support_records,
            "context_characters": self.context_characters,
            "read_tool_calls": self.read_tool_calls,
            "records_returned": self.records_returned,
            "wall_clock_seconds": self.wall_clock_seconds,
        }


@dataclass(frozen=True, slots=True)
class AcceptedMaintenanceRead:
    """A read result admitted from an authoritative maintenance path.

    ``record`` is the authoritative record used for identity tokens.  The
    optional edges are the relevant incoming/outgoing graph slice returned by
    the same maintenance read.  Counts describe the tool call that produced
    the result, rather than provider activity.
    """

    record: Mapping[str, Any]
    edges: tuple[Mapping[str, Any], ...] = ()
    read_tool_calls: int = 1
    records_returned: int = 1

    def __post_init__(self) -> None:
        if self.read_tool_calls < 0 or self.records_returned < 0:
            raise ValueError("maintenance read counters must be non-negative")


def accept_maintenance_read(
    response: Mapping[str, Any] | AcceptedMaintenanceRead,
    *,
    edges: Sequence[Mapping[str, Any]] | None = None,
    read_tool_calls: int = 1,
    records_returned: int | None = None,
) -> AcceptedMaintenanceRead:
    """Validate and normalize one internal maintenance-read response.

    Error responses are rejected before context construction.  A raw record
    mapping is also accepted for pure callers that already performed the
    authoritative read and stripped the service envelope.
    """

    if isinstance(response, AcceptedMaintenanceRead):
        return response
    if not isinstance(response, Mapping):
        raise TypeError("maintenance read must be a mapping")
    if "status" in response and response["status"] != "ok":
        raise ValueError("maintenance read was not accepted")
    raw_record = response.get("record", response)
    if not isinstance(raw_record, Mapping):
        raise ValueError("accepted maintenance read must contain a record mapping")
    response_edges = edges if edges is not None else _edges_from_mapping(response)
    return AcceptedMaintenanceRead(
        record=raw_record,
        edges=tuple(response_edges),
        read_tool_calls=read_tool_calls,
        records_returned=(1 if records_returned is None else records_returned),
    )


@dataclass(frozen=True, slots=True)
class CurationContextPacket:
    """Immutable provider-ready context and its canonical identity."""

    frontier_fingerprint: str
    seeds: tuple[Mapping[str, Any], ...]
    support: tuple[Mapping[str, Any], ...]
    record_tokens: Mapping[str, str]
    graph_tokens: Mapping[str, str]
    disclosure: tuple[Mapping[str, Any], ...]
    omissions: tuple[Mapping[str, Any], ...]
    limits: Mapping[str, int | float]
    usage: CurationReadCounters
    context_fingerprint: str

    @property
    def seed_memory_ids(self) -> tuple[str, ...]:
        return tuple(str(record["memory_id"]) for record in self.seeds)

    @property
    def support_memory_ids(self) -> tuple[str, ...]:
        return tuple(str(record["memory_id"]) for record in self.support)

    @property
    def provider_records(self) -> tuple[Mapping[str, Any], ...]:
        return self.seeds + self.support

    def as_dict(self) -> dict[str, Any]:
        """Return a mutable serialization without changing this packet."""

        return _thaw(
            {
                "frontier_fingerprint": self.frontier_fingerprint,
                "seeds": self.seeds,
                "support": self.support,
                "record_tokens": self.record_tokens,
                "graph_tokens": self.graph_tokens,
                "disclosure": self.disclosure,
                "omissions": self.omissions,
                "limits": self.limits,
                "usage": self.usage.as_dict(),
                "context_fingerprint": self.context_fingerprint,
            }
        )


def build_context_packet(
    *,
    family: str,
    strategy: str,
    seed_reads: Iterable[Mapping[str, Any] | AcceptedMaintenanceRead],
    support_reads: Iterable[Mapping[str, Any] | AcceptedMaintenanceRead] = (),
    provider: ProviderTrust,
    budget: CurationReadBudget | None = None,
    protections_by_memory: Mapping[UUID | str, set[ProtectionMode] | frozenset[ProtectionMode]] | None = None,
    sensitive_fields_by_memory: Mapping[UUID | str, set[str] | frozenset[str]] | None = None,
    max_record_characters: int | None = None,
    require_authoritative_disclosure_context: bool = False,
    clock: Callable[[], float] = monotonic,
) -> CurationContextPacket:
    """Build one bounded packet from accepted maintenance reads.

    Disclosure is decided before a provider-facing record is assembled.  A
    denied read contributes audit omission metadata and read usage, but no
    record, content, revision token, or graph token enters the packet.
    """

    limits = budget or CurationReadBudget()
    if max_record_characters is not None and max_record_characters < 0:
        raise ValueError("max_record_characters must be non-negative")
    seed_values = tuple(seed_reads)
    support_values = tuple(support_reads)
    seed_ids = [_memory_id(_coerce_read(value).record) for value in seed_values]
    frontier = frontier_fingerprint(family, strategy, seed_ids)
    started = clock()
    usage = CurationReadCounters()
    seeds: list[Mapping[str, Any]] = []
    support: list[Mapping[str, Any]] = []
    record_tokens: dict[str, str] = {}
    graph_tokens: dict[str, str] = {}
    disclosures: list[Mapping[str, Any]] = []
    omissions: list[Mapping[str, Any]] = []
    seen: set[str] = set()

    for role, values in (("seed", seed_values), ("support", support_values)):
        for value in values:
            read = _coerce_read(value)
            usage = _advance_read_usage(usage, read, limits, clock() - started)
            raw_record = _identity_record(read.record)
            memory_id = _memory_id(raw_record)
            protections = _lookup(protections_by_memory, memory_id)
            sensitive_fields = _lookup(sensitive_fields_by_memory, memory_id)
            selected_fields = _seed_fields(raw_record) if role == "seed" else _support_fields(raw_record)
            result = decide_record_disclosure(
                {"memory_id": _as_uuid(memory_id), **raw_record},
                provider=provider,
                protections=protections,
                sensitive_fields=sensitive_fields,
                fields=selected_fields,
                max_characters=max_record_characters,
                authoritative_context_available=(
                    not require_authoritative_disclosure_context
                    or (
                        _has_mapping_entry(protections_by_memory, memory_id)
                        and _has_mapping_entry(sensitive_fields_by_memory, memory_id)
                    )
                ),
            )
            disclosures.append(_disclosure_payload(role, result))
            if result.decision is DisclosureDecision.DENY:
                omissions.append(
                    _freeze(
                        {
                            "memory_id": memory_id,
                            "role": role,
                            "reason": result.reason,
                        }
                    )
                )
                continue
            if memory_id in seen:
                omissions.append(
                    _freeze(
                        {
                            "memory_id": memory_id,
                            "role": role,
                            "reason": "duplicate accepted maintenance read",
                        }
                    )
                )
                continue
            next_count = usage.seed_records + (1 if role == "seed" else 0)
            next_support = usage.support_records + (1 if role == "support" else 0)
            _check_limit(
                BudgetDimension.SEED_RECORDS if role == "seed" else BudgetDimension.SUPPORT_RECORDS,
                next_count if role == "seed" else next_support,
                0,
                limits.max_seed_records if role == "seed" else limits.max_support_records,
            )
            provider_record = _provider_record(raw_record, result, role, read.edges)
            chars = len(_json_text(provider_record))
            _check_limit(
                BudgetDimension.CONTEXT_CHARACTERS,
                usage.context_characters,
                chars,
                limits.max_context_characters,
            )
            seen.add(memory_id)
            usage = CurationReadCounters(
                seed_records=next_count,
                support_records=next_support,
                context_characters=usage.context_characters + chars,
                read_tool_calls=usage.read_tool_calls,
                records_returned=usage.records_returned,
                wall_clock_seconds=usage.wall_clock_seconds,
            )
            frozen_record = _freeze(provider_record)
            (seeds if role == "seed" else support).append(frozen_record)
            record_tokens[memory_id] = _record_revision_token(raw_record)
            graph_tokens[memory_id] = _graph_revision_token(memory_id, read)

    packet_fields = {
        "frontier_fingerprint": frontier,
        "seeds": seeds,
        "support": support,
        "record_tokens": record_tokens,
        "graph_tokens": graph_tokens,
        "disclosure": disclosures,
        "omissions": omissions,
        "limits": _limits_dict(limits),
    }
    fingerprint = context_fingerprint(packet_fields)
    return CurationContextPacket(
        frontier_fingerprint=frontier,
        seeds=tuple(seeds),
        support=tuple(support),
        record_tokens=_freeze(record_tokens),
        graph_tokens=_freeze(graph_tokens),
        disclosure=tuple(disclosures),
        omissions=tuple(omissions),
        limits=_freeze(_limits_dict(limits)),
        usage=usage,
        context_fingerprint=fingerprint,
    )


def disclosure_audit_manifest(
    context: CurationContextPacket,
    provider: ProviderTrust,
    *,
    max_records: int = 256,
    max_fields: int = 32,
) -> dict[str, Any]:
    """Return bounded disclosure metadata without copying provider content."""

    if max_records < 1 or max_fields < 1:
        raise ValueError("disclosure audit bounds must be positive")
    records: list[dict[str, Any]] = []
    truncated = False
    for disclosure in context.disclosure:
        if len(records) >= max_records:
            truncated = True
            break
        fields = list(disclosure.get("fields", ()))
        if len(fields) > max_fields:
            truncated = True
            fields = fields[:max_fields]
        records.append(
            {
                "memory_id": str(disclosure.get("memory_id", "")),
                "role": str(disclosure.get("role", "")),
                "decision": str(disclosure.get("decision", "")),
                "reason": str(disclosure.get("reason", "")),
                "fields": [
                    {
                        "field": str(field.get("field", "")),
                        "decision": str(field.get("decision", "")),
                        "reason": None if field.get("reason") is None else str(field["reason"]),
                    }
                    for field in fields
                    if isinstance(field, Mapping)
                ],
            }
        )
    return {
        "version": 1,
        "provider_trust_class": str(provider.trust_class),
        "provider_allowlisted": provider.allowlisted,
        "records": records,
        "truncated": truncated,
    }


# Descriptive aliases for callers that use "construct" or "build" language.
construct_context_packet = build_context_packet
build_curation_context = build_context_packet


def _coerce_read(value: Mapping[str, Any] | AcceptedMaintenanceRead) -> AcceptedMaintenanceRead:
    return value if isinstance(value, AcceptedMaintenanceRead) else accept_maintenance_read(value)


def _memory_id(record: Mapping[str, Any]) -> str:
    value = record.get("id", record.get("memory_id"))
    if not isinstance(value, (str, UUID)) or not str(value).strip():
        raise ValueError("accepted maintenance record must contain an id")
    return str(value).strip()


def _as_uuid(value: str) -> UUID:
    try:
        return UUID(value)
    except ValueError as exc:
        raise ValueError("context records must use UUID memory IDs") from exc


def _identity_record(record: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(record)
    value["id"] = _memory_id(value)
    value.setdefault("title", "")
    value.setdefault("content", "")
    value.setdefault("summary", None)
    value.setdefault("type", value.pop("memory_type", "unknown"))
    value.setdefault("status", "active")
    value.setdefault("tags", [])
    value.setdefault("workspace_ids", [])
    value.setdefault("metadata", {})
    return value


def _seed_fields(record: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        field
        for field in ("title", "content", "summary", "type", "status", "tags", "workspace_ids", "metadata")
        if field in record
    )


def _support_fields(record: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(field for field in ("title", "summary") if field in record)


def _provider_record(
    record: Mapping[str, Any], result: Any, role: str, edges: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    payload: dict[str, Any] = {"memory_id": _memory_id(record)}
    payload.update(result.disclosed_fields)
    if role == "support":
        payload = {
            "memory_id": payload["memory_id"],
            "title": payload.get("title"),
            "summary": payload.get("summary"),
        }
    normalized_edges = _provider_edges(edges)
    if normalized_edges:
        payload["relationships"] = normalized_edges
    return payload


def _provider_edges(edges: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for edge in edges:
        nested_link = edge.get("link")
        link: Mapping[str, Any] = nested_link if isinstance(nested_link, Mapping) else edge
        source = link.get("source_id")
        target = link.get("target_id")
        link_type = link.get("type", link.get("link_type"))
        if source is None or target is None or link_type is None:
            continue
        result.append(
            {
                "source_id": str(source),
                "target_id": str(target),
                "type": str(link_type),
                "context": link.get("context"),
            }
        )
    return result


def _edges_from_mapping(value: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    direct = value.get("edges")
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes, bytearray)):
        return tuple(item for item in direct if isinstance(item, Mapping))
    relationships = value.get("relationships")
    edges: list[Mapping[str, Any]] = []
    if isinstance(relationships, Mapping):
        for items in relationships.values():
            if isinstance(items, Sequence) and not isinstance(items, (str, bytes, bytearray)):
                edges.extend(item for item in items if isinstance(item, Mapping))
    elif isinstance(relationships, Sequence) and not isinstance(relationships, (str, bytes, bytearray)):
        edges.extend(item for item in relationships if isinstance(item, Mapping))
    return tuple(edges)


def _record_revision_token(record: Mapping[str, Any]) -> str:
    supplied = record.get("revision_token", record.get("record_token"))
    return str(supplied) if isinstance(supplied, str) and supplied else record_token(record)


def _graph_revision_token(memory_id: str, read: AcceptedMaintenanceRead) -> str:
    supplied = read.record.get("graph_token")
    return str(supplied) if isinstance(supplied, str) and supplied else graph_token(memory_id, _provider_edges(read.edges))


def _lookup(mapping: Mapping[UUID | str, Any] | None, memory_id: str) -> Any:
    if mapping is None:
        return frozenset()
    return mapping.get(memory_id, mapping.get(_as_uuid(memory_id), frozenset()))


def _has_mapping_entry(mapping: Mapping[UUID | str, Any] | None, memory_id: str) -> bool:
    if mapping is None:
        return False
    return memory_id in mapping or _as_uuid(memory_id) in mapping


def _disclosure_payload(role: str, result: Any) -> Mapping[str, Any]:
    return _freeze(
        {
            "memory_id": str(result.memory_id),
            "role": role,
            "decision": result.decision.value,
            "reason": result.reason,
            "fields": [
                {
                    "field": item.field,
                    "decision": item.decision.value,
                    "reason": item.reason,
                }
                for item in result.fields
            ],
        }
    )


def _advance_read_usage(
    usage: CurationReadCounters,
    read: AcceptedMaintenanceRead,
    budget: CurationReadBudget,
    elapsed: float,
) -> CurationReadCounters:
    _check_limit(BudgetDimension.READ_TOOL_CALLS, usage.read_tool_calls, read.read_tool_calls, budget.max_read_tool_calls)
    _check_limit(BudgetDimension.RECORDS_RETURNED, usage.records_returned, read.records_returned, budget.max_records_returned)
    if elapsed > budget.max_wall_clock_seconds:
        raise CurationBudgetExhausted(
            BudgetDimension.WALL_CLOCK_SECONDS,
            used=elapsed,
            requested=0,
            limit=budget.max_wall_clock_seconds,
        )
    return CurationReadCounters(
        seed_records=usage.seed_records,
        support_records=usage.support_records,
        context_characters=usage.context_characters,
        read_tool_calls=usage.read_tool_calls + read.read_tool_calls,
        records_returned=usage.records_returned + read.records_returned,
        wall_clock_seconds=elapsed,
    )


def _check_limit(dimension: BudgetDimension, used: int, requested: int, limit: int) -> None:
    if used + requested > limit:
        raise CurationBudgetExhausted(dimension, used=used, requested=requested, limit=limit)


def _limits_dict(budget: CurationReadBudget) -> dict[str, int | float]:
    return {
        "max_seed_records": budget.max_seed_records,
        "max_support_records": budget.max_support_records,
        "max_context_characters": budget.max_context_characters,
        "max_read_tool_calls": budget.max_read_tool_calls,
        "max_records_returned": budget.max_records_returned,
        "max_wall_clock_seconds": budget.max_wall_clock_seconds,
    }


def _json_text(value: Any) -> str:
    from mcp_memory.core.curation_identity import canonical_json

    return canonical_json(value).decode("utf-8")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value
