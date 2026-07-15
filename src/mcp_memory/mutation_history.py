"""Pure domain contracts for mutation history, protection, and restore."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from mcp_memory.core.curation_identity import record_snapshot, record_token


class MutationHistoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MutationActorKind(StrEnum):
    USER = "user"
    MAINTENANCE = "maintenance"
    SYSTEM = "system"
    RESTORE = "restore"


class MutationEventStatus(StrEnum):
    APPLIED = "applied"
    REJECTED = "rejected"
    STALE = "stale"
    FAILED = "failed"


class RevisionRole(StrEnum):
    TARGET = "target"
    CANONICAL = "canonical"
    SOURCE = "source"
    ORIGINAL = "original"
    SPLIT_CHILD = "split_child"


class ProtectionMode(StrEnum):
    NO_AUTONOMOUS_MUTATION = "no_autonomous_mutation"
    NO_AUTONOMOUS_DESTRUCTIVE_CHANGE = "no_autonomous_destructive_change"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    LOCAL_PROVIDER_ONLY = "local_provider_only"
    NO_EXTERNAL_PROVIDER_DISCLOSURE = "no_external_provider_disclosure"
    PINNED_ACTIVE = "pinned_active"


class RestoreScope(StrEnum):
    ALL = "all"
    RECORDS = "records"
    LINKS = "links"


class RestoreResultStatus(StrEnum):
    APPLIED = "applied"
    CONFLICT = "conflict"
    REJECTED = "rejected"
    ALREADY_APPLIED = "already_applied"
    FAILED = "failed"


class MutationEvent(MutationHistoryModel):
    id: UUID
    operation: str
    actor_kind: MutationActorKind
    actor_id: str | None = None
    family: str | None = None
    task_id: UUID | None = None
    curation_run_id: UUID | None = None
    plan_id: UUID | None = None
    action_id: UUID | None = None
    provider_id: str | None = None
    reason_code: str | None = None
    rationale: str | None = None
    policy_version: str | None = None
    schema_version: int = 1
    status: MutationEventStatus
    restores_event_id: UUID | None = None
    idempotency_key: str | None = None
    created_at: datetime | None = None
    terminalized_at: datetime | None = None


class RecordRevision(MutationHistoryModel):
    event_id: UUID
    memory_id: UUID
    role: RevisionRole
    before_exists: bool
    before_snapshot: dict[str, Any] | None = None
    after_exists: bool
    after_snapshot: dict[str, Any] | None = None
    before_token: str | None = None
    after_token: str | None = None


class LinkRevision(MutationHistoryModel):
    event_id: UUID
    source_id: UUID
    target_id: UUID
    link_type: str
    context: str | None = None
    before_exists: bool
    after_exists: bool


class Protection(MutationHistoryModel):
    memory_id: UUID
    mode: ProtectionMode
    reason: str
    actor_id: str | None = None
    created_at: datetime | None = None
    expires_at: datetime | None = None


class RestoreRequest(MutationHistoryModel):
    target_event_id: UUID
    scope: RestoreScope = RestoreScope.ALL
    expected_record_tokens: dict[UUID, str] = Field(default_factory=dict)
    expected_link_tokens: dict[str, str] = Field(default_factory=dict)
    actor_id: str | None = None
    reason: str
    idempotency_key: str
    confirmation: bool = False


class RestoreResult(MutationHistoryModel):
    status: RestoreResultStatus
    request_id: UUID | None = None
    event_id: UUID | None = None
    target_event_id: UUID
    conflict_reason: str | None = None
    conflict_details: dict[str, Any] = Field(default_factory=dict)


class MutationHistoryStore(Protocol):
    """Backend-neutral contract for authoritative mutation history storage."""

    def append_event(self, event: MutationEvent) -> MutationEvent: ...

    def append_record_revisions(self, revisions: Sequence[RecordRevision]) -> None: ...

    def append_link_revisions(self, revisions: Sequence[LinkRevision]) -> None: ...

    def get_event(self, event_id: UUID) -> MutationEvent | None: ...

    def list_events(self, *, memory_id: UUID | None = None, limit: int = 100) -> list[MutationEvent]: ...

    def get_record_revisions(self, event_id: UUID) -> list[RecordRevision]: ...

    def get_link_revisions(self, event_id: UUID) -> list[LinkRevision]: ...

    def get_protections(self, memory_id: UUID) -> list[Protection]: ...

    def set_protection(self, protection: Protection) -> Protection: ...

    def remove_protection(self, memory_id: UUID, mode: ProtectionMode) -> None: ...

    def request_restore(self, request: RestoreRequest) -> RestoreResult: ...


def semantic_snapshot(
    record: Any,
    *,
    workspace_ids: Sequence[Any] | None = None,
    lineage: dict[str, Any] | None = None,
    mutation_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return mutation-relevant state, intentionally excluding access telemetry."""
    return record_snapshot(
        record,
        workspace_ids=workspace_ids,
        lineage=lineage,
        mutation_metadata=mutation_metadata,
    )


def semantic_snapshot_token(
    record: Any,
    *,
    workspace_ids: Sequence[Any] | None = None,
    lineage: dict[str, Any] | None = None,
    mutation_metadata: dict[str, Any] | None = None,
) -> str:
    """Return the canonical token for a semantic record snapshot."""
    return record_token(
        record,
        workspace_ids=workspace_ids,
        lineage=lineage,
        mutation_metadata=mutation_metadata,
    )


# Short aliases are useful at storage boundaries without duplicating the contract.
record_semantic_snapshot = semantic_snapshot
record_semantic_token = semantic_snapshot_token
