"""SQLite persistence for the additive mutation-history foundation."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from mcp_memory.mutation_history import (
    LinkRevision,
    MutationActorKind,
    MutationEvent,
    MutationEventStatus,
    Protection,
    ProtectionMode,
    RecordRevision,
    RestoreRequest,
    RestoreResult,
    RestoreResultStatus,
    RevisionRole,
)
from mcp_memory.utils.db import DatabaseManager

MAX_HISTORY_READ_LIMIT = 1000


class MutationHistoryIdentityConflictError(ValueError):
    """Raised when an idempotency key is reused for a different request."""


class MutationHistoryTerminalizationError(ValueError):
    """Raised when a terminal result conflicts with the first terminal result."""


class SQLiteMutationHistoryStore:
    """Store immutable mutation history and current protection intent.

    This class deliberately does not call the memory repository or mutate
    production memory paths.  Its write methods are primitives for the later
    action transaction and restore executor tasks.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db = db_manager

    def append_event(self, event: MutationEvent) -> MutationEvent:
        created_at = event.created_at or _now()
        normalized = event.model_copy(update={"created_at": created_at})
        conn = self._db.get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO memory_mutation_events (
                        id, operation, actor_kind, actor_id, family, task_id,
                        curation_run_id, plan_id, action_id, provider_id,
                        reason_code, rationale, policy_version, schema_version,
                        status, restores_event_id, idempotency_key, created_at,
                        terminalized_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(normalized.id),
                        normalized.operation,
                        str(normalized.actor_kind),
                        normalized.actor_id,
                        normalized.family,
                        _uuid_text(normalized.task_id),
                        _uuid_text(normalized.curation_run_id),
                        _uuid_text(normalized.plan_id),
                        _uuid_text(normalized.action_id),
                        normalized.provider_id,
                        normalized.reason_code,
                        normalized.rationale,
                        normalized.policy_version,
                        normalized.schema_version,
                        str(normalized.status),
                        _uuid_text(normalized.restores_event_id),
                        normalized.idempotency_key,
                        _datetime_text(created_at),
                        _datetime_text(normalized.terminalized_at),
                    ),
                )
        except sqlite3.IntegrityError:
            existing = self._find_existing_event(normalized)
            if existing is None:
                raise
            return existing
        return normalized

    def get_event(self, event_id: UUID) -> MutationEvent | None:
        row = self._db.get_connection().execute(
            "SELECT * FROM memory_mutation_events WHERE id = ?",
            (str(event_id),),
        ).fetchone()
        return None if row is None else _event_from_row(row)

    def get_event_by_action(self, run_id: UUID, action_id: UUID) -> MutationEvent | None:
        row = self._db.get_connection().execute(
            """
            SELECT * FROM memory_mutation_events
            WHERE curation_run_id = ? AND action_id = ?
            LIMIT 1
            """,
            (str(run_id), str(action_id)),
        ).fetchone()
        return None if row is None else _event_from_row(row)

    def get_event_by_idempotency_key(self, idempotency_key: str) -> MutationEvent | None:
        row = self._db.get_connection().execute(
            "SELECT * FROM memory_mutation_events WHERE idempotency_key = ? LIMIT 1",
            (idempotency_key,),
        ).fetchone()
        return None if row is None else _event_from_row(row)

    def _find_existing_event(self, event: MutationEvent) -> MutationEvent | None:
        candidates = [(self.get_event(event.id), True)]
        if event.curation_run_id is not None and event.action_id is not None:
            candidates.append((self.get_event_by_action(event.curation_run_id, event.action_id), False))
        if event.idempotency_key is not None:
            candidates.append((self.get_event_by_idempotency_key(event.idempotency_key), False))
        for candidate, same_id in candidates:
            if candidate is not None:
                if not _event_identity_matches(candidate, event, include_id=same_id):
                    raise MutationHistoryIdentityConflictError(
                        f"mutation event idempotency identity is already in use for {candidate.id}"
                    )
                return candidate
        return None

    def list_events(
        self,
        *,
        memory_id: UUID | None = None,
        actor_kind: MutationActorKind | str | None = None,
        family: str | None = None,
        operation: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MutationEvent]:
        bounded_limit = _bounded_limit(limit)
        if offset < 0:
            raise ValueError("offset must not be negative")
        params: list[object] = []
        conditions: list[str] = []
        if memory_id is not None:
            conditions.append(
                "(EXISTS (SELECT 1 FROM memory_record_revisions r WHERE r.event_id = memory_mutation_events.id AND r.memory_id = ?)"
                " OR EXISTS (SELECT 1 FROM memory_link_revisions l WHERE l.event_id = memory_mutation_events.id AND (l.source_id = ? OR l.target_id = ?)))"
            )
            memory_text = str(memory_id)
            params.extend([memory_text, memory_text, memory_text])
        for column, value in (("actor_kind", actor_kind), ("family", family), ("operation", operation)):
            if value is not None:
                conditions.append(f"{column} = ?")
                params.append(str(value))
        if created_after is not None:
            conditions.append("created_at >= ?")
            params.append(created_after.isoformat())
        if created_before is not None:
            conditions.append("created_at <= ?")
            params.append(created_before.isoformat())
        query = "SELECT * FROM memory_mutation_events"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
        params.extend([bounded_limit, offset])
        rows = self._db.get_connection().execute(query, params).fetchall()
        return [_event_from_row(row) for row in rows]

    def append_record_revisions(self, revisions: Sequence[RecordRevision]) -> None:
        rows = list(revisions)
        if not rows:
            return
        conn = self._db.get_connection()
        with conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO memory_record_revisions (
                    event_id, memory_id, role, before_exists, before_snapshot,
                    after_exists, after_snapshot, before_token, after_token
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(revision.event_id),
                        str(revision.memory_id),
                        str(revision.role),
                        int(revision.before_exists),
                        _json_text(revision.before_snapshot),
                        int(revision.after_exists),
                        _json_text(revision.after_snapshot),
                        revision.before_token,
                        revision.after_token,
                    )
                    for revision in rows
                ],
            )

    def append_link_revisions(self, revisions: Sequence[LinkRevision]) -> None:
        rows = list(revisions)
        if not rows:
            return
        conn = self._db.get_connection()
        with conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO memory_link_revisions (
                    event_id, source_id, target_id, link_type, context,
                    before_exists, after_exists
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(revision.event_id),
                        str(revision.source_id),
                        str(revision.target_id),
                        revision.link_type,
                        revision.context,
                        int(revision.before_exists),
                        int(revision.after_exists),
                    )
                    for revision in rows
                ],
            )

    def get_record_revisions(self, event_id: UUID, *, limit: int | None = None) -> list[RecordRevision]:
        query = "SELECT * FROM memory_record_revisions WHERE event_id = ? ORDER BY id ASC"
        params: list[object] = [str(event_id)]
        if limit is not None:
            query += " LIMIT ?"
            params.append(_bounded_limit(limit))
        rows = self._db.get_connection().execute(
            query,
            params,
        ).fetchall()
        return [_record_revision_from_row(row) for row in rows]

    def get_link_revisions(self, event_id: UUID, *, limit: int | None = None) -> list[LinkRevision]:
        query = "SELECT * FROM memory_link_revisions WHERE event_id = ? ORDER BY id ASC"
        params: list[object] = [str(event_id)]
        if limit is not None:
            query += " LIMIT ?"
            params.append(_bounded_limit(limit))
        rows = self._db.get_connection().execute(
            query,
            params,
        ).fetchall()
        return [_link_revision_from_row(row) for row in rows]

    def get_protections(self, memory_id: UUID) -> list[Protection]:
        rows = self._db.get_connection().execute(
            "SELECT * FROM memory_protections WHERE memory_id = ? ORDER BY created_at DESC, mode ASC",
            (str(memory_id),),
        ).fetchall()
        return [_protection_from_row(row) for row in rows]

    def set_protection(self, protection: Protection) -> Protection:
        created_at = protection.created_at or _now()
        normalized = protection.model_copy(update={"created_at": created_at})
        conn = self._db.get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO memory_protections (
                    memory_id, mode, reason, actor_id, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id, mode) DO UPDATE SET
                    reason = excluded.reason,
                    actor_id = excluded.actor_id,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (
                    str(normalized.memory_id),
                    str(normalized.mode),
                    normalized.reason,
                    normalized.actor_id,
                    _datetime_text(created_at),
                    _datetime_text(normalized.expires_at),
                ),
            )
        return normalized

    def remove_protection(self, memory_id: UUID, mode: ProtectionMode) -> None:
        conn = self._db.get_connection()
        with conn:
            conn.execute(
                "DELETE FROM memory_protections WHERE memory_id = ? AND mode = ?",
                (str(memory_id), str(mode)),
            )

    def terminalize_event(
        self,
        event_id: UUID,
        status: MutationEventStatus,
        *,
        terminalized_at: datetime | None = None,
    ) -> MutationEvent:
        """Set an event's terminal result once; later conflicts are rejected."""
        event = self.get_event(event_id)
        if event is None:
            raise KeyError(f"mutation event {event_id} was not found")
        terminal_time = terminalized_at or _now()
        conn = self._db.get_connection()
        with conn:
            cursor = conn.execute(
                """
                UPDATE memory_mutation_events
                SET status = ?, terminalized_at = ?
                WHERE id = ? AND terminalized_at IS NULL
                """,
                (str(status), _datetime_text(terminal_time), str(event_id)),
            )
        updated = self.get_event(event_id)
        assert updated is not None
        if cursor.rowcount == 0 and updated.status != status:
            raise MutationHistoryTerminalizationError(
                f"mutation event {event_id} already terminalized as {updated.status}"
            )
        return updated

    def request_restore(self, request: RestoreRequest) -> RestoreResult:
        """Persist an idempotent restore request without executing the restore."""
        request_id = uuid4()
        created_at = _now()
        conn = self._db.get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO memory_restore_requests (
                        id, target_event_id, scope, expected_record_tokens,
                        expected_link_tokens, actor_id, reason, idempotency_key,
                        confirmation, status, event_id, conflict_reason,
                        conflict_details, created_at, terminalized_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, '{}', ?, NULL)
                    """,
                    (
                        str(request_id),
                        str(request.target_event_id),
                        str(request.scope),
                        _json_text({str(key): value for key, value in request.expected_record_tokens.items()}),
                        _json_text(request.expected_link_tokens),
                        request.actor_id,
                        request.reason,
                        request.idempotency_key,
                        int(request.confirmation),
                        str(RestoreResultStatus.APPLIED),
                        _datetime_text(created_at),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            existing = self._db.get_connection().execute(
                "SELECT * FROM memory_restore_requests WHERE idempotency_key = ?",
                (request.idempotency_key,),
            ).fetchone()
            if existing is None:
                raise
            if not _restore_request_identity_matches(
                request,
                target_event_id=str(existing["target_event_id"]),
                scope=str(existing["scope"]),
                expected_record_tokens=_json_value(existing["expected_record_tokens"], default={}),
                expected_link_tokens=_json_value(existing["expected_link_tokens"], default={}),
                actor_id=existing["actor_id"],
                reason=str(existing["reason"]),
                confirmation=bool(existing["confirmation"]),
            ):
                raise MutationHistoryIdentityConflictError(
                    f"restore idempotency key {request.idempotency_key!r} is already in use"
                ) from exc
            return _restore_result_from_row(existing, replay=True)
        row = self._db.get_connection().execute(
            "SELECT * FROM memory_restore_requests WHERE id = ?",
            (str(request_id),),
        ).fetchone()
        assert row is not None
        return _restore_result_from_row(row)

    def terminalize_restore_request(
        self,
        request_id: UUID,
        result: RestoreResult,
        *,
        terminalized_at: datetime | None = None,
    ) -> RestoreResult:
        """Conditionally record a restore outcome using first-terminal-write-wins."""
        conn = self._db.get_connection()
        with conn:
            cursor = conn.execute(
                """
                UPDATE memory_restore_requests
                SET status = ?, event_id = ?, conflict_reason = ?,
                    conflict_details = ?, terminalized_at = ?
                WHERE id = ? AND terminalized_at IS NULL
                """,
                (
                    str(result.status),
                    _uuid_text(result.event_id),
                    result.conflict_reason,
                    _json_text(result.conflict_details),
                    _datetime_text(terminalized_at or _now()),
                    str(request_id),
                ),
            )
        row = conn.execute(
            "SELECT * FROM memory_restore_requests WHERE id = ?",
            (str(request_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"restore request {request_id} was not found")
        stored = _restore_result_from_row(row)
        if cursor.rowcount == 0 and stored.status != result.status:
            raise MutationHistoryTerminalizationError(
                f"restore request {request_id} already terminalized as {stored.status}"
            )
        return stored


# Keep the repository spelling available to callers following existing store names.
SQLiteMutationHistoryRepository = SQLiteMutationHistoryStore


def _bounded_limit(limit: int) -> int:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    return min(limit, MAX_HISTORY_READ_LIMIT)


def _now() -> datetime:
    return datetime.now(UTC)


def _uuid_text(value: UUID | None) -> str | None:
    return None if value is None else str(value)


def _datetime_text(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _datetime_value(value: object) -> datetime | None:
    return None if value is None else datetime.fromisoformat(str(value))


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _json_value(value: object, *, default: Any) -> Any:
    if value is None:
        return default
    return json.loads(str(value))


def _event_from_row(row: sqlite3.Row) -> MutationEvent:
    return MutationEvent(
        id=UUID(str(row["id"])),
        operation=str(row["operation"]),
        actor_kind=MutationActorKind(str(row["actor_kind"])),
        actor_id=row["actor_id"],
        family=row["family"],
        task_id=None if row["task_id"] is None else UUID(str(row["task_id"])),
        curation_run_id=None if row["curation_run_id"] is None else UUID(str(row["curation_run_id"])),
        plan_id=None if row["plan_id"] is None else UUID(str(row["plan_id"])),
        action_id=None if row["action_id"] is None else UUID(str(row["action_id"])),
        provider_id=row["provider_id"],
        reason_code=row["reason_code"],
        rationale=row["rationale"],
        policy_version=row["policy_version"],
        schema_version=int(row["schema_version"]),
        status=MutationEventStatus(str(row["status"])),
        restores_event_id=None if row["restores_event_id"] is None else UUID(str(row["restores_event_id"])),
        idempotency_key=row["idempotency_key"],
        created_at=_datetime_value(row["created_at"]),
        terminalized_at=_datetime_value(row["terminalized_at"]),
    )


def _record_revision_from_row(row: sqlite3.Row) -> RecordRevision:
    return RecordRevision(
        event_id=UUID(str(row["event_id"])),
        memory_id=UUID(str(row["memory_id"])),
        role=RevisionRole(str(row["role"])),
        before_exists=bool(row["before_exists"]),
        before_snapshot=_json_value(row["before_snapshot"], default=None),
        after_exists=bool(row["after_exists"]),
        after_snapshot=_json_value(row["after_snapshot"], default=None),
        before_token=row["before_token"],
        after_token=row["after_token"],
    )


def _link_revision_from_row(row: sqlite3.Row) -> LinkRevision:
    return LinkRevision(
        event_id=UUID(str(row["event_id"])),
        source_id=UUID(str(row["source_id"])),
        target_id=UUID(str(row["target_id"])),
        link_type=str(row["link_type"]),
        context=row["context"],
        before_exists=bool(row["before_exists"]),
        after_exists=bool(row["after_exists"]),
    )


def _protection_from_row(row: sqlite3.Row) -> Protection:
    return Protection(
        memory_id=UUID(str(row["memory_id"])),
        mode=ProtectionMode(str(row["mode"])),
        reason=str(row["reason"]),
        actor_id=row["actor_id"],
        created_at=_datetime_value(row["created_at"]),
        expires_at=_datetime_value(row["expires_at"]),
    )


def _restore_result_from_row(row: sqlite3.Row, *, replay: bool = False) -> RestoreResult:
    status = RestoreResultStatus(str(row["status"]))
    if replay and status is RestoreResultStatus.APPLIED:
        status = RestoreResultStatus.ALREADY_APPLIED
    return RestoreResult(
        status=status,
        request_id=UUID(str(row["id"])),
        event_id=None if row["event_id"] is None else UUID(str(row["event_id"])),
        target_event_id=UUID(str(row["target_event_id"])),
        conflict_reason=row["conflict_reason"],
        conflict_details=_json_value(row["conflict_details"], default={}),
    )


def _event_identity_matches(left: MutationEvent, right: MutationEvent, *, include_id: bool) -> bool:
    left_data = left.model_dump(mode="json")
    right_data = right.model_dump(mode="json")
    for key in ("created_at", "terminalized_at"):
        left_data.pop(key, None)
        right_data.pop(key, None)
    if not include_id:
        left_data.pop("id", None)
        right_data.pop("id", None)
    return left_data == right_data


def _restore_request_identity_matches(
    request: RestoreRequest,
    *,
    target_event_id: str,
    scope: str,
    expected_record_tokens: Any,
    expected_link_tokens: Any,
    actor_id: object,
    reason: str,
    confirmation: bool,
) -> bool:
    return (
        target_event_id == str(request.target_event_id)
        and scope == str(request.scope)
        and expected_record_tokens == {str(key): value for key, value in request.expected_record_tokens.items()}
        and expected_link_tokens == request.expected_link_tokens
        and actor_id == request.actor_id
        and reason == request.reason
        and confirmation == request.confirmation
    )
