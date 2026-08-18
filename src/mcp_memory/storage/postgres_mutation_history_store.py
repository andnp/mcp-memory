"""Postgres persistence for the additive mutation-history foundation."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
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
from mcp_memory.mutation_history_store import (
    MAX_HISTORY_READ_LIMIT,
    MutationHistoryIdentityConflictError,
    MutationHistoryTerminalizationError,
    _event_identity_matches,
    _restore_request_identity_matches,
)
from mcp_memory.storage.session import CursorLike, DbConnectionLike, SessionManager


class PostgresMutationHistoryStore:
    """Store immutable mutation history and current protection intent.

    This is an additive primitive. It deliberately does not call the memory
    repository or integrate with production mutation paths.
    """

    def __init__(self, session_manager: SessionManager[DbConnectionLike] | None) -> None:
        self._sessions = session_manager

    def append_event(self, event: MutationEvent) -> MutationEvent:
        normalized = event.model_copy(update={"created_at": event.created_at or _now()})
        with self._open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO memory_mutation_events (
                        id, operation, actor_kind, actor_id, family, task_id,
                        curation_run_id, plan_id, action_id, provider_id,
                        reason_code, rationale, policy_version, schema_version,
                        status, restores_event_id, idempotency_key, created_at,
                        terminalized_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
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
                        _datetime_text(normalized.created_at),
                        _datetime_text(normalized.terminalized_at),
                    ),
                )
            connection.commit()
        existing = self._find_existing_event(normalized)
        if existing is None:
            raise RuntimeError("mutation event insert was not persisted")
        return existing

    def get_event(self, event_id: UUID) -> MutationEvent | None:
        row = self._fetchone("SELECT * FROM memory_mutation_events WHERE id = %s", (str(event_id),))
        return None if row is None else _event_from_row(row)

    def get_event_by_action(self, run_id: UUID, action_id: UUID) -> MutationEvent | None:
        row = self._fetchone(
            "SELECT * FROM memory_mutation_events WHERE curation_run_id = %s AND action_id = %s LIMIT 1",
            (str(run_id), str(action_id)),
        )
        return None if row is None else _event_from_row(row)

    def get_event_by_idempotency_key(self, idempotency_key: str) -> MutationEvent | None:
        row = self._fetchone(
            "SELECT * FROM memory_mutation_events WHERE idempotency_key = %s LIMIT 1",
            (idempotency_key,),
        )
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
                "(EXISTS (SELECT 1 FROM memory_record_revisions r WHERE r.event_id = memory_mutation_events.id AND r.memory_id = %s)"
                " OR EXISTS (SELECT 1 FROM memory_link_revisions l WHERE l.event_id = memory_mutation_events.id AND (l.source_id = %s OR l.target_id = %s)))"
            )
            memory_text = str(memory_id)
            params.extend([memory_text, memory_text, memory_text])
        for column, value in (("actor_kind", actor_kind), ("family", family), ("operation", operation)):
            if value is not None:
                conditions.append(f"{column} = %s")
                params.append(str(value))
        if created_after is not None:
            conditions.append("created_at >= %s")
            params.append(created_after)
        if created_before is not None:
            conditions.append("created_at <= %s")
            params.append(created_before)
        query = "SELECT * FROM memory_mutation_events"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s"
        params.extend([bounded_limit, offset])
        return [_event_from_row(row) for row in self._fetchall(query, tuple(params))]

    def append_record_revisions(self, revisions: Sequence[RecordRevision]) -> None:
        rows = list(revisions)
        if not rows:
            return
        with self._open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO memory_record_revisions (
                        event_id, memory_id, role, before_exists, before_snapshot,
                        after_exists, after_snapshot, before_token, after_token
                    ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, %s)
                    ON CONFLICT (event_id, memory_id, role) DO NOTHING
                    """,
                    [
                        (
                            str(revision.event_id),
                            str(revision.memory_id),
                            str(revision.role),
                            revision.before_exists,
                            _json_text(revision.before_snapshot),
                            revision.after_exists,
                            _json_text(revision.after_snapshot),
                            revision.before_token,
                            revision.after_token,
                        )
                        for revision in rows
                    ],
                )
            connection.commit()

    def append_link_revisions(self, revisions: Sequence[LinkRevision]) -> None:
        rows = list(revisions)
        if not rows:
            return
        with self._open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO memory_link_revisions (
                        event_id, source_id, target_id, link_type, context,
                        before_exists, after_exists
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (event_id, source_id, target_id, link_type) DO NOTHING
                    """,
                    [
                        (
                            str(revision.event_id),
                            str(revision.source_id),
                            str(revision.target_id),
                            revision.link_type,
                            revision.context,
                            revision.before_exists,
                            revision.after_exists,
                        )
                        for revision in rows
                    ],
                )
            connection.commit()

    def get_record_revisions(self, event_id: UUID, *, limit: int | None = None) -> list[RecordRevision]:
        query = "SELECT * FROM memory_record_revisions WHERE event_id = %s ORDER BY id ASC"
        params: list[object] = [str(event_id)]
        if limit is not None:
            query += " LIMIT %s"
            params.append(_bounded_limit(limit))
        rows = self._fetchall(
            query,
            tuple(params),
        )
        return [_record_revision_from_row(row) for row in rows]

    def get_link_revisions(self, event_id: UUID, *, limit: int | None = None) -> list[LinkRevision]:
        query = "SELECT * FROM memory_link_revisions WHERE event_id = %s ORDER BY id ASC"
        params: list[object] = [str(event_id)]
        if limit is not None:
            query += " LIMIT %s"
            params.append(_bounded_limit(limit))
        rows = self._fetchall(
            query,
            tuple(params),
        )
        return [_link_revision_from_row(row) for row in rows]

    def get_protections(self, memory_id: UUID) -> list[Protection]:
        rows = self._fetchall(
            "SELECT * FROM memory_protections WHERE memory_id = %s ORDER BY created_at DESC, mode ASC",
            (str(memory_id),),
        )
        return [_protection_from_row(row) for row in rows]

    def set_protection(self, protection: Protection) -> Protection:
        normalized = protection.model_copy(update={"created_at": protection.created_at or _now()})
        with self._open_connection() as connection:
            with connection.cursor() as cursor:
                self._lock_targets(cursor, [str(normalized.memory_id)])
                cursor.execute(
                    """
                    INSERT INTO memory_protections (memory_id, mode, reason, actor_id, created_at, expires_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (memory_id, mode) DO UPDATE SET
                        reason = EXCLUDED.reason,
                        actor_id = EXCLUDED.actor_id,
                        created_at = EXCLUDED.created_at,
                        expires_at = EXCLUDED.expires_at
                    """,
                    (
                        str(normalized.memory_id),
                        str(normalized.mode),
                        normalized.reason,
                        normalized.actor_id,
                        _datetime_text(normalized.created_at),
                        _datetime_text(normalized.expires_at),
                    ),
                )
            connection.commit()
        return normalized

    def remove_protection(self, memory_id: UUID, mode: ProtectionMode) -> None:
        with self._open_connection() as connection:
            with connection.cursor() as cursor:
                self._lock_targets(cursor, [str(memory_id)])
                cursor.execute(
                    "DELETE FROM memory_protections WHERE memory_id = %s AND mode = %s",
                    (str(memory_id), str(mode)),
                )
            connection.commit()

    def _lock_targets(self, cursor: CursorLike, target_ids: Sequence[str]) -> None:
        """Lock existing targets in the action store's canonical order.

        Protection rows intentionally outlive deleted memories, so a missing
        target has no row to lock and remains a valid protection write. The
        attempted lock still uses the same transaction and ordering as an
        action for every existing target.
        """
        for memory_id in sorted(target_ids, key=lambda value: value.encode("utf-8")):
            cursor.execute("SELECT id FROM memories WHERE id = %s FOR UPDATE", (memory_id,))
            cursor.fetchone()

    def terminalize_event(
        self,
        event_id: UUID,
        status: MutationEventStatus,
        *,
        terminalized_at: datetime | None = None,
    ) -> MutationEvent:
        terminal_time = terminalized_at or _now()
        with self._open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE memory_mutation_events
                    SET status = %s, terminalized_at = %s
                    WHERE id = %s AND terminalized_at IS NULL
                    """,
                    (str(status), _datetime_text(terminal_time), str(event_id)),
                )
                updated = int(getattr(cursor, "rowcount", 0) or 0)
            connection.commit()
        event = self.get_event(event_id)
        if event is None:
            raise KeyError(f"mutation event {event_id} was not found")
        if updated == 0 and event.status != status:
            raise MutationHistoryTerminalizationError(
                f"mutation event {event_id} already terminalized as {event.status}"
            )
        return event

    def request_restore(self, request: RestoreRequest) -> RestoreResult:
        request_id = uuid4()
        created_at = _now()
        with self._open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO memory_restore_requests (
                        id, target_event_id, scope, expected_record_tokens,
                        expected_link_tokens, actor_id, reason, idempotency_key,
                        confirmation, status, event_id, conflict_reason,
                        conflict_details, created_at, terminalized_at
                    ) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s, NULL, NULL, '{}'::jsonb, %s, NULL)
                    ON CONFLICT (idempotency_key) DO NOTHING
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
                        request.confirmation,
                        str(RestoreResultStatus.APPLIED),
                        _datetime_text(created_at),
                    ),
                )
            connection.commit()
        row = self._fetchone(
            "SELECT * FROM memory_restore_requests WHERE idempotency_key = %s",
            (request.idempotency_key,),
        )
        if row is None:
            raise RuntimeError("restore request insert was not persisted")
        if not _restore_request_identity_matches(
            request,
            target_event_id=str(row[1]),
            scope=str(row[2]),
            expected_record_tokens=_json_value(row[3], default={}),
            expected_link_tokens=_json_value(row[4], default={}),
            actor_id=row[5],
            reason=str(row[6]),
            confirmation=bool(row[8]),
        ):
            raise MutationHistoryIdentityConflictError(
                f"restore idempotency key {request.idempotency_key!r} is already in use"
            )
        return _restore_result_from_row(row, replay=str(row[0]) != str(request_id))

    def terminalize_restore_request(
        self,
        request_id: UUID,
        result: RestoreResult,
        *,
        terminalized_at: datetime | None = None,
    ) -> RestoreResult:
        with self._open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE memory_restore_requests
                    SET status = %s, event_id = %s, conflict_reason = %s,
                        conflict_details = %s::jsonb, terminalized_at = %s
                    WHERE id = %s AND terminalized_at IS NULL
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
                updated = int(getattr(cursor, "rowcount", 0) or 0)
            connection.commit()
        row = self._fetchone("SELECT * FROM memory_restore_requests WHERE id = %s", (str(request_id),))
        if row is None:
            raise KeyError(f"restore request {request_id} was not found")
        stored = _restore_result_from_row(row)
        if updated == 0 and stored.status != result.status:
            raise MutationHistoryTerminalizationError(
                f"restore request {request_id} already terminalized as {stored.status}"
            )
        return stored

    @contextmanager
    def _open_connection(self) -> Iterator[DbConnectionLike]:
        if self._sessions is None:
            raise RuntimeError("mutation_history_unavailable")
        with self._sessions.open_connection() as connection:
            yield connection

    def _fetchone(self, query: str, params: tuple[object, ...]) -> tuple[object, ...] | None:
        if self._sessions is None:
            return None
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchone()

    def _fetchall(self, query: str, params: tuple[object, ...]) -> list[tuple[object, ...]]:
        if self._sessions is None:
            return []
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()


def _bounded_limit(limit: int) -> int:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    return min(limit, MAX_HISTORY_READ_LIMIT)


def _now() -> datetime:
    from datetime import UTC

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
    return json.loads(value) if isinstance(value, str) else value


def _event_from_row(row: tuple[object, ...]) -> MutationEvent:
    return MutationEvent(
        id=UUID(str(row[0])),
        operation=str(row[1]),
        actor_kind=MutationActorKind(str(row[2])),
        actor_id=_optional_text(row[3]),
        family=_optional_text(row[4]),
        task_id=None if row[5] is None else UUID(str(row[5])),
        curation_run_id=None if row[6] is None else UUID(str(row[6])),
        plan_id=None if row[7] is None else UUID(str(row[7])),
        action_id=None if row[8] is None else UUID(str(row[8])),
        provider_id=_optional_text(row[9]),
        reason_code=_optional_text(row[10]),
        rationale=_optional_text(row[11]),
        policy_version=_optional_text(row[12]),
        schema_version=int(str(row[13])),
        status=MutationEventStatus(str(row[14])),
        restores_event_id=None if row[15] is None else UUID(str(row[15])),
        idempotency_key=_optional_text(row[16]),
        created_at=_datetime_value(row[17]),
        terminalized_at=_datetime_value(row[18]),
    )


def _record_revision_from_row(row: tuple[object, ...]) -> RecordRevision:
    return RecordRevision(
        event_id=UUID(str(row[1])),
        memory_id=UUID(str(row[2])),
        role=RevisionRole(str(row[3])),
        before_exists=bool(row[4]),
        before_snapshot=_json_value(row[5], default=None),
        after_exists=bool(row[6]),
        after_snapshot=_json_value(row[7], default=None),
        before_token=_optional_text(row[8]),
        after_token=_optional_text(row[9]),
    )


def _link_revision_from_row(row: tuple[object, ...]) -> LinkRevision:
    return LinkRevision(
        event_id=UUID(str(row[1])),
        source_id=UUID(str(row[2])),
        target_id=UUID(str(row[3])),
        link_type=str(row[4]),
        context=_optional_text(row[5]),
        before_exists=bool(row[6]),
        after_exists=bool(row[7]),
    )


def _protection_from_row(row: tuple[object, ...]) -> Protection:
    return Protection(
        memory_id=UUID(str(row[0])),
        mode=ProtectionMode(str(row[1])),
        reason=str(row[2]),
        actor_id=_optional_text(row[3]),
        created_at=_datetime_value(row[4]),
        expires_at=_datetime_value(row[5]),
    )


def _restore_result_from_row(row: tuple[object, ...], *, replay: bool = False) -> RestoreResult:
    status = RestoreResultStatus(str(row[9]))
    if replay and status is RestoreResultStatus.APPLIED:
        status = RestoreResultStatus.ALREADY_APPLIED
    return RestoreResult(
        status=status,
        request_id=UUID(str(row[0])),
        event_id=None if row[10] is None else UUID(str(row[10])),
        target_event_id=UUID(str(row[1])),
        conflict_reason=_optional_text(row[11]),
        conflict_details=_json_value(row[12], default={}),
    )


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)
