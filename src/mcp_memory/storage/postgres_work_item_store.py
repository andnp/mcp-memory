from __future__ import annotations

import json
import time
from uuid import uuid4

from mcp_memory.storage.postgres_store_support import optional_connection, require_connection
from mcp_memory.storage.session import DbConnectionLike, SessionManager
from mcp_memory.work_item_store import (
    DEFAULT_WORK_ITEM_LEASE_TTL_SECONDS,
    WORK_ITEM_STATUS_COMPLETED,
    WORK_ITEM_STATUS_DEFERRED,
    WORK_ITEM_STATUS_PENDING,
    WORK_ITEM_STATUS_RUNNING,
    WorkItemRecord,
)


def _coerce_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Expected int-compatible value, got {type(value)!r}")


def _coerce_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected float-compatible value, got {type(value)!r}")


def _decode_payload_object(value: object) -> dict[str, object]:
    if isinstance(value, str):
        decoded = json.loads(value)
        return decoded if isinstance(decoded, dict) else {}
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return {}


class PostgresWorkItemRepository:
    def __init__(self, session_manager: SessionManager[DbConnectionLike] | None) -> None:
        self._sessions = session_manager

    def enqueue_unique(
        self,
        *,
        family_key: str,
        execution_lane: str,
        payload: dict | None = None,
        workspace_id: str | None = None,
        priority: int = 100,
        available_at: float | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[WorkItemRecord, bool]:
        item_id = str(uuid4())
        now = time.time()
        ready_at = now if available_at is None else available_at
        payload_json = json.dumps(payload or {}, sort_keys=True)
        with require_connection(self._sessions, error="work_items_unavailable") as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO work_items (
                        id, family_key, execution_lane, workspace_id, payload_json, status,
                        priority, attempt_count, available_at, created_at, updated_at,
                        claimed_at, completed_at, lease_owner, lease_expires_at,
                        idempotency_key, last_error
                    ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, 0, %s, %s, %s, NULL, NULL, NULL, NULL, %s, NULL)
                    ON CONFLICT (idempotency_key) DO NOTHING
                    """,
                    (
                        item_id,
                        family_key,
                        execution_lane,
                        workspace_id,
                        payload_json,
                        WORK_ITEM_STATUS_PENDING,
                        priority,
                        ready_at,
                        now,
                        now,
                        idempotency_key,
                    ),
                )
                inserted = int(getattr(cursor, "rowcount", 0) or 0) == 1
            connection.commit()
        if not inserted:
            existing = self.find_by_idempotency_key(idempotency_key)
            if existing is None:
                raise RuntimeError("work_item_unique_conflict_missing_row")
            return existing, False
        return self.get_item(item_id), True

    def claim_batch(
        self,
        *,
        family_key: str,
        execution_lane: str,
        lease_owner: str,
        limit: int,
        workspace_id: str | None = None,
        now: float | None = None,
        lease_ttl_seconds: float = DEFAULT_WORK_ITEM_LEASE_TTL_SECONDS,
    ) -> list[WorkItemRecord]:
        return self._claim_items(
            family_clause="family_key = %s",
            family_params=[family_key],
            execution_lane=execution_lane,
            lease_owner=lease_owner,
            limit=limit,
            workspace_id=workspace_id,
            now=now,
            lease_ttl_seconds=lease_ttl_seconds,
        )

    def claim_compatible_batch(
        self,
        *,
        family_keys: list[str] | tuple[str, ...],
        execution_lane: str,
        lease_owner: str,
        limit: int,
        workspace_id: str | None = None,
        now: float | None = None,
        lease_ttl_seconds: float = DEFAULT_WORK_ITEM_LEASE_TTL_SECONDS,
    ) -> list[WorkItemRecord]:
        normalized_family_keys = [family_key for family_key in family_keys if family_key]
        if not normalized_family_keys:
            raise ValueError("family_keys must include at least one family")
        placeholders = ",".join(["%s"] * len(normalized_family_keys))
        return self._claim_items(
            family_clause=f"family_key IN ({placeholders})",
            family_params=list(normalized_family_keys),
            execution_lane=execution_lane,
            lease_owner=lease_owner,
            limit=limit,
            workspace_id=workspace_id,
            now=now,
            lease_ttl_seconds=lease_ttl_seconds,
        )

    def _claim_items(
        self,
        *,
        family_clause: str,
        family_params: list[object],
        execution_lane: str,
        lease_owner: str,
        limit: int,
        workspace_id: str | None,
        now: float | None,
        lease_ttl_seconds: float,
    ) -> list[WorkItemRecord]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        current_time = time.time() if now is None else now
        lease_expires_at = current_time + lease_ttl_seconds
        clauses = [
            family_clause,
            "execution_lane = %s",
            "((status IN ('pending', 'deferred') AND available_at <= %s) OR (status = 'running' AND lease_expires_at IS NOT NULL AND lease_expires_at <= %s))",
        ]
        params: list[object] = [*family_params, execution_lane, current_time, current_time]
        if workspace_id is None:
            clauses.append("workspace_id IS NULL")
        elif workspace_id != "*":
            clauses.append("workspace_id = %s")
            params.append(workspace_id)
        with optional_connection(self._sessions) as connection:
            if connection is None:
                return []
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT id FROM work_items WHERE " + " AND ".join(clauses) + " ORDER BY priority ASC, created_at ASC LIMIT %s FOR UPDATE SKIP LOCKED",
                    tuple([*params, limit]),
                )
                rows = cursor.fetchall()
                item_ids = [str(row[0]) for row in rows]
                if not item_ids:
                    connection.commit()
                    return []
                placeholders = ",".join(["%s"] * len(item_ids))
                cursor.execute(
                    f"""
                    UPDATE work_items
                    SET status = %s,
                        attempt_count = attempt_count + 1,
                        updated_at = %s,
                        claimed_at = %s,
                        lease_owner = %s,
                        lease_expires_at = %s,
                        last_error = NULL
                    WHERE id IN ({placeholders})
                    """,
                    (
                        WORK_ITEM_STATUS_RUNNING,
                        current_time,
                        current_time,
                        lease_owner,
                        lease_expires_at,
                        *item_ids,
                    ),
                )
                cursor.execute(
                    f"""
                    SELECT id, family_key, execution_lane, workspace_id, payload_json, status,
                           priority, attempt_count, available_at, created_at, updated_at,
                           claimed_at, completed_at, lease_owner, lease_expires_at,
                           idempotency_key, last_error
                    FROM work_items
                    WHERE id IN ({placeholders})
                    ORDER BY priority ASC, created_at ASC
                    """,
                    tuple(item_ids),
                )
                claimed_rows = cursor.fetchall()
            connection.commit()
        return [self._row_to_record(row) for row in claimed_rows]

    def heartbeat_item(
        self,
        item_id: str,
        *,
        lease_owner: str,
        lease_ttl_seconds: float = DEFAULT_WORK_ITEM_LEASE_TTL_SECONDS,
        heartbeated_at: float | None = None,
    ) -> WorkItemRecord:
        now = time.time() if heartbeated_at is None else heartbeated_at
        lease_expires_at = now + lease_ttl_seconds
        with require_connection(self._sessions, error="work_items_unavailable") as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE work_items
                    SET updated_at = %s,
                        lease_expires_at = %s
                    WHERE id = %s AND status = %s AND lease_owner = %s
                    """,
                    (now, lease_expires_at, item_id, WORK_ITEM_STATUS_RUNNING, lease_owner),
                )
                if int(getattr(cursor, "rowcount", 0) or 0) != 1:
                    connection.rollback()
                    raise ValueError(f"Work item {item_id} is not running for lease owner {lease_owner}")
            connection.commit()
        return self.get_item(item_id)

    def complete_item(self, item_id: str, *, completed_at: float | None = None) -> WorkItemRecord:
        if self._sessions is None:
            raise RuntimeError("work_items_unavailable")
        now = time.time() if completed_at is None else completed_at
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE work_items
                    SET status = %s,
                        updated_at = %s,
                        completed_at = %s,
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        last_error = NULL
                    WHERE id = %s AND status = %s
                    """,
                    (WORK_ITEM_STATUS_COMPLETED, now, now, item_id, WORK_ITEM_STATUS_RUNNING),
                )
                if int(getattr(cursor, "rowcount", 0) or 0) != 1:
                    connection.rollback()
                    raise ValueError(f"Work item {item_id} is not running")
            connection.commit()
        return self.get_item(item_id)

    def defer_item(
        self,
        item_id: str,
        *,
        error: str,
        retry_delay_seconds: float,
        deferred_at: float | None = None,
    ) -> WorkItemRecord:
        if self._sessions is None:
            raise RuntimeError("work_items_unavailable")
        now = time.time() if deferred_at is None else deferred_at
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE work_items
                    SET status = %s,
                        updated_at = %s,
                        available_at = %s,
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        last_error = %s
                    WHERE id = %s AND status = %s
                    """,
                    (WORK_ITEM_STATUS_DEFERRED, now, now + max(retry_delay_seconds, 0.0), error, item_id, WORK_ITEM_STATUS_RUNNING),
                )
                if int(getattr(cursor, "rowcount", 0) or 0) != 1:
                    connection.rollback()
                    raise ValueError(f"Work item {item_id} is not running")
            connection.commit()
        return self.get_item(item_id)

    def release_item(self, item_id: str, *, released_at: float | None = None) -> WorkItemRecord:
        if self._sessions is None:
            raise RuntimeError("work_items_unavailable")
        now = time.time() if released_at is None else released_at
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE work_items
                    SET status = %s,
                        updated_at = %s,
                        lease_owner = NULL,
                        lease_expires_at = NULL
                    WHERE id = %s AND status = %s
                    """,
                    (WORK_ITEM_STATUS_PENDING, now, item_id, WORK_ITEM_STATUS_RUNNING),
                )
                if int(getattr(cursor, "rowcount", 0) or 0) != 1:
                    connection.rollback()
                    raise ValueError(f"Work item {item_id} is not running")
            connection.commit()
        return self.get_item(item_id)

    def get_item(self, item_id: str) -> WorkItemRecord:
        with require_connection(self._sessions, error="work_items_unavailable") as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, family_key, execution_lane, workspace_id, payload_json, status,
                           priority, attempt_count, available_at, created_at, updated_at,
                           claimed_at, completed_at, lease_owner, lease_expires_at,
                           idempotency_key, last_error
                    FROM work_items WHERE id = %s
                    """,
                    (item_id,),
                )
                row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Work item {item_id} was not found")
        return self._row_to_record(row)

    def list_items(
        self,
        *,
        family_key: str | None = None,
        execution_lane: str | None = None,
        status: str | None = None,
        workspace_id: str | None = None,
        lease_owner: str | None = None,
        limit: int = 50,
    ) -> list[WorkItemRecord]:
        query = (
            "SELECT id, family_key, execution_lane, workspace_id, payload_json, status, priority, attempt_count, available_at, "
            "created_at, updated_at, claimed_at, completed_at, lease_owner, lease_expires_at, idempotency_key, last_error FROM work_items"
        )
        clauses: list[str] = []
        params: list[object] = []
        if family_key is not None:
            clauses.append("family_key = %s")
            params.append(family_key)
        if execution_lane is not None:
            clauses.append("execution_lane = %s")
            params.append(execution_lane)
        if status is not None:
            clauses.append("status = %s")
            params.append(status)
        if workspace_id is not None:
            clauses.append("workspace_id = %s")
            params.append(workspace_id)
        if lease_owner is not None:
            clauses.append("lease_owner = %s")
            params.append(lease_owner)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC, created_at DESC LIMIT %s"
        params.append(limit)
        with optional_connection(self._sessions) as connection:
            if connection is None:
                return []
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                return [self._row_to_record(row) for row in cursor.fetchall()]

    def find_by_idempotency_key(self, idempotency_key: str | None) -> WorkItemRecord | None:
        if idempotency_key is None:
            return None
        with optional_connection(self._sessions) as connection:
            if connection is None:
                return None
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, family_key, execution_lane, workspace_id, payload_json, status,
                           priority, attempt_count, available_at, created_at, updated_at,
                           claimed_at, completed_at, lease_owner, lease_expires_at,
                           idempotency_key, last_error
                    FROM work_items WHERE idempotency_key = %s LIMIT 1
                    """,
                    (idempotency_key,),
                )
                row = cursor.fetchone()
        return None if row is None else self._row_to_record(row)

    def _row_to_record(self, row: tuple[object, ...]) -> WorkItemRecord:
        return WorkItemRecord(
            id=str(row[0]),
            family_key=str(row[1]),
            execution_lane=str(row[2]),
            workspace_id=None if row[3] is None else str(row[3]),
            payload=_decode_payload_object(row[4]),
            status=str(row[5]),
            priority=_coerce_int(row[6]),
            attempt_count=_coerce_int(row[7]),
            available_at=_coerce_float(row[8]),
            created_at=_coerce_float(row[9]),
            updated_at=_coerce_float(row[10]),
            claimed_at=None if row[11] is None else _coerce_float(row[11]),
            completed_at=None if row[12] is None else _coerce_float(row[12]),
            lease_owner=None if row[13] is None else str(row[13]),
            lease_expires_at=None if row[14] is None else _coerce_float(row[14]),
            idempotency_key=None if row[15] is None else str(row[15]),
            last_error=None if row[16] is None else str(row[16]),
        )
