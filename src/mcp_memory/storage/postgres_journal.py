from __future__ import annotations

import time

from mcp_memory.core.journal import JournalEntry, RECOVERABLE_RETENTION_SECONDS, _ALL_WORKSPACES
from mcp_memory.storage.session import DbConnectionLike, SessionManager


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


class PostgresSystem1Journal:
    def __init__(self, session_manager: SessionManager[DbConnectionLike] | None) -> None:
        self._sessions = session_manager

    def record(self, content: str, workspace_id: str | None = None) -> JournalEntry:
        return self.record_with_timestamp(content, workspace_id=workspace_id)

    def record_with_timestamp(
        self,
        content: str,
        workspace_id: str | None = None,
        *,
        timestamp: float | None = None,
    ) -> JournalEntry:
        if self._sessions is None:
            raise RuntimeError("journal_unavailable")
        if not content or not content.strip():
            raise ValueError("Journal content cannot be empty")
        stripped = content.strip()
        now = time.time() if timestamp is None else float(timestamp)
        entry_id: int | None = None
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, content, workspace_id, timestamp, status
                    FROM system1_journal
                    WHERE content = %s
                      AND workspace_id IS NOT DISTINCT FROM %s
                      AND timestamp = %s
                    LIMIT 1
                    """,
                    (stripped, workspace_id, now),
                )
                existing_row = cursor.fetchone()
                if existing_row is not None:
                    return JournalEntry(
                        id=_coerce_int(existing_row[0]),
                        content=str(existing_row[1]),
                        workspace_id=None if existing_row[2] is None else str(existing_row[2]),
                        timestamp=_coerce_float(existing_row[3]),
                        status=str(existing_row[4]),
                    )
                cursor.execute(
                    """
                    INSERT INTO system1_journal (content, workspace_id, timestamp, status)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (stripped, workspace_id, now, "pending"),
                )
                row = cursor.fetchone()
                if row is not None:
                    entry_id = _coerce_int(row[0])
            connection.commit()
        assert entry_id is not None
        return JournalEntry(id=entry_id, content=stripped, workspace_id=workspace_id, timestamp=now, status="pending")

    def get_pending(self, limit: int = 50, workspace_id: str | None | object = _ALL_WORKSPACES) -> list[JournalEntry]:
        return self._list_entries(
            base_clause="status = 'pending'",
            workspace_id=workspace_id,
            order_by="timestamp ASC",
            limit=limit,
        )

    def claim_pending(
        self,
        *,
        task_id: str,
        limit: int = 50,
        workspace_id: str | None | object = _ALL_WORKSPACES,
        claimed_at: float | None = None,
    ) -> list[JournalEntry]:
        if self._sessions is None:
            return []
        if not task_id or not task_id.strip():
            raise ValueError("task_id is required")
        now = time.time() if claimed_at is None else claimed_at
        clauses = ["status = 'pending'"]
        params: list[object] = []
        self._append_workspace_clause(clauses, params, workspace_id)
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT id, content, workspace_id, timestamp, status FROM system1_journal WHERE "
                    + " AND ".join(clauses)
                    + " ORDER BY timestamp ASC LIMIT %s FOR UPDATE SKIP LOCKED",
                    tuple([*params, limit]),
                )
                rows = cursor.fetchall()
                entry_ids = [_coerce_int(row[0]) for row in rows]
                if not entry_ids:
                    connection.commit()
                    return []
                placeholders = ",".join(["%s"] * len(entry_ids))
                cursor.execute(
                    f"UPDATE system1_journal SET status = 'claimed', claim_task_id = %s, claimed_at = %s WHERE id IN ({placeholders}) AND status = 'pending'",
                    tuple([task_id.strip(), now, *entry_ids]),
                )
                if int(getattr(cursor, "rowcount", 0) or 0) != len(entry_ids):
                    connection.rollback()
                    raise RuntimeError("failed to claim journal entries atomically")
            connection.commit()
        return [JournalEntry(id=_coerce_int(row[0]), content=str(row[1]), workspace_id=None if row[2] is None else str(row[2]), timestamp=_coerce_float(row[3]), status="claimed") for row in rows]

    def release_claims(self, task_id: str) -> list[int]:
        if not task_id or not task_id.strip():
            return []
        return self.release_claimed_entry_ids(task_id, self.get_claimed_entry_ids(task_id))

    def move_claims_to_recoverable(self, task_id: str, *, recoverable_until: float | None = None) -> list[int]:
        if not task_id or not task_id.strip():
            return []
        return self.move_claimed_entry_ids_to_recoverable(task_id, self.get_claimed_entry_ids(task_id), recoverable_until=recoverable_until)

    def delete_claims(self, task_id: str) -> list[int]:
        if not task_id or not task_id.strip():
            return []
        return self.delete_claimed_entry_ids(task_id, self.get_claimed_entry_ids(task_id))

    def get_claimed_entry_ids(self, task_id: str) -> list[int]:
        return self._normalize_claimed_entry_ids(task_id, None)

    def release_claimed_entry_ids(self, task_id: str, entry_ids: list[int]) -> list[int]:
        claimed_entry_ids = self._normalize_claimed_entry_ids(task_id, entry_ids)
        if not claimed_entry_ids or self._sessions is None:
            return []
        placeholders = ",".join(["%s"] * len(claimed_entry_ids))
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"UPDATE system1_journal SET status = 'pending', claim_task_id = NULL, claimed_at = NULL, recoverable_until = NULL WHERE id IN ({placeholders}) AND status = 'claimed' AND claim_task_id = %s",
                    tuple([*claimed_entry_ids, task_id.strip()]),
                )
            connection.commit()
        return claimed_entry_ids

    def move_claimed_entry_ids_to_recoverable(
        self,
        task_id: str,
        entry_ids: list[int],
        *,
        recoverable_until: float | None = None,
    ) -> list[int]:
        claimed_entry_ids = self._normalize_claimed_entry_ids(task_id, entry_ids)
        if not claimed_entry_ids or self._sessions is None:
            return []
        expiry_timestamp = time.time() + RECOVERABLE_RETENTION_SECONDS if recoverable_until is None else recoverable_until
        placeholders = ",".join(["%s"] * len(claimed_entry_ids))
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"UPDATE system1_journal SET status = 'recoverable', claim_task_id = NULL, claimed_at = NULL, recoverable_until = %s WHERE id IN ({placeholders}) AND status = 'claimed' AND claim_task_id = %s",
                    tuple([expiry_timestamp, *claimed_entry_ids, task_id.strip()]),
                )
            connection.commit()
        return claimed_entry_ids

    def delete_claimed_entry_ids(self, task_id: str, entry_ids: list[int]) -> list[int]:
        claimed_entry_ids = self._normalize_claimed_entry_ids(task_id, entry_ids)
        if not claimed_entry_ids or self._sessions is None:
            return []
        placeholders = ",".join(["%s"] * len(claimed_entry_ids))
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM system1_journal WHERE id IN ({placeholders}) AND status = 'claimed' AND claim_task_id = %s",
                    tuple([*claimed_entry_ids, task_id.strip()]),
                )
            connection.commit()
        return claimed_entry_ids

    def release_orphaned_claims(self) -> list[int]:
        if self._sessions is None:
            return []
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id
                    FROM system1_journal
                    WHERE status = 'claimed'
                      AND (
                        claim_task_id IS NULL
                        OR claim_task_id NOT IN (SELECT id FROM tasks WHERE status = 'running')
                      )
                    ORDER BY timestamp ASC
                    """,
                    (),
                )
                rows = cursor.fetchall()
                entry_ids = [_coerce_int(row[0]) for row in rows]
                if not entry_ids:
                    connection.commit()
                    return []
                placeholders = ",".join(["%s"] * len(entry_ids))
                cursor.execute(
                    f"UPDATE system1_journal SET status = 'pending', claim_task_id = NULL, claimed_at = NULL, recoverable_until = NULL WHERE id IN ({placeholders}) AND status = 'claimed'",
                    tuple(entry_ids),
                )
            connection.commit()
        return entry_ids

    def purge_expired_recoverable(self, *, now: float | None = None) -> list[int]:
        if self._sessions is None:
            return []
        cutoff = time.time() if now is None else now
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id FROM system1_journal
                    WHERE status = 'recoverable' AND recoverable_until IS NOT NULL AND recoverable_until <= %s
                    ORDER BY timestamp ASC
                    """,
                    (cutoff,),
                )
                rows = cursor.fetchall()
                entry_ids = [_coerce_int(row[0]) for row in rows]
                if not entry_ids:
                    connection.commit()
                    return []
                placeholders = ",".join(["%s"] * len(entry_ids))
                cursor.execute(
                    f"DELETE FROM system1_journal WHERE id IN ({placeholders}) AND status = 'recoverable'",
                    tuple(entry_ids),
                )
            connection.commit()
        return entry_ids

    def mark_processed(self, entry_ids: list[int]) -> int:
        return self._update_status(entry_ids, from_status="pending", to_status="processed")

    def mark_archived(self, entry_ids: list[int]) -> int:
        return self._update_status(entry_ids, from_status="processed", to_status="archived")

    def _update_status(self, entry_ids: list[int], *, from_status: str, to_status: str) -> int:
        if not entry_ids or self._sessions is None:
            return 0
        placeholders = ",".join(["%s"] * len(entry_ids))
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"UPDATE system1_journal SET status = %s, claim_task_id = NULL, claimed_at = NULL, recoverable_until = NULL WHERE id IN ({placeholders}) AND status = %s",
                    tuple([to_status, *entry_ids, from_status]),
                )
                updated = int(getattr(cursor, "rowcount", 0) or 0)
            connection.commit()
        return updated

    def count_by_status(self, workspace_id: str | None | object = _ALL_WORKSPACES) -> dict[str, int]:
        if self._sessions is None:
            return {}
        clauses: list[str] = []
        params: list[object] = []
        self._append_workspace_clause(clauses, params, workspace_id)
        query = "SELECT status, COUNT(*) FROM system1_journal"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " GROUP BY status"
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
        return {str(row[0]): _coerce_int(row[1]) for row in rows}

    def get_oldest_pending_timestamp(self, workspace_id: str | None | object = _ALL_WORKSPACES):
        if self._sessions is None:
            return None
        clauses = ["status = 'pending'"]
        params: list[object] = []
        self._append_workspace_clause(clauses, params, workspace_id)
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT MIN(timestamp) FROM system1_journal WHERE " + " AND ".join(clauses), tuple(params))
                row = cursor.fetchone()
        if row is None or row[0] is None:
            return None
        return _coerce_float(row[0])

    def get_latest_thought_timestamp(self, workspace_id: str | None | object = _ALL_WORKSPACES) -> float | None:
        if self._sessions is None:
            return None
        clauses: list[str] = []
        params: list[object] = []
        self._append_workspace_clause(clauses, params, workspace_id)
        query = "SELECT MAX(timestamp) FROM system1_journal"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                row = cursor.fetchone()
        if row is None or row[0] is None:
            return None
        return _coerce_float(row[0])

    def get_recent(self, limit: int = 10) -> list[JournalEntry]:
        return self._list_entries(base_clause=None, workspace_id=_ALL_WORKSPACES, order_by="timestamp DESC", limit=limit)

    def _normalize_claimed_entry_ids(self, task_id: str, entry_ids: list[int] | None) -> list[int]:
        if self._sessions is None or not task_id or not task_id.strip():
            return []
        params: list[object] = [task_id.strip()]
        query = "SELECT id FROM system1_journal WHERE status = 'claimed' AND claim_task_id = %s"
        if entry_ids is not None:
            normalized_entry_ids = [entry_id for entry_id in entry_ids if isinstance(entry_id, int) and not isinstance(entry_id, bool) and entry_id > 0]
            if not normalized_entry_ids:
                return []
            placeholders = ",".join(["%s"] * len(normalized_entry_ids))
            query += f" AND id IN ({placeholders})"
            params.extend(normalized_entry_ids)
        query += " ORDER BY timestamp ASC"
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
        return [_coerce_int(row[0]) for row in rows]

    def _list_entries(
        self,
        *,
        base_clause: str | None,
        workspace_id: str | None | object,
        order_by: str,
        limit: int,
    ) -> list[JournalEntry]:
        if self._sessions is None:
            return []
        clauses: list[str] = []
        params: list[object] = []
        if base_clause is not None:
            clauses.append(base_clause)
        self._append_workspace_clause(clauses, params, workspace_id)
        query = "SELECT id, content, workspace_id, timestamp, status FROM system1_journal"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += f" ORDER BY {order_by} LIMIT %s"
        params.append(limit)
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
        return [
            JournalEntry(
                id=_coerce_int(row[0]),
                content=str(row[1]),
                workspace_id=None if row[2] is None else str(row[2]),
                timestamp=_coerce_float(row[3]),
                status=str(row[4]),
            )
            for row in rows
        ]

    def _append_workspace_clause(self, clauses: list[str], params: list[object], workspace_id: str | None | object) -> None:
        if workspace_id is None:
            clauses.append("workspace_id IS NULL")
        elif workspace_id is not _ALL_WORKSPACES:
            clauses.append("workspace_id = %s")
            params.append(workspace_id)
