"""System 1 journal — raw thought capture and retrieval."""

from __future__ import annotations

import logging
import time

from mcp_memory.utils.db import DatabaseManager

logger = logging.getLogger(__name__)
_ALL_WORKSPACES = object()


class JournalEntry:
    """A single System 1 journal entry."""

    __slots__ = ("id", "content", "workspace_id", "timestamp", "status")

    def __init__(self, id: int, content: str, workspace_id: str | None, timestamp: float, status: str) -> None:
        self.id = id
        self.content = content
        self.workspace_id = workspace_id
        self.timestamp = timestamp
        self.status = status

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "workspace_id": self.workspace_id,
            "timestamp": self.timestamp,
            "status": self.status,
        }


class System1Journal:
    """Manages raw thought capture in the system1_journal table.

    System 1 thoughts are unprocessed observations recorded during
    AI interactions. They accumulate as 'pending' entries until a
    consolidation task processes them into refined memories.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db = db_manager

    def record(self, content: str, workspace_id: str | None = None) -> JournalEntry:
        """Record a raw thought. Returns the created entry."""
        if not content or not content.strip():
            raise ValueError("Journal content cannot be empty")

        content = content.strip()
        now = time.time()
        conn = self._db.get_connection()
        cursor = conn.execute(
            "INSERT INTO system1_journal (content, workspace_id, timestamp, status) VALUES (?, ?, ?, ?)",
            (content, workspace_id, now, "pending"),
        )
        conn.commit()
        entry_id = cursor.lastrowid
        assert entry_id is not None

        logger.info("Recorded journal entry %d (%d chars)", entry_id, len(content))
        return JournalEntry(
            id=entry_id, content=content, workspace_id=workspace_id, timestamp=now, status="pending"
        )

    def get_pending(self, limit: int = 50, workspace_id: str | None | object = _ALL_WORKSPACES) -> list[JournalEntry]:
        """Get pending entries, oldest first."""
        conn = self._db.get_connection()
        clauses = ["status = 'pending'"]
        params: list[object] = []
        if workspace_id is None:
            clauses.append("workspace_id IS NULL")
        elif workspace_id is not _ALL_WORKSPACES:
            clauses.append("workspace_id = ?")
            params.append(workspace_id)
        query = (
            "SELECT id, content, workspace_id, timestamp, status FROM system1_journal WHERE "
            + " AND ".join(clauses)
            + " ORDER BY timestamp ASC LIMIT ?"
        )
        rows = conn.execute(query, [*params, limit]).fetchall()
        return [
            JournalEntry(id=r[0], content=r[1], workspace_id=r[2], timestamp=r[3], status=r[4])
            for r in rows
        ]

    def claim_pending(
        self,
        *,
        task_id: str,
        limit: int = 50,
        workspace_id: str | None | object = _ALL_WORKSPACES,
        claimed_at: float | None = None,
    ) -> list[JournalEntry]:
        """Claim pending entries for one task, oldest first, atomically."""
        if not task_id or not task_id.strip():
            raise ValueError("task_id is required")

        now = time.time() if claimed_at is None else claimed_at
        conn = self._db.get_connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            clauses = ["status = 'pending'"]
            params: list[object] = []
            if workspace_id is None:
                clauses.append("workspace_id IS NULL")
            elif workspace_id is not _ALL_WORKSPACES:
                clauses.append("workspace_id = ?")
                params.append(workspace_id)

            rows = conn.execute(
                "SELECT id, content, workspace_id, timestamp, status FROM system1_journal WHERE "
                + " AND ".join(clauses)
                + " ORDER BY timestamp ASC LIMIT ?",
                [*params, limit],
            ).fetchall()
            if not rows:
                conn.commit()
                return []

            entry_ids = [int(row[0]) for row in rows]
            placeholders = ",".join("?" for _ in entry_ids)
            cursor = conn.execute(
                f"UPDATE system1_journal SET status = 'claimed', claim_task_id = ?, claimed_at = ? "
                f"WHERE id IN ({placeholders}) AND status = 'pending'",
                [task_id.strip(), now, *entry_ids],
            )
            if cursor.rowcount != len(entry_ids):
                conn.rollback()
                raise RuntimeError("failed to claim journal entries atomically")

            conn.commit()
            return [
                JournalEntry(id=int(row[0]), content=row[1], workspace_id=row[2], timestamp=row[3], status="claimed")
                for row in rows
            ]
        except Exception:
            conn.rollback()
            raise

    def release_claims(self, task_id: str) -> list[int]:
        """Release claimed entries back to pending for one task."""
        if not task_id or not task_id.strip():
            return []

        conn = self._db.get_connection()
        rows = conn.execute(
            "SELECT id FROM system1_journal WHERE status = 'claimed' AND claim_task_id = ? ORDER BY timestamp ASC",
            (task_id.strip(),),
        ).fetchall()
        entry_ids = [int(row[0]) for row in rows]
        if not entry_ids:
            return []

        placeholders = ",".join("?" for _ in entry_ids)
        conn.execute(
            f"UPDATE system1_journal SET status = 'pending', claim_task_id = NULL, claimed_at = NULL "
            f"WHERE id IN ({placeholders}) AND status = 'claimed' AND claim_task_id = ?",
            [*entry_ids, task_id.strip()],
        )
        conn.commit()
        return entry_ids

    def delete_claims(self, task_id: str) -> list[int]:
        """Delete claimed entries owned by one task and return their ids."""
        if not task_id or not task_id.strip():
            return []

        conn = self._db.get_connection()
        rows = conn.execute(
            "SELECT id FROM system1_journal WHERE status = 'claimed' AND claim_task_id = ? ORDER BY timestamp ASC",
            (task_id.strip(),),
        ).fetchall()
        entry_ids = [int(row[0]) for row in rows]
        if not entry_ids:
            return []

        placeholders = ",".join("?" for _ in entry_ids)
        conn.execute(
            f"DELETE FROM system1_journal WHERE id IN ({placeholders}) AND status = 'claimed' AND claim_task_id = ?",
            [*entry_ids, task_id.strip()],
        )
        conn.commit()
        return entry_ids

    def release_orphaned_claims(self) -> list[int]:
        """Release claimed thoughts whose owning task is no longer running."""
        conn = self._db.get_connection()
        rows = conn.execute(
            """
            SELECT id
            FROM system1_journal
            WHERE status = 'claimed'
              AND (
                claim_task_id IS NULL
                OR claim_task_id NOT IN (
                    SELECT id FROM tasks WHERE status = 'running'
                )
              )
            ORDER BY timestamp ASC
            """
        ).fetchall()
        entry_ids = [int(row[0]) for row in rows]
        if not entry_ids:
            return []

        placeholders = ",".join("?" for _ in entry_ids)
        conn.execute(
            f"UPDATE system1_journal SET status = 'pending', claim_task_id = NULL, claimed_at = NULL "
            f"WHERE id IN ({placeholders}) AND status = 'claimed'",
            entry_ids,
        )
        conn.commit()
        return entry_ids

    def mark_processed(self, entry_ids: list[int]) -> int:
        """Mark entries as processed. Returns count updated."""
        if not entry_ids:
            return 0
        conn = self._db.get_connection()
        placeholders = ",".join("?" for _ in entry_ids)
        cursor = conn.execute(
            f"UPDATE system1_journal SET status = 'processed', claim_task_id = NULL, claimed_at = NULL "
            f"WHERE id IN ({placeholders}) AND status = 'pending'",
            entry_ids,
        )
        conn.commit()
        return cursor.rowcount

    def mark_archived(self, entry_ids: list[int]) -> int:
        """Move processed entries to archived. Returns count updated."""
        if not entry_ids:
            return 0
        conn = self._db.get_connection()
        placeholders = ",".join("?" for _ in entry_ids)
        cursor = conn.execute(
            f"UPDATE system1_journal SET status = 'archived', claim_task_id = NULL, claimed_at = NULL "
            f"WHERE id IN ({placeholders}) AND status = 'processed'",
            entry_ids,
        )
        conn.commit()
        return cursor.rowcount

    def count_by_status(self, workspace_id: str | None | object = _ALL_WORKSPACES) -> dict[str, int]:
        """Count entries by status."""
        conn = self._db.get_connection()
        clauses: list[str] = []
        params: list[object] = []
        query = "SELECT status, COUNT(*) FROM system1_journal"
        if workspace_id is None:
            clauses.append("workspace_id IS NULL")
        elif workspace_id is not _ALL_WORKSPACES:
            clauses.append("workspace_id = ?")
            params.append(workspace_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " GROUP BY status"
        rows = conn.execute(query, params).fetchall()
        return {row[0]: row[1] for row in rows}

    def get_oldest_pending_timestamp(self, workspace_id: str | None | object = _ALL_WORKSPACES):
        conn = self._db.get_connection()
        clauses = ["status = 'pending'"]
        params: list[object] = []
        if workspace_id is None:
            clauses.append("workspace_id IS NULL")
        elif workspace_id is not _ALL_WORKSPACES:
            clauses.append("workspace_id = ?")
            params.append(workspace_id)
        row = conn.execute(
            "SELECT MIN(timestamp) FROM system1_journal WHERE " + " AND ".join(clauses),
            params,
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return float(row[0])

    def get_recent(self, limit: int = 10) -> list[JournalEntry]:
        """Get most recent entries regardless of status."""
        conn = self._db.get_connection()
        rows = conn.execute(
            "SELECT id, content, workspace_id, timestamp, status FROM system1_journal "
            "ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            JournalEntry(id=r[0], content=r[1], workspace_id=r[2], timestamp=r[3], status=r[4])
            for r in rows
        ]
