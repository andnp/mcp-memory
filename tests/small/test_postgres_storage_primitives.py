from __future__ import annotations

import json

import pytest

from mcp_memory.core.journal import _ALL_WORKSPACES
from mcp_memory.storage.postgres_embedding_repair_store import PostgresEmbeddingRepairQueue
from mcp_memory.storage.postgres_journal import PostgresSystem1Journal
from mcp_memory.storage.postgres_vector_store import PostgresVectorStore
from mcp_memory.storage.postgres_work_item_store import PostgresWorkItemRepository
from mcp_memory.work_item_store import EXECUTION_LANE_DETERMINISTIC


pytestmark = pytest.mark.small


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Expected int-compatible value, got {type(value)!r}")


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected float-compatible value, got {type(value)!r}")


class FakePrimitiveState:
    def __init__(self) -> None:
        self.system1_journal: list[dict[str, object]] = []
        self.next_journal_id = 1
        self.work_items: list[dict[str, object]] = []
        self.embedding_repair_queue: list[dict[str, object]] = []
        self.embeddings: list[dict[str, object]] = []


class FakePrimitiveCursor:
    def __init__(self, state: FakePrimitiveState) -> None:
        self._state = state
        self._result: list[tuple[object, ...]] = []
        self.rowcount = 0

    def __enter__(self) -> FakePrimitiveCursor:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        normalized = " ".join(query.split())
        arguments = tuple(() if params is None else params)
        self.rowcount = 0
        self._result = []

        if normalized.startswith("INSERT INTO system1_journal"):
            entry = {
                "id": self._state.next_journal_id,
                "content": str(arguments[0]),
                "workspace_id": arguments[1],
                "timestamp": _as_float(arguments[2]),
                "status": str(arguments[3]),
                "claim_task_id": None,
                "claimed_at": None,
                "recoverable_until": None,
            }
            self._state.next_journal_id += 1
            self._state.system1_journal.append(entry)
            self._result = [(entry["id"],)]
            self.rowcount = 1
            return
        if normalized.startswith("SELECT id, content, workspace_id, timestamp, status FROM system1_journal"):
            rows = self._filter_journal_rows(normalized, arguments) if " WHERE " in normalized else list(self._state.system1_journal)
            if "ORDER BY timestamp DESC" in normalized:
                rows.sort(key=lambda row: _as_float(row["timestamp"]), reverse=True)
            else:
                rows.sort(key=lambda row: _as_float(row["timestamp"]))
            limit = _as_int(arguments[-1])
            self._result = [
                (row["id"], row["content"], row["workspace_id"], row["timestamp"], row["status"])
                for row in rows[:limit]
            ]
            return
        if normalized.startswith("UPDATE system1_journal SET status = 'claimed'"):
            task_id = str(arguments[0])
            claimed_at = _as_float(arguments[1])
            entry_ids = {_as_int(value) for value in arguments[2:]}
            updated = 0
            for row in self._state.system1_journal:
                if _as_int(row["id"]) in entry_ids and str(row["status"]) == "pending":
                    row["status"] = "claimed"
                    row["claim_task_id"] = task_id
                    row["claimed_at"] = claimed_at
                    updated += 1
            self.rowcount = updated
            return
        if normalized.startswith("SELECT status, COUNT(*) FROM system1_journal"):
            rows = self._filter_journal_group_rows(normalized, arguments)
            counts: dict[str, int] = {}
            for row in rows:
                status = str(row["status"])
                counts[status] = counts.get(status, 0) + 1
            self._result = [(status, count) for status, count in counts.items()]
            return
        if normalized.startswith("SELECT MIN(timestamp) FROM system1_journal"):
            rows = self._filter_journal_rows(normalized, arguments, include_limit=False)
            minimum = None if not rows else min(_as_float(row["timestamp"]) for row in rows)
            self._result = [(minimum,)]
            return
        if normalized.startswith("SELECT MAX(timestamp) FROM system1_journal"):
            rows = list(self._state.system1_journal)
            maximum = None if not rows else max(_as_float(row["timestamp"]) for row in rows)
            self._result = [(maximum,)]
            return
        if normalized.startswith("SELECT id FROM system1_journal WHERE status = 'claimed' AND claim_task_id = %s"):
            task_id = str(arguments[0])
            entry_ids = None if "AND id IN (" not in normalized else {_as_int(value) for value in arguments[1:]}
            rows = [
                row for row in self._state.system1_journal
                if str(row["status"]) == "claimed" and str(row["claim_task_id"]) == task_id and (entry_ids is None or _as_int(row["id"]) in entry_ids)
            ]
            rows.sort(key=lambda row: _as_float(row["timestamp"]))
            self._result = [(row["id"],) for row in rows]
            return
        if normalized.startswith("UPDATE system1_journal SET status = 'recoverable'"):
            recoverable_until = _as_float(arguments[0])
            task_id = str(arguments[-1])
            entry_ids = {_as_int(value) for value in arguments[1:-1]}
            for row in self._state.system1_journal:
                if _as_int(row["id"]) in entry_ids and str(row["status"]) == "claimed" and str(row["claim_task_id"]) == task_id:
                    row["status"] = "recoverable"
                    row["claim_task_id"] = None
                    row["claimed_at"] = None
                    row["recoverable_until"] = recoverable_until
            return
        if normalized.startswith("INSERT INTO work_items"):
            idempotency_key = arguments[10]
            if idempotency_key is not None and any(row["idempotency_key"] == idempotency_key for row in self._state.work_items):
                self.rowcount = 0
                return
            row = {
                "id": str(arguments[0]),
                "family_key": str(arguments[1]),
                "execution_lane": str(arguments[2]),
                "workspace_id": arguments[3],
                "payload_json": arguments[4],
                "status": str(arguments[5]),
                "priority": _as_int(arguments[6]),
                "attempt_count": 0,
                "available_at": _as_float(arguments[7]),
                "created_at": _as_float(arguments[8]),
                "updated_at": _as_float(arguments[9]),
                "claimed_at": None,
                "completed_at": None,
                "lease_owner": None,
                "lease_expires_at": None,
                "idempotency_key": idempotency_key,
                "last_error": None,
            }
            self._state.work_items.append(row)
            self.rowcount = 1
            return
        if normalized.startswith("SELECT id FROM work_items WHERE"):
            rows = self._filter_work_items_for_claim(normalized, arguments)
            limit = _as_int(arguments[-1])
            self._result = [(row["id"],) for row in rows[:limit]]
            return
        if normalized.startswith("UPDATE work_items SET status = %s, attempt_count = attempt_count + 1"):
            item_ids = {str(value) for value in arguments[5:]}
            for row in self._state.work_items:
                if str(row["id"]) in item_ids:
                    row["status"] = str(arguments[0])
                    row["attempt_count"] = _as_int(row["attempt_count"]) + 1
                    row["updated_at"] = _as_float(arguments[1])
                    row["claimed_at"] = _as_float(arguments[2])
                    row["lease_owner"] = str(arguments[3])
                    row["lease_expires_at"] = _as_float(arguments[4])
                    row["last_error"] = None
            return
        if normalized.startswith("SELECT id, family_key, execution_lane, workspace_id, payload_json, status") and "FROM work_items WHERE id IN (" in normalized:
            item_ids = {str(value) for value in arguments}
            rows = [row for row in self._state.work_items if str(row["id"]) in item_ids]
            rows.sort(key=lambda row: (_as_int(row["priority"]), _as_float(row["created_at"])))
            self._result = [self._work_item_row(row) for row in rows]
            return
        if normalized.startswith("SELECT id, family_key, execution_lane, workspace_id, payload_json, status") and "FROM work_items WHERE id = %s" in normalized:
            item_id = str(arguments[0])
            row = next((row for row in self._state.work_items if str(row["id"]) == item_id), None)
            self._result = [] if row is None else [self._work_item_row(row)]
            return
        if normalized.startswith("SELECT id, family_key, execution_lane, workspace_id, payload_json, status") and "FROM work_items WHERE idempotency_key = %s" in normalized:
            key = arguments[0]
            row = next((row for row in self._state.work_items if row["idempotency_key"] == key), None)
            self._result = [] if row is None else [self._work_item_row(row)]
            return
        if normalized.startswith("UPDATE work_items SET updated_at = %s, lease_expires_at = %s"):
            item_id = str(arguments[2])
            lease_owner = str(arguments[4])
            updated = 0
            for row in self._state.work_items:
                if str(row["id"]) == item_id and str(row["status"]) == str(arguments[3]) and str(row["lease_owner"]) == lease_owner:
                    row["updated_at"] = _as_float(arguments[0])
                    row["lease_expires_at"] = _as_float(arguments[1])
                    updated = 1
            self.rowcount = updated
            return
        if normalized.startswith("UPDATE work_items SET status = %s, updated_at = %s, completed_at = %s"):
            item_id = str(arguments[3])
            updated = 0
            for row in self._state.work_items:
                if str(row["id"]) == item_id and str(row["status"]) == str(arguments[4]):
                    row["status"] = str(arguments[0])
                    row["updated_at"] = _as_float(arguments[1])
                    row["completed_at"] = _as_float(arguments[2])
                    row["lease_owner"] = None
                    row["lease_expires_at"] = None
                    row["last_error"] = None
                    updated = 1
            self.rowcount = updated
            return
        if normalized.startswith("INSERT INTO embedding_repair_queue"):
            duplicate = next(
                (
                    row for row in self._state.embedding_repair_queue
                    if str(row["memory_id"]) == str(arguments[1])
                    and str(row["model_name"]) == str(arguments[3])
                    and str(row["memory_updated_at"]) == str(arguments[4])
                ),
                None,
            )
            if duplicate is not None:
                self.rowcount = 0
                return
            row = {
                "id": str(arguments[0]),
                "memory_id": str(arguments[1]),
                "workspace_id": arguments[2],
                "model_name": str(arguments[3]),
                "memory_updated_at": str(arguments[4]),
                "status": str(arguments[5]),
                "attempt_count": 0,
                "available_at": _as_float(arguments[6]),
                "created_at": _as_float(arguments[7]),
                "updated_at": _as_float(arguments[8]),
                "claimed_at": None,
                "completed_at": None,
                "lease_owner": None,
                "lease_expires_at": None,
                "last_error": None,
            }
            self._state.embedding_repair_queue.append(row)
            self.rowcount = 1
            return
        if normalized.startswith("SELECT id, memory_id, workspace_id, model_name, memory_updated_at, status") and "FROM embedding_repair_queue WHERE memory_id = %s" in normalized:
            row = next(
                (
                    row for row in self._state.embedding_repair_queue
                    if str(row["memory_id"]) == str(arguments[0])
                    and str(row["model_name"]) == str(arguments[1])
                    and str(row["memory_updated_at"]) == str(arguments[2])
                ),
                None,
            )
            self._result = [] if row is None else [self._repair_row(row)]
            return
        if normalized.startswith("SELECT id FROM embedding_repair_queue WHERE status = %s"):
            status = str(arguments[0])
            completed_before = _as_float(arguments[1])
            limit = _as_int(arguments[2])
            rows = [
                row for row in self._state.embedding_repair_queue
                if str(row["status"]) == status and row["completed_at"] is not None and _as_float(row["completed_at"]) <= completed_before
            ]
            rows.sort(key=lambda row: _as_float(row["completed_at"]))
            self._result = [(row["id"],) for row in rows[:limit]]
            return
        if normalized.startswith("SELECT id FROM embedding_repair_queue WHERE"):
            rows = self._filter_repairs_for_claim(normalized, arguments)
            limit = _as_int(arguments[-1])
            self._result = [(row["id"],) for row in rows[:limit]]
            return
        if normalized.startswith("UPDATE embedding_repair_queue SET status = %s, attempt_count = attempt_count + 1"):
            item_ids = {str(value) for value in arguments[5:]}
            for row in self._state.embedding_repair_queue:
                if str(row["id"]) in item_ids:
                    row["status"] = str(arguments[0])
                    row["attempt_count"] = _as_int(row["attempt_count"]) + 1
                    row["updated_at"] = _as_float(arguments[1])
                    row["claimed_at"] = _as_float(arguments[2])
                    row["lease_owner"] = str(arguments[3])
                    row["lease_expires_at"] = _as_float(arguments[4])
                    row["last_error"] = None
            return
        if normalized.startswith("SELECT id, memory_id, workspace_id, model_name, memory_updated_at, status") and "FROM embedding_repair_queue WHERE id IN (" in normalized:
            item_ids = {str(value) for value in arguments}
            rows = [row for row in self._state.embedding_repair_queue if str(row["id"]) in item_ids]
            rows.sort(key=lambda row: _as_float(row["created_at"]))
            self._result = [self._repair_row(row) for row in rows]
            return
        if normalized.startswith("UPDATE embedding_repair_queue SET status = %s, updated_at = %s, completed_at = %s"):
            item_id = str(arguments[3])
            updated = 0
            for row in self._state.embedding_repair_queue:
                if str(row["id"]) == item_id and str(row["status"]) == str(arguments[4]):
                    row["status"] = str(arguments[0])
                    row["updated_at"] = _as_float(arguments[1])
                    row["completed_at"] = _as_float(arguments[2])
                    row["lease_owner"] = None
                    row["lease_expires_at"] = None
                    row["last_error"] = None
                    updated = 1
            self.rowcount = updated
            return
        if normalized.startswith("SELECT id, memory_id, workspace_id, model_name, memory_updated_at, status") and "FROM embedding_repair_queue WHERE id = %s" in normalized:
            item_id = str(arguments[0])
            row = next((row for row in self._state.embedding_repair_queue if str(row["id"]) == item_id), None)
            self._result = [] if row is None else [self._repair_row(row)]
            return
        if normalized.startswith("SELECT COALESCE(SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END), 0),"):
            queued = [row for row in self._state.embedding_repair_queue if str(row["status"]) == "pending"]
            running = [row for row in self._state.embedding_repair_queue if str(row["status"]) == "running"]
            oldest = None if not queued else min(_as_float(row["created_at"]) for row in queued)
            self._result = [(len(queued), len(running), oldest)]
            return
        if normalized.startswith("DELETE FROM embedding_repair_queue WHERE id IN ("):
            item_ids = {str(value) for value in arguments}
            before = len(self._state.embedding_repair_queue)
            self._state.embedding_repair_queue = [row for row in self._state.embedding_repair_queue if str(row["id"]) not in item_ids]
            self.rowcount = before - len(self._state.embedding_repair_queue)
            return
        if normalized.startswith("INSERT INTO embeddings"):
            record = next(
                (
                    row for row in self._state.embeddings
                    if str(row["source_kind"]) == str(arguments[0])
                    and str(row["source_id"]) == str(arguments[1])
                    and str(row["model_name"]) == str(arguments[3])
                ),
                None,
            )
            payload = {
                "source_kind": str(arguments[0]),
                "source_id": str(arguments[1]),
                "workspace_id": arguments[2],
                "model_name": str(arguments[3]),
                "embedding_json": arguments[4],
                "updated_at": _as_float(arguments[5]),
            }
            if record is None:
                self._state.embeddings.append(payload)
            else:
                record.update(payload)
            return
        if normalized.startswith("SELECT source_kind, source_id, workspace_id, model_name, embedding_json, updated_at FROM embeddings WHERE source_kind = %s AND source_id = %s AND model_name = %s"):
            row = next(
                (
                    row for row in self._state.embeddings
                    if str(row["source_kind"]) == str(arguments[0])
                    and str(row["source_id"]) == str(arguments[1])
                    and str(row["model_name"]) == str(arguments[2])
                ),
                None,
            )
            self._result = [] if row is None else [
                (row["source_kind"], row["source_id"], row["workspace_id"], row["model_name"], row["embedding_json"], row["updated_at"])
            ]
            return
        if normalized.startswith("SELECT source_id, embedding_json FROM embeddings WHERE source_kind = %s AND model_name = %s"):
            rows = [
                row for row in self._state.embeddings
                if str(row["source_kind"]) == str(arguments[0]) and str(row["model_name"]) == str(arguments[1])
            ]
            next_argument_index = 2
            if "workspace_id = %s" in normalized:
                rows = [row for row in rows if row["workspace_id"] == arguments[next_argument_index]]
                next_argument_index += 1
            if "source_id = ANY(%s::text[])" in normalized:
                candidate_ids_raw = arguments[next_argument_index]
                if not isinstance(candidate_ids_raw, list | tuple):
                    raise TypeError("expected candidate id sequence")
                candidate_ids = {str(candidate_id) for candidate_id in candidate_ids_raw}
                rows = [row for row in rows if str(row["source_id"]) in candidate_ids]
            self._result = [(row["source_id"], row["embedding_json"]) for row in rows]
            return
        if normalized.startswith("DELETE FROM embeddings WHERE source_kind = %s AND source_id = %s"):
            before = len(self._state.embeddings)
            rows = [
                row for row in self._state.embeddings
                if not (
                    str(row["source_kind"]) == str(arguments[0])
                    and str(row["source_id"]) == str(arguments[1])
                    and (len(arguments) < 3 or str(row["model_name"]) == str(arguments[2]))
                )
            ]
            self._state.embeddings = rows
            self.rowcount = before - len(rows)
            return
        raise AssertionError(f"Unhandled query: {normalized}")

    def fetchone(self) -> tuple[object, ...] | None:
        return None if not self._result else self._result[0]

    def fetchall(self) -> list[tuple[object, ...]]:
        return list(self._result)

    def _filter_journal_rows(
        self,
        normalized: str,
        arguments: tuple[object, ...],
        *,
        include_limit: bool = True,
    ) -> list[dict[str, object]]:
        rows = list(self._state.system1_journal)
        clauses = normalized.split(" WHERE ", 1)[1].split(" ORDER BY ", 1)[0]
        params = list(arguments[:-1] if include_limit else arguments)
        for clause in clauses.split(" AND "):
            if clause == "status = 'pending'":
                rows = [row for row in rows if str(row["status"]) == "pending"]
            elif clause == "workspace_id = %s":
                value = params.pop(0)
                rows = [row for row in rows if row["workspace_id"] == value]
            elif clause == "workspace_id IS NULL":
                rows = [row for row in rows if row["workspace_id"] is None]
        return rows

    def _filter_journal_group_rows(self, normalized: str, arguments: tuple[object, ...]) -> list[dict[str, object]]:
        rows = list(self._state.system1_journal)
        if " WHERE " not in normalized:
            return rows
        clauses = normalized.split(" WHERE ", 1)[1].split(" GROUP BY ", 1)[0]
        params = list(arguments)
        for clause in clauses.split(" AND "):
            if clause == "workspace_id = %s":
                value = params.pop(0)
                rows = [row for row in rows if row["workspace_id"] == value]
            elif clause == "workspace_id IS NULL":
                rows = [row for row in rows if row["workspace_id"] is None]
        return rows

    def _filter_work_items_for_claim(self, normalized: str, arguments: tuple[object, ...]) -> list[dict[str, object]]:
        rows = list(self._state.work_items)
        params = list(arguments[:-1])
        family_token = params.pop(0)
        if "family_key IN (" in normalized:
            family_count = normalized.split("family_key IN (", 1)[1].split(")", 1)[0].count("%s")
            family_keys = [family_token, *[params.pop(0) for _ in range(family_count - 1)]]
            rows = [row for row in rows if str(row["family_key"]) in {str(value) for value in family_keys}]
        else:
            rows = [row for row in rows if str(row["family_key"]) == str(family_token)]
        execution_lane = params.pop(0)
        now_pending = _as_float(params.pop(0))
        now_running = _as_float(params.pop(0))
        rows = [
            row for row in rows
            if str(row["execution_lane"]) == str(execution_lane)
            and (
                (str(row["status"]) in {"pending", "deferred"} and _as_float(row["available_at"]) <= now_pending)
                or (str(row["status"]) == "running" and row["lease_expires_at"] is not None and _as_float(row["lease_expires_at"]) <= now_running)
            )
        ]
        if "workspace_id IS NULL" in normalized:
            rows = [row for row in rows if row["workspace_id"] is None]
        elif "workspace_id = %s" in normalized:
            workspace_id = params.pop(0)
            rows = [row for row in rows if row["workspace_id"] == workspace_id]
        rows.sort(key=lambda row: (_as_int(row["priority"]), _as_float(row["created_at"])))
        return rows

    def _filter_repairs_for_claim(self, normalized: str, arguments: tuple[object, ...]) -> list[dict[str, object]]:
        rows = list(self._state.embedding_repair_queue)
        now_pending = _as_float(arguments[0])
        now_running = _as_float(arguments[1])
        rows = [
            row for row in rows
            if (
                (str(row["status"]) == "pending" and _as_float(row["available_at"]) <= now_pending)
                or (str(row["status"]) == "running" and row["lease_expires_at"] is not None and _as_float(row["lease_expires_at"]) <= now_running)
            )
        ]
        if "workspace_id IS NULL" in normalized:
            rows = [row for row in rows if row["workspace_id"] is None]
        elif "workspace_id = %s" in normalized:
            workspace_id = arguments[2]
            rows = [row for row in rows if row["workspace_id"] == workspace_id]
        rows.sort(key=lambda row: _as_float(row["created_at"]))
        return rows

    def _work_item_row(self, row: dict[str, object]) -> tuple[object, ...]:
        return (
            row["id"],
            row["family_key"],
            row["execution_lane"],
            row["workspace_id"],
            row["payload_json"],
            row["status"],
            row["priority"],
            row["attempt_count"],
            row["available_at"],
            row["created_at"],
            row["updated_at"],
            row["claimed_at"],
            row["completed_at"],
            row["lease_owner"],
            row["lease_expires_at"],
            row["idempotency_key"],
            row["last_error"],
        )

    def _repair_row(self, row: dict[str, object]) -> tuple[object, ...]:
        return (
            row["id"],
            row["memory_id"],
            row["workspace_id"],
            row["model_name"],
            row["memory_updated_at"],
            row["status"],
            row["attempt_count"],
            row["available_at"],
            row["created_at"],
            row["updated_at"],
            row["claimed_at"],
            row["completed_at"],
            row["lease_owner"],
            row["lease_expires_at"],
            row["last_error"],
        )


class FakePrimitiveConnection:
    def __init__(self, state: FakePrimitiveState) -> None:
        self._state = state

    def cursor(self) -> FakePrimitiveCursor:
        return FakePrimitiveCursor(self._state)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class FakePrimitiveLease:
    def __init__(self, connection: FakePrimitiveConnection) -> None:
        self._connection = connection

    def __enter__(self) -> FakePrimitiveConnection:
        return self._connection

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self._connection.rollback()
        return False

    def close(self) -> None:
        return None


class FakePrimitiveSessionManager:
    def __init__(self) -> None:
        self.state = FakePrimitiveState()

    def __enter__(self) -> FakePrimitiveSessionManager:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def open_connection(self) -> FakePrimitiveLease:
        return FakePrimitiveLease(FakePrimitiveConnection(self.state))

    def close(self) -> None:
        return None


def test_postgres_system1_journal_records_and_moves_claims_to_recoverable() -> None:
    session_manager = FakePrimitiveSessionManager()
    journal = PostgresSystem1Journal(session_manager)

    entry_one = journal.record("  first thought  ", workspace_id="workspace-a")
    entry_two = journal.record("second thought", workspace_id="workspace-a")

    claimed = journal.claim_pending(task_id="task-1", limit=10, workspace_id="workspace-a", claimed_at=50.0)
    moved = journal.move_claims_to_recoverable("task-1", recoverable_until=90.0)

    assert entry_one.content == "first thought"
    assert [entry.id for entry in claimed] == [entry_one.id, entry_two.id]
    assert moved == [entry_one.id, entry_two.id]
    assert journal.count_by_status("workspace-a") == {"recoverable": 2}
    assert journal.get_pending(workspace_id="workspace-a") == []
    assert journal.get_oldest_pending_timestamp(workspace_id="workspace-a") is None
    assert journal.get_recent(limit=2)[0].id == entry_two.id
    assert journal.get_latest_thought_timestamp(_ALL_WORKSPACES) == pytest.approx(entry_two.timestamp)


def test_postgres_work_item_repository_claims_compatible_batches_and_completes_items() -> None:
    session_manager = FakePrimitiveSessionManager()
    repository = PostgresWorkItemRepository(session_manager)

    first, created = repository.enqueue_unique(
        family_key="memory_tagging",
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        payload={"memory_id": "m-1"},
        workspace_id="workspace-a",
        priority=10,
        available_at=5.0,
        idempotency_key="dedupe-1",
    )
    duplicate, duplicate_created = repository.enqueue_unique(
        family_key="memory_tagging",
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        payload={"memory_id": "m-1"},
        workspace_id="workspace-a",
        priority=99,
        available_at=6.0,
        idempotency_key="dedupe-1",
    )
    second, _ = repository.enqueue_unique(
        family_key="graph_link_review",
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        payload={"memory_id": "m-2"},
        workspace_id="workspace-a",
        priority=20,
        available_at=5.0,
        idempotency_key="dedupe-2",
    )

    claimed = repository.claim_compatible_batch(
        family_keys=["memory_tagging", "graph_link_review"],
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        lease_owner="worker-a",
        limit=10,
        workspace_id="workspace-a",
        now=10.0,
    )
    heartbeated = repository.heartbeat_item(first.id, lease_owner="worker-a", heartbeated_at=12.0)
    completed = repository.complete_item(first.id, completed_at=15.0)

    assert created is True
    assert duplicate_created is False
    assert duplicate.id == first.id
    assert [item.id for item in claimed] == [first.id, second.id]
    assert heartbeated.lease_expires_at == pytest.approx(12.0 + 1800.0)
    assert completed.status == "completed"
    assert completed.completed_at == pytest.approx(15.0)


def test_postgres_embedding_repair_queue_tracks_backlog_and_prunes_completed_items() -> None:
    session_manager = FakePrimitiveSessionManager()
    queue = PostgresEmbeddingRepairQueue(session_manager)

    first, created = queue.enqueue_unique(
        memory_id="memory-1",
        workspace_id="workspace-a",
        model_name="mini-embed",
        memory_updated_at="2026-03-27T00:00:00+00:00",
        available_at=5.0,
    )
    duplicate, duplicate_created = queue.enqueue_unique(
        memory_id="memory-1",
        workspace_id="workspace-a",
        model_name="mini-embed",
        memory_updated_at="2026-03-27T00:00:00+00:00",
        available_at=6.0,
    )

    claimed = queue.claim_batch(lease_owner="repair-worker", limit=10, workspace_id="workspace-a", now=10.0)
    completed = queue.complete_item(first.id, completed_at=20.0)
    snapshot = queue.backlog_snapshot(now=25.0)
    pruned = queue.prune_completed(older_than_seconds=1.0, limit=10, now=25.0)

    assert created is True
    assert duplicate_created is False
    assert duplicate.id == first.id
    assert [item.id for item in claimed] == [first.id]
    assert completed.status == "completed"
    assert snapshot.queued_count == 0
    assert snapshot.running_count == 0
    assert snapshot.oldest_queued_age_seconds is None
    assert pruned == 1


def test_postgres_vector_store_round_trips_and_ranks_embeddings() -> None:
    session_manager = FakePrimitiveSessionManager()
    store = PostgresVectorStore(session_manager)

    store.upsert(
        source_kind="memory",
        source_id="memory-a",
        workspace_id="workspace-a",
        model_name="mini-embed",
        embedding=[1.0, 0.0],
    )
    store.upsert(
        source_kind="memory",
        source_id="memory-b",
        workspace_id="workspace-a",
        model_name="mini-embed",
        embedding=[0.0, 1.0],
    )

    record = store.get(source_kind="memory", source_id="memory-a", model_name="mini-embed")
    ranked = store.search(
        source_kind="memory",
        model_name="mini-embed",
        query_embedding=[0.8, 0.2],
        workspace_id="workspace-a",
        limit=2,
    )
    deleted = store.delete(source_kind="memory", source_id="memory-b", model_name="mini-embed")

    assert record is not None
    assert record.embedding == [1.0, 0.0]
    assert [memory_id for memory_id, _score in ranked] == ["memory-a", "memory-b"]
    assert json.loads(str(session_manager.state.embeddings[0]["embedding_json"])) == [1.0, 0.0]
    assert deleted == 1


def test_postgres_vector_store_can_bound_search_to_candidate_ids() -> None:
    session_manager = FakePrimitiveSessionManager()
    store = PostgresVectorStore(session_manager)

    store.upsert(
        source_kind="memory",
        source_id="memory-a",
        workspace_id=None,
        model_name="mini-embed",
        embedding=[1.0, 0.0],
    )
    store.upsert(
        source_kind="memory",
        source_id="memory-b",
        workspace_id=None,
        model_name="mini-embed",
        embedding=[0.0, 1.0],
    )

    ranked = store.search(
        source_kind="memory",
        model_name="mini-embed",
        query_embedding=[0.8, 0.2],
        candidate_ids=["memory-b"],
        limit=5,
    )

    assert ranked == [("memory-b", pytest.approx(0.24253562503633294))]