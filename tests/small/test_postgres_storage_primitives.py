from __future__ import annotations

from collections.abc import Sequence
import json

import pytest

from mcp_memory.core.journal import _ALL_WORKSPACES
from mcp_memory.embedding_integrity_event_store import (
    EMBEDDING_INTEGRITY_EVENT_KIND_BLOCKED_FALLBACK_WRITE,
    EMBEDDING_INTEGRITY_EVENT_KIND_SCAN_SUMMARY,
)
from mcp_memory.storage.postgres_embedding_integrity_event_store import PostgresEmbeddingIntegrityEventRepository
from mcp_memory.storage.postgres_embedding_repair_store import PostgresEmbeddingRepairQueue
from mcp_memory.storage.postgres_journal import PostgresSystem1Journal
from mcp_memory.storage.postgres_vector_store import PostgresVectorStore
from mcp_memory.storage.postgres_work_item_store import PostgresWorkItemRepository
from mcp_memory.work_item_store import EXECUTION_LANE_DETERMINISTIC
from searchkernel.utils.similarity import cosine_similarity_lists
from tests.small.work_item_repository_contract import (
    assert_claim_batch_orders_ready_items,
    assert_enqueue_unique_deduplicates_idempotency_keys,
    assert_heartbeat_extends_leases_and_allows_expired_reclaim,
    assert_release_defer_and_complete_items,
)


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


def _json_payload_type(value: object) -> str | None:
    parsed = value
    if isinstance(value, str):
        parsed = json.loads(value)
    if isinstance(parsed, list | tuple):
        return "array"
    if isinstance(parsed, dict):
        return "object"
    if isinstance(parsed, bool):
        return "boolean"
    if parsed is None:
        return "null"
    if isinstance(parsed, (int, float)):
        return "number"
    if isinstance(parsed, str):
        return "string"
    return None


class FakePrimitiveState:
    def __init__(self) -> None:
        self.system1_journal: list[dict[str, object]] = []
        self.next_journal_id = 1
        self.work_items: list[dict[str, object]] = []
        self.embedding_repair_queue: list[dict[str, object]] = []
        self.embedding_integrity_events: list[dict[str, object]] = []
        self.next_embedding_integrity_event_id = 1
        self.embeddings: list[dict[str, object]] = []
        self.pgvector_extension_installed = False
        self.embedding_vector_column_present = False
        self.vector_capability_query_count = 0


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
        if normalized.startswith("SELECT id, content, workspace_id, timestamp, status FROM system1_journal WHERE content = %s"):
            content = str(arguments[0])
            workspace_id = arguments[1]
            timestamp = _as_float(arguments[2])
            row = next(
                (
                    candidate for candidate in self._state.system1_journal
                    if str(candidate["content"]) == content
                    and candidate["workspace_id"] == workspace_id
                    and _as_float(candidate["timestamp"]) == timestamp
                ),
                None,
            )
            self._result = [] if row is None else [
                (row["id"], row["content"], row["workspace_id"], row["timestamp"], row["status"])
            ]
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
        if normalized.startswith("UPDATE work_items SET status = %s, updated_at = %s, available_at = %s"):
            item_id = str(arguments[4])
            updated = 0
            for row in self._state.work_items:
                if str(row["id"]) == item_id and str(row["status"]) == str(arguments[5]):
                    row["status"] = str(arguments[0])
                    row["updated_at"] = _as_float(arguments[1])
                    row["available_at"] = _as_float(arguments[2])
                    row["lease_owner"] = None
                    row["lease_expires_at"] = None
                    row["last_error"] = None if arguments[3] is None else str(arguments[3])
                    updated = 1
            self.rowcount = updated
            return
        if normalized.startswith("UPDATE work_items SET status = %s, updated_at = %s, lease_owner = NULL, lease_expires_at = NULL"):
            item_id = str(arguments[2])
            updated = 0
            for row in self._state.work_items:
                if str(row["id"]) == item_id and str(row["status"]) == str(arguments[3]):
                    row["status"] = str(arguments[0])
                    row["updated_at"] = _as_float(arguments[1])
                    row["lease_owner"] = None
                    row["lease_expires_at"] = None
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
                "embedding_vector": None if "embedding_vector" not in normalized else str(arguments[5]),
                "updated_at": _as_float(arguments[5] if "embedding_vector" not in normalized else arguments[6]),
            }
            if record is None:
                self._state.embeddings.append(payload)
            else:
                record.update(payload)
            return
        if normalized.startswith("INSERT INTO embedding_integrity_events"):
            details_json = arguments[8]
            if isinstance(details_json, str):
                details_json = json.loads(details_json)
            self._state.embedding_integrity_events.append(
                {
                    "id": self._state.next_embedding_integrity_event_id,
                    "workspace_id": arguments[0],
                    "event_kind": str(arguments[1]),
                    "model_name": arguments[2],
                    "source_kind": arguments[3],
                    "source_id": arguments[4],
                    "scanned_row_count": arguments[5],
                    "invalid_row_count": arguments[6],
                    "mixed_dimension_group_count": arguments[7],
                    "details_json": details_json,
                    "created_at": _as_float(arguments[9]),
                }
            )
            self._state.next_embedding_integrity_event_id += 1
            self.rowcount = 1
            return
        if normalized.startswith("SELECT COUNT(*) FROM embedding_integrity_events"):
            rows = self._filter_embedding_integrity_events(normalized, arguments)
            self._result = [(len(rows),)]
            return
        if normalized.startswith("SELECT event_kind, COUNT(*) FROM embedding_integrity_events"):
            rows = self._filter_embedding_integrity_events(normalized, arguments)
            grouped: dict[str, int] = {}
            for row in rows:
                event_kind = str(row["event_kind"])
                grouped[event_kind] = grouped.get(event_kind, 0) + 1
            self._result = [(event_kind, count) for event_kind, count in sorted(grouped.items())]
            return
        if normalized.startswith("SELECT id, workspace_id, event_kind, model_name, source_kind, source_id, scanned_row_count, invalid_row_count, mixed_dimension_group_count, details_json, created_at FROM embedding_integrity_events WHERE"):
            rows = self._filter_embedding_integrity_events(normalized, arguments)
            rows.sort(key=lambda row: (_as_float(row["created_at"]), _as_int(row["id"])), reverse=True)
            limited = rows[:1]
            self._result = [
                (
                    row["id"],
                    row["workspace_id"],
                    row["event_kind"],
                    row["model_name"],
                    row["source_kind"],
                    row["source_id"],
                    row["scanned_row_count"],
                    row["invalid_row_count"],
                    row["mixed_dimension_group_count"],
                    row["details_json"],
                    row["created_at"],
                )
                for row in limited
            ]
            return
        if normalized.startswith("SELECT source_kind, source_id, workspace_id, model_name, embedding_json, updated_at, memory_updated_at FROM embeddings WHERE source_kind = %s AND source_id = %s AND model_name = %s"):
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
                (
                    row["source_kind"],
                    row["source_id"],
                    row["workspace_id"],
                    row["model_name"],
                    row["embedding_json"],
                    row["updated_at"],
                    row.get("memory_updated_at"),
                )
            ]
            return
        if normalized.startswith("SELECT DISTINCT jsonb_array_length(embedding_json) FROM embeddings WHERE model_name = %s AND jsonb_typeof(embedding_json) = 'array'"):
            dimensions = sorted(
                {
                    len(json.loads(str(row["embedding_json"])))
                    for row in self._state.embeddings
                    if str(row["model_name"]) == str(arguments[0])
                }
            )
            self._result = [(dimension,) for dimension in dimensions]
            return
        if normalized.startswith("SELECT source_kind, source_id, workspace_id, model_name, CASE WHEN jsonb_typeof(embedding_json) = 'array' THEN jsonb_array_length(embedding_json) ELSE NULL END AS embedding_dimension, jsonb_typeof(embedding_json) AS payload_type, updated_at FROM embeddings ORDER BY model_name ASC, source_kind ASC, source_id ASC"):
            rows = sorted(
                self._state.embeddings,
                key=lambda row: (str(row["model_name"]), str(row["source_kind"]), str(row["source_id"])),
            )
            self._result = []
            for row in rows:
                payload_type = _json_payload_type(row["embedding_json"])
                embedding_dimension = None
                if payload_type == "array":
                    payload = row["embedding_json"]
                    if isinstance(payload, str):
                        embedding_dimension = len(json.loads(payload))
                    elif isinstance(payload, list | tuple):
                        embedding_dimension = len(payload)
                self._result.append(
                    (
                        row["source_kind"],
                        row["source_id"],
                        row["workspace_id"],
                        row["model_name"],
                        embedding_dimension,
                        payload_type,
                        row["updated_at"],
                    )
                )
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
                next_argument_index += 1
            rows.sort(key=lambda row: str(row["source_id"]))
            limit = _as_int(arguments[next_argument_index])
            self._result = [(row["source_id"], row["embedding_json"]) for row in rows[:limit]]
            return
        if normalized.startswith("SELECT source_id, 1 - (embedding_vector <=> CAST(%s AS vector)) AS score FROM embeddings WHERE source_kind = %s AND model_name = %s AND embedding_vector IS NOT NULL AND jsonb_typeof(embedding_json) = 'array' AND jsonb_array_length(embedding_json) = %s"):
            query_embedding = [float(value) for value in json.loads(str(arguments[0]))]
            rows = [
                row for row in self._state.embeddings
                if str(row["source_kind"]) == str(arguments[1])
                and str(row["model_name"]) == str(arguments[2])
                and row["embedding_vector"] is not None
                and len(json.loads(str(row["embedding_json"]))) == _as_int(arguments[3])
            ]
            next_argument_index = 4
            if "workspace_id = %s" in normalized:
                rows = [row for row in rows if row["workspace_id"] == arguments[next_argument_index]]
                next_argument_index += 1
            if "source_id = ANY(%s::text[])" in normalized:
                candidate_ids_raw = arguments[next_argument_index]
                if not isinstance(candidate_ids_raw, list | tuple):
                    raise TypeError("expected candidate id sequence")
                candidate_ids = {str(candidate_id) for candidate_id in candidate_ids_raw}
                rows = [row for row in rows if str(row["source_id"]) in candidate_ids]
                next_argument_index += 1
            assert str(arguments[next_argument_index]) == str(arguments[0])
            limit = _as_int(arguments[next_argument_index + 1])
            scored_rows = [
                (
                    str(row["source_id"]),
                    cosine_similarity_lists(query_embedding, [float(value) for value in json.loads(str(row["embedding_json"]))]),
                )
                for row in rows
            ]
            scored_rows.sort(key=lambda item: item[1], reverse=True)
            self._result = [
                (source_id, score)
                for source_id, score in scored_rows[:limit]
            ]
            return
        if normalized.startswith("SELECT EXISTS ( SELECT 1 FROM pg_extension WHERE extname = 'vector' )"):
            self._state.vector_capability_query_count += 1
            self._result = [(self._state.pgvector_extension_installed,)]
            return
        if normalized.startswith("SELECT EXISTS ( SELECT 1 FROM information_schema.columns"):
            self._state.vector_capability_query_count += 1
            self._result = [(self._state.embedding_vector_column_present,)]
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

    def executemany(self, query: str, rows: Sequence[tuple[object, ...]]) -> None:
        for row in rows:
            self.execute(query, tuple(row))

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

    def _filter_embedding_integrity_events(
        self,
        normalized: str,
        arguments: tuple[object, ...],
    ) -> list[dict[str, object]]:
        rows = list(self._state.embedding_integrity_events)
        if " WHERE " not in normalized:
            return rows
        clauses = normalized.split(" WHERE ", 1)[1].split(" ORDER BY ", 1)[0].split(" GROUP BY ", 1)[0]
        params = list(arguments)
        for clause in clauses.split(" AND "):
            if clause == "event_kind = %s":
                value = str(params.pop(0))
                rows = [row for row in rows if str(row["event_kind"]) == value]
            elif clause == "workspace_id = %s":
                value = params.pop(0)
                rows = [row for row in rows if row["workspace_id"] == value]
        return rows


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


def test_postgres_work_item_repository_deduplicates_idempotency_keys() -> None:
    assert_enqueue_unique_deduplicates_idempotency_keys(
        lambda: PostgresWorkItemRepository(FakePrimitiveSessionManager())
    )


def test_postgres_work_item_repository_claim_batch_orders_ready_items() -> None:
    assert_claim_batch_orders_ready_items(
        lambda: PostgresWorkItemRepository(FakePrimitiveSessionManager())
    )


def test_postgres_work_item_repository_heartbeat_and_reclaim_share_backend_contract() -> None:
    assert_heartbeat_extends_leases_and_allows_expired_reclaim(
        lambda: PostgresWorkItemRepository(FakePrimitiveSessionManager())
    )


def test_postgres_work_item_repository_release_defer_and_complete_share_backend_contract() -> None:
    assert_release_defer_and_complete_items(
        lambda: PostgresWorkItemRepository(FakePrimitiveSessionManager())
    )


def test_postgres_work_item_repository_claims_compatible_batches_across_families() -> None:
    session_manager = FakePrimitiveSessionManager()
    repository = PostgresWorkItemRepository(session_manager)

    first, created = repository.enqueue_unique(
        family_key="memory_tagging",
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        payload={"memory_id": "m-1"},
        workspace_id="workspace-a",
        priority=10,
        available_at=5.0,
        idempotency_key="compatible-1",
    )
    second, second_created = repository.enqueue_unique(
        family_key="graph_link_review",
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        payload={"memory_id": "m-2"},
        workspace_id="workspace-a",
        priority=20,
        available_at=5.0,
        idempotency_key="compatible-2",
    )

    claimed = repository.claim_compatible_batch(
        family_keys=["memory_tagging", "graph_link_review"],
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        lease_owner="worker-a",
        limit=10,
        workspace_id="workspace-a",
        now=10.0,
    )

    assert created is True
    assert second_created is True
    assert [item.id for item in claimed] == [first.id, second.id]
    assert [item.status for item in claimed] == ["running", "running"]
    assert [item.lease_owner for item in claimed] == ["worker-a", "worker-a"]


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


def test_postgres_vector_store_rejects_empty_embeddings() -> None:
    session_manager = FakePrimitiveSessionManager()
    store = PostgresVectorStore(session_manager)

    with pytest.raises(ValueError, match="embedding dimension must be positive"):
        store.upsert(
            source_kind="memory",
            source_id="memory-empty",
            workspace_id=None,
            model_name="mini-embed",
            embedding=[],
        )

    assert session_manager.state.embeddings == []


def test_postgres_vector_store_matches_kernel_diagnostics_and_tie_ordering() -> None:
    session_manager = FakePrimitiveSessionManager()
    store = PostgresVectorStore(session_manager)

    for source_id in ("memory-b", "memory-a"):
        store.upsert(
            source_kind="memory",
            source_id=source_id,
            workspace_id=None,
            model_name="mini-embed",
            embedding=[1.0, 0.0],
        )

    diagnostics: dict[str, object] = {}
    ranked = store.search(
        source_kind="memory",
        model_name="mini-embed",
        query_embedding=[1.0, 0.0],
        diagnostics=diagnostics,
        limit=2,
    )

    assert [source_id for source_id, _score in ranked] == ["memory-a", "memory-b"]
    assert store.last_search_diagnostics == diagnostics
    diagnostics["row_count"] = -1
    assert store.last_search_diagnostics["row_count"] == 2


def test_postgres_vector_store_returns_no_results_for_nonpositive_limit() -> None:
    session_manager = FakePrimitiveSessionManager()
    store = PostgresVectorStore(session_manager)

    store.upsert(
        source_kind="memory",
        source_id="memory-a",
        workspace_id=None,
        model_name="mini-embed",
        embedding=[1.0, 0.0],
    )

    assert store.search(
        source_kind="memory",
        model_name="mini-embed",
        query_embedding=[1.0, 0.0],
        limit=0,
    ) == []


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


def test_postgres_vector_store_reports_search_diagnostics() -> None:
    session_manager = FakePrimitiveSessionManager()
    store = PostgresVectorStore(session_manager)

    store.upsert(
        source_kind="memory",
        source_id="memory-a",
        workspace_id=None,
        model_name="mini-embed",
        embedding=[1.0, 0.0],
    )
    diagnostics: dict[str, object] = {}

    ranked = store.search(
        source_kind="memory",
        model_name="mini-embed",
        query_embedding=[1.0, 0.0],
        diagnostics=diagnostics,
        limit=5,
    )

    assert ranked == [("memory-a", pytest.approx(1.0))]
    assert diagnostics["backend"] == "postgres"
    assert diagnostics["row_count"] == 1
    assert diagnostics["raw_type_counts"] == {"str": 1}
    assert diagnostics["candidate_filter_count"] == 0
    assert diagnostics["search_mode"] == "client_python_fallback"
    assert diagnostics["pgvector_extension_installed"] is False
    assert diagnostics["embedding_vector_column_present"] is False
    assert diagnostics["server_side_vector_search_available"] is False
    assert set(diagnostics) >= {"fetch_ms", "decode_ms", "score_ms", "sort_ms"}


def test_postgres_vector_store_skips_malformed_fallback_rows_with_diagnostics() -> None:
    session_manager = FakePrimitiveSessionManager()
    store = PostgresVectorStore(session_manager)
    store.upsert(
        source_kind="memory",
        source_id="memory-good",
        workspace_id=None,
        model_name="mini-embed",
        embedding=[1.0, 0.0],
    )
    session_manager.state.embeddings.extend(
        [
            {
                "source_kind": "memory",
                "source_id": "memory-object",
                "workspace_id": None,
                "model_name": "mini-embed",
                "embedding_json": json.dumps({"not": "an-array"}),
                "embedding_vector": None,
                "updated_at": 0.0,
            },
            {
                "source_kind": "memory",
                "source_id": "memory-invalid-json",
                "workspace_id": None,
                "model_name": "mini-embed",
                "embedding_json": "not-json",
                "embedding_vector": None,
                "updated_at": 0.0,
            },
        ]
    )

    diagnostics: dict[str, object] = {}
    ranked = store.search(
        source_kind="memory",
        model_name="mini-embed",
        query_embedding=[1.0, 0.0],
        diagnostics=diagnostics,
        limit=5,
    )

    assert ranked == [("memory-good", pytest.approx(1.0))]
    assert diagnostics["skipped_invalid_embedding_count"] == 2
    assert diagnostics["compatible_row_count"] == 1


def test_postgres_vector_store_bounds_client_fallback_rows() -> None:
    session_manager = FakePrimitiveSessionManager()
    store = PostgresVectorStore(session_manager, fallback_row_cap=2)
    for source_id in ("memory-c", "memory-a", "memory-b"):
        store.upsert(
            source_kind="memory",
            source_id=source_id,
            workspace_id=None,
            model_name="mini-embed",
            embedding=[1.0, 0.0],
        )

    diagnostics: dict[str, object] = {}
    ranked = store.search(
        source_kind="memory",
        model_name="mini-embed",
        query_embedding=[1.0, 0.0],
        diagnostics=diagnostics,
        limit=5,
    )

    assert [source_id for source_id, _score in ranked] == ["memory-a", "memory-b"]
    assert diagnostics["row_count"] == 2
    assert diagnostics["fallback_row_cap"] == 2
    assert diagnostics["fallback_row_cap_applied"] is True


def test_postgres_vector_store_caches_server_side_vector_capability_probe() -> None:
    session_manager = FakePrimitiveSessionManager()
    session_manager.state.pgvector_extension_installed = True
    session_manager.state.embedding_vector_column_present = True
    store = PostgresVectorStore(session_manager)

    store.upsert(
        source_kind="memory",
        source_id="memory-a",
        workspace_id=None,
        model_name="mini-embed",
        embedding=[1.0, 0.0],
    )

    first_diagnostics: dict[str, object] = {}
    second_diagnostics: dict[str, object] = {}

    first_ranked = store.search(
        source_kind="memory",
        model_name="mini-embed",
        query_embedding=[1.0, 0.0],
        diagnostics=first_diagnostics,
        limit=5,
    )
    second_ranked = store.search(
        source_kind="memory",
        model_name="mini-embed",
        query_embedding=[1.0, 0.0],
        diagnostics=second_diagnostics,
        limit=5,
    )

    assert first_ranked == [("memory-a", pytest.approx(1.0))]
    assert second_ranked == [("memory-a", pytest.approx(1.0))]
    assert first_diagnostics["search_mode"] == "server_side_pgvector"
    assert first_diagnostics["server_side_vector_search_available"] is True
    assert second_diagnostics["server_side_vector_search_available"] is True
    assert session_manager.state.vector_capability_query_count == 2


def test_postgres_vector_store_dual_writes_vector_column_when_capability_is_available() -> None:
    session_manager = FakePrimitiveSessionManager()
    session_manager.state.pgvector_extension_installed = True
    session_manager.state.embedding_vector_column_present = True
    store = PostgresVectorStore(session_manager)

    store.upsert(
        source_kind="memory",
        source_id="memory-a",
        workspace_id="workspace-a",
        model_name="mini-embed",
        embedding=[1.0, 0.25],
    )

    assert json.loads(str(session_manager.state.embeddings[0]["embedding_json"])) == [1.0, 0.25]
    assert session_manager.state.embeddings[0]["embedding_vector"] == "[1.0,0.25]"


def test_postgres_vector_store_uses_server_side_search_when_vector_capability_is_available() -> None:
    session_manager = FakePrimitiveSessionManager()
    session_manager.state.pgvector_extension_installed = True
    session_manager.state.embedding_vector_column_present = True
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

    diagnostics: dict[str, object] = {}
    ranked = store.search(
        source_kind="memory",
        model_name="mini-embed",
        query_embedding=[0.8, 0.2],
        diagnostics=diagnostics,
        limit=2,
    )

    assert [memory_id for memory_id, _score in ranked] == ["memory-a", "memory-b"]
    assert diagnostics["search_mode"] == "server_side_pgvector"
    assert diagnostics["server_side_vector_search_available"] is True
    assert diagnostics["raw_type_counts"] == {}
    assert diagnostics["decode_ms"] == 0.0
    assert diagnostics["score_ms"] == 0.0
    assert diagnostics["sort_ms"] == 0.0


def test_postgres_vector_store_server_side_search_skips_mismatched_dimensions() -> None:
    session_manager = FakePrimitiveSessionManager()
    session_manager.state.pgvector_extension_installed = True
    session_manager.state.embedding_vector_column_present = True
    store = PostgresVectorStore(session_manager)

    store.upsert(
        source_kind="memory",
        source_id="memory-a",
        workspace_id=None,
        model_name="mini-embed",
        embedding=[1.0, 0.0],
    )
    session_manager.state.embeddings.append(
        {
            "source_kind": "memory",
            "source_id": "memory-bad",
            "workspace_id": None,
            "model_name": "mini-embed",
            "embedding_json": json.dumps([1.0, 0.0, 0.0]),
            "embedding_vector": "[1.0,0.0,0.0]",
            "updated_at": 0.0,
        }
    )

    diagnostics: dict[str, object] = {}
    ranked = store.search(
        source_kind="memory",
        model_name="mini-embed",
        query_embedding=[1.0, 0.0],
        diagnostics=diagnostics,
        limit=5,
    )

    assert ranked == [("memory-a", pytest.approx(1.0))]
    assert diagnostics["search_mode"] == "server_side_pgvector"
    assert diagnostics["query_dimension"] == 2
    assert diagnostics["dimension_filter_applied"] is True


def test_postgres_vector_store_keeps_json_only_upsert_when_vector_capability_is_unavailable() -> None:
    session_manager = FakePrimitiveSessionManager()
    store = PostgresVectorStore(session_manager)

    store.upsert(
        source_kind="memory",
        source_id="memory-a",
        workspace_id="workspace-a",
        model_name="mini-embed",
        embedding=[1.0, 0.25],
    )

    assert json.loads(str(session_manager.state.embeddings[0]["embedding_json"])) == [1.0, 0.25]
    assert session_manager.state.embeddings[0]["embedding_vector"] is None


def test_postgres_vector_store_rejects_mismatched_dimension_for_existing_model_identity() -> None:
    session_manager = FakePrimitiveSessionManager()
    store = PostgresVectorStore(session_manager)

    store.upsert(
        source_kind="memory",
        source_id="memory-a",
        workspace_id=None,
        model_name="mini-embed",
        embedding=[1.0, 0.0],
    )

    with pytest.raises(ValueError, match="does not match new dimension 3"):
        store.upsert(
            source_kind="memory",
            source_id="memory-b",
            workspace_id=None,
            model_name="mini-embed",
            embedding=[1.0, 0.0, 0.0],
        )

    assert [str(row["source_id"]) for row in session_manager.state.embeddings] == ["memory-a"]


def test_postgres_vector_store_rejects_upsert_when_existing_model_identity_is_already_mixed() -> None:
    session_manager = FakePrimitiveSessionManager()
    store = PostgresVectorStore(session_manager)
    session_manager.state.embeddings.extend(
        [
            {
                "source_kind": "memory",
                "source_id": "memory-a",
                "workspace_id": None,
                "model_name": "mini-embed",
                "embedding_json": json.dumps([1.0, 0.0]),
                "embedding_vector": None,
                "updated_at": 0.0,
            },
            {
                "source_kind": "memory",
                "source_id": "memory-b",
                "workspace_id": None,
                "model_name": "mini-embed",
                "embedding_json": json.dumps([1.0, 0.0, 0.0]),
                "embedding_vector": None,
                "updated_at": 0.0,
            },
        ]
    )

    with pytest.raises(ValueError, match="already have mixed dimensions"):
        store.upsert(
            source_kind="memory",
            source_id="memory-c",
            workspace_id=None,
            model_name="mini-embed",
            embedding=[1.0, 0.0],
        )


def test_postgres_vector_store_blocks_fallback_hash_embedding_writes_and_tracks_policy_state() -> None:
    session_manager = FakePrimitiveSessionManager()
    store = PostgresVectorStore(session_manager)

    with pytest.raises(ValueError, match="Fallback/hash embeddings cannot be persisted in Postgres/shared mode"):
        store.upsert(
            source_kind="memory",
            source_id="memory-a",
            workspace_id=None,
            model_name="hash:sentence-transformers/all-MiniLM-L6-v2",
            embedding=[1.0, 0.0],
        )

    assert session_manager.state.embeddings == []
    assert store.get_write_policy_state() == store.get_write_policy_state().__class__(
        fallback_persistence_policy="blocked",
        blocked_fallback_write_count=1,
        last_blocked_fallback_model_name="hash:sentence-transformers/all-MiniLM-L6-v2",
    )


def test_postgres_vector_store_scan_integrity_reports_mixed_dimensions_and_invalid_rows() -> None:
    session_manager = FakePrimitiveSessionManager()
    store = PostgresVectorStore(session_manager)
    session_manager.state.embeddings.extend(
        [
            {
                "source_kind": "memory",
                "source_id": "memory-a",
                "workspace_id": None,
                "model_name": "mini-embed",
                "embedding_json": json.dumps([1.0, 0.0]),
                "embedding_vector": None,
                "updated_at": 10.0,
            },
            {
                "source_kind": "memory",
                "source_id": "memory-b",
                "workspace_id": None,
                "model_name": "mini-embed",
                "embedding_json": json.dumps([1.0, 0.0, 0.0]),
                "embedding_vector": None,
                "updated_at": 20.0,
            },
            {
                "source_kind": "thought",
                "source_id": "thought-bad",
                "workspace_id": None,
                "model_name": "mini-embed",
                "embedding_json": json.dumps({"not": "an-array"}),
                "embedding_vector": None,
                "updated_at": 30.0,
            },
            {
                "source_kind": "memory",
                "source_id": "memory-c",
                "workspace_id": None,
                "model_name": "other-embed",
                "embedding_json": json.dumps([1.0, 0.0, 0.0, 0.0]),
                "embedding_vector": None,
                "updated_at": 40.0,
            },
        ]
    )

    summary = store.scan_integrity(active_model_name="mini-embed", expected_dimension=2)

    assert summary["scanned_row_count"] == 4
    assert summary["invalid_row_count"] == 2
    assert summary["mixed_dimension_group_count"] == 1
    assert summary["mixed_dimension_groups"] == [
        {
            "model_name": "mini-embed",
            "dimensions": [2, 3],
            "row_count": 3,
            "source_kind_counts": {"memory": 2, "thought": 1},
        }
    ]
    assert summary["active_model_row_count"] == 3
    invalid_rows = {str(row["source_id"]): list(row["issues"]) for row in summary["invalid_rows"]}
    assert invalid_rows == {
        "memory-b": ["dimension_mismatch"],
        "thought-bad": ["invalid_payload_type"],
    }


def test_postgres_vector_store_persists_embedding_integrity_events() -> None:
    session_manager = FakePrimitiveSessionManager()
    event_repository = PostgresEmbeddingIntegrityEventRepository(session_manager, workspace_id="workspace-a")
    store = PostgresVectorStore(session_manager, event_repository=event_repository)

    with pytest.raises(ValueError, match="Fallback/hash embeddings cannot be persisted in Postgres/shared mode"):
        store.upsert(
            source_kind="memory",
            source_id="memory-blocked",
            workspace_id=None,
            model_name="hash:sentence-transformers/all-MiniLM-L6-v2",
            embedding=[1.0, 0.0],
        )

    session_manager.state.embeddings.extend(
        [
            {
                "source_kind": "memory",
                "source_id": "memory-a",
                "workspace_id": None,
                "model_name": "mini-embed",
                "embedding_json": json.dumps([1.0, 0.0]),
                "embedding_vector": None,
                "updated_at": 10.0,
            },
            {
                "source_kind": "memory",
                "source_id": "memory-b",
                "workspace_id": None,
                "model_name": "mini-embed",
                "embedding_json": json.dumps([1.0, 0.0, 0.0]),
                "embedding_vector": None,
                "updated_at": 20.0,
            },
        ]
    )

    store.scan_integrity(active_model_name="mini-embed", expected_dimension=2)
    summary = event_repository.summarize_events()

    assert summary.total == 2
    assert summary.by_kind == {
        EMBEDDING_INTEGRITY_EVENT_KIND_BLOCKED_FALLBACK_WRITE: 1,
        EMBEDDING_INTEGRITY_EVENT_KIND_SCAN_SUMMARY: 1,
    }
    assert summary.last_blocked_fallback_write is not None
    assert summary.last_blocked_fallback_write.model_name == "hash:sentence-transformers/all-MiniLM-L6-v2"
    assert summary.last_blocked_fallback_write.source_id == "memory-blocked"
    assert summary.last_scan is not None
    assert summary.last_scan.model_name == "mini-embed"
    assert summary.last_scan.scanned_row_count == 2
    assert summary.last_scan.invalid_row_count == 1
    assert summary.last_scan.mixed_dimension_group_count == 1
