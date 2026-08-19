from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

REMINDER_INTERVAL_SECONDS = 300.0
ACTIVE_CLIENT_STALE_AFTER_SECONDS = 3600.0
REMINDER_MESSAGE = (
    "Before moving on, record any durable finding, decision, anomaly, or reusable next step with the "
    "record_thought tool. Skip routine play-by-play and status-only updates."
)


@dataclass(slots=True)
class HookConversationRecord:
    conversation_id: str
    workspace_id: str | None
    started_at: float
    updated_at: float
    last_ping_at: float | None
    last_reminder_at: float | None
    last_tool_name: str | None
    last_payload: dict[str, object]
    ended_at: float | None


class HookReminderService:
    def __init__(self, db_manager: object, workspace_id: str | None) -> None:
        self._db = db_manager
        self._workspace_id = workspace_id

    def record_session_start(self, conversation_id: str, payload: dict[str, object] | None = None) -> dict[str, str]:
        normalized_id = _normalize_conversation_id(conversation_id)
        now = _payload_timestamp(payload)
        self._execute_write(
            """
            INSERT INTO hook_conversations (
                conversation_id, workspace_id, started_at, updated_at, last_ping_at, last_payload_json, ended_at
            ) VALUES ({}, {}, {}, {}, {}, {}, NULL)
            ON CONFLICT(conversation_id) DO UPDATE SET
                workspace_id = excluded.workspace_id,
                updated_at = excluded.updated_at,
                last_ping_at = excluded.last_ping_at,
                last_payload_json = excluded.last_payload_json,
                ended_at = NULL
            """,
            (normalized_id, self._workspace_id, now, now, now, json.dumps(payload or {}, sort_keys=True)),
        )
        return {"status": "ok", "conversation_id": normalized_id}

    def record_post_tool_use(self, payload: dict[str, object]) -> dict[str, str]:
        conversation_id = _conversation_id_from_payload(payload)
        now = _payload_timestamp(payload)
        tool_name = _tool_name_from_payload(payload)
        existing = self.get_conversation(conversation_id)

        started_at = now if existing is None else existing.started_at
        last_reminder_at = None if existing is None else existing.last_reminder_at

        anchor = started_at if last_reminder_at is None else last_reminder_at
        should_remind = now - anchor >= REMINDER_INTERVAL_SECONDS

        def _write(executor) -> None:
            self._execute_on_executor(
                executor,
                """
                INSERT INTO hook_conversations (
                    conversation_id,
                    workspace_id,
                    started_at,
                    updated_at,
                    last_ping_at,
                    last_reminder_at,
                    last_tool_name,
                    last_payload_json,
                    ended_at
                ) VALUES ({}, {}, {}, {}, {}, {}, {}, {}, NULL)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    workspace_id = excluded.workspace_id,
                    updated_at = excluded.updated_at,
                    last_ping_at = excluded.last_ping_at,
                    last_tool_name = excluded.last_tool_name,
                    last_payload_json = excluded.last_payload_json,
                    ended_at = NULL
                """,
                (
                    conversation_id,
                    self._workspace_id,
                    started_at,
                    now,
                    now,
                    last_reminder_at,
                    tool_name,
                    json.dumps(payload, sort_keys=True),
                ),
            )
            if should_remind:
                self._execute_on_executor(
                    executor,
                    "UPDATE hook_conversations SET last_reminder_at = {}, updated_at = {} WHERE conversation_id = {}",
                    (now, now, conversation_id),
                )

        self._with_write_transaction(_write)

        if not should_remind:
            return {}
        return {
            "systemMessage": REMINDER_MESSAGE,
            "additionalContext": REMINDER_MESSAGE,
        }

    def record_session_end(self, conversation_id: str, payload: dict[str, object] | None = None) -> dict[str, str]:
        normalized_id = _normalize_conversation_id(conversation_id)
        now = _payload_timestamp(payload)
        self._execute_write(
            "UPDATE hook_conversations SET updated_at = {}, ended_at = {} WHERE conversation_id = {}",
            (now, now, normalized_id),
        )
        return {"status": "ok", "conversation_id": normalized_id}

    def get_conversation(self, conversation_id: str) -> HookConversationRecord | None:
        normalized_id = _normalize_conversation_id(conversation_id)
        row = self._fetchone(
            """
            SELECT
                conversation_id,
                workspace_id,
                started_at,
                updated_at,
                last_ping_at,
                last_reminder_at,
                last_tool_name,
                last_payload_json,
                ended_at
            FROM hook_conversations
            WHERE conversation_id = {}
            """,
            (normalized_id,),
        )
        if row is None:
            return None
        (
            conversation_id_value,
            workspace_id,
            started_at,
            updated_at,
            last_ping_at,
            last_reminder_at,
            last_tool_name,
            last_payload_json,
            ended_at,
        ) = row
        return HookConversationRecord(
            conversation_id=str(conversation_id_value),
            workspace_id=None if workspace_id is None else str(workspace_id),
            started_at=_coerce_float(started_at),
            updated_at=_coerce_float(updated_at),
            last_ping_at=None if last_ping_at is None else _coerce_float(last_ping_at),
            last_reminder_at=None if last_reminder_at is None else _coerce_float(last_reminder_at),
            last_tool_name=None if last_tool_name is None else str(last_tool_name),
            last_payload=_decode_payload(last_payload_json),
            ended_at=None if ended_at is None else _coerce_float(ended_at),
        )

    def get_active_client_count(self, *, now: float | None = None) -> int:
        current_time = time.time() if now is None else now
        active_after = current_time - ACTIVE_CLIENT_STALE_AFTER_SECONDS
        row = self._fetchone(
            "SELECT COUNT(*) FROM hook_conversations WHERE ended_at IS NULL AND updated_at >= {}",
            (active_after,),
        )
        return 0 if row is None else _coerce_int(row[0])

    def _uses_sqlite(self) -> bool:
        return hasattr(self._db, "get_connection")

    def _format_query(self, query_template: str) -> str:
        placeholder = "?" if self._uses_sqlite() else "%s"
        return query_template.format(*([placeholder] * query_template.count("{}")))

    def _with_write_transaction(self, operation) -> None:
        db = cast(Any, self._db)
        if self._uses_sqlite():
            conn = db.get_connection()
            operation(conn)
            conn.commit()
            return
        with db.open_connection() as connection:
            with connection.cursor() as cursor:
                operation(cursor)
            connection.commit()

    def _execute_on_executor(self, executor, query_template: str, params: tuple[object, ...]) -> None:
        executor.execute(self._format_query(query_template), params)

    def _execute_write(self, query_template: str, params: tuple[object, ...]) -> None:
        self._with_write_transaction(lambda executor: self._execute_on_executor(executor, query_template, params))

    def _fetchone(self, query_template: str, params: tuple[object, ...]) -> tuple[object, ...] | None:
        db = cast(Any, self._db)
        if self._uses_sqlite():
            row = db.get_connection().execute(self._format_query(query_template), params).fetchone()
            if row is None:
                return None
            return tuple(row)
        with db.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(self._format_query(query_template), params)
                row = cursor.fetchone()
        return None if row is None else tuple(row)


def _conversation_id_from_payload(payload: dict[str, object]) -> str:
    raw = payload.get("conversation_id") or payload.get("sessionId") or payload.get("session_id")
    return _normalize_conversation_id(raw)


def _tool_name_from_payload(payload: dict[str, object]) -> str:
    raw = payload.get("tool_name") or payload.get("tool") or payload.get("action") or "unknown"
    return str(raw).strip() or "unknown"


def _normalize_conversation_id(value: object) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError("conversation_id_required")
    return normalized


def _payload_timestamp(payload: dict[str, object] | None) -> float:
    if payload is None:
        return time.time()
    raw = payload.get("timestamp")
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str) and raw.strip():
        stripped = raw.strip()
        try:
            return float(stripped)
        except ValueError:
            try:
                return datetime.fromisoformat(stripped.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return time.time()
    return time.time()


def _decode_payload(raw: object) -> dict[str, object]:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        decoded = json.loads(raw)
    except ValueError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _coerce_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected float-compatible value, got {type(value)!r}")


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
