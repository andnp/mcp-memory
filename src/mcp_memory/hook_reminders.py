from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from mcp_memory.utils.db import DatabaseManager


REMINDER_INTERVAL_SECONDS = 300.0
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
    last_payload: dict[str, Any]
    ended_at: float | None


class HookReminderService:
    def __init__(self, db_manager: DatabaseManager, workspace_id: str | None) -> None:
        self._db = db_manager
        self._workspace_id = workspace_id

    def record_session_start(self, conversation_id: str, payload: dict[str, Any] | None = None) -> dict[str, str]:
        normalized_id = _normalize_conversation_id(conversation_id)
        now = _payload_timestamp(payload)
        conn = self._db.get_connection()
        conn.execute(
            """
            INSERT INTO hook_conversations (
                conversation_id, workspace_id, started_at, updated_at, last_ping_at, last_payload_json, ended_at
            ) VALUES (?, ?, ?, ?, ?, ?, NULL)
            ON CONFLICT(conversation_id) DO UPDATE SET
                workspace_id = excluded.workspace_id,
                updated_at = excluded.updated_at,
                last_ping_at = excluded.last_ping_at,
                last_payload_json = excluded.last_payload_json,
                ended_at = NULL
            """,
            (normalized_id, self._workspace_id, now, now, now, json.dumps(payload or {}, sort_keys=True)),
        )
        conn.commit()
        return {"status": "ok", "conversation_id": normalized_id}

    def record_post_tool_use(self, payload: dict[str, Any]) -> dict[str, str]:
        conversation_id = _conversation_id_from_payload(payload)
        now = _payload_timestamp(payload)
        tool_name = _tool_name_from_payload(payload)
        existing = self.get_conversation(conversation_id)

        started_at = now if existing is None else existing.started_at
        last_reminder_at = None if existing is None else existing.last_reminder_at

        conn = self._db.get_connection()
        conn.execute(
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
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
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

        anchor = started_at if last_reminder_at is None else last_reminder_at
        should_remind = now - anchor >= REMINDER_INTERVAL_SECONDS
        if should_remind:
            conn.execute(
                "UPDATE hook_conversations SET last_reminder_at = ?, updated_at = ? WHERE conversation_id = ?",
                (now, now, conversation_id),
            )
        conn.commit()

        if not should_remind:
            return {}
        return {
            "systemMessage": REMINDER_MESSAGE,
            "additionalContext": REMINDER_MESSAGE,
        }

    def record_session_end(self, conversation_id: str, payload: dict[str, Any] | None = None) -> dict[str, str]:
        normalized_id = _normalize_conversation_id(conversation_id)
        now = _payload_timestamp(payload)
        conn = self._db.get_connection()
        conn.execute(
            "UPDATE hook_conversations SET updated_at = ?, ended_at = ? WHERE conversation_id = ?",
            (now, now, normalized_id),
        )
        conn.commit()
        return {"status": "ok", "conversation_id": normalized_id}

    def get_conversation(self, conversation_id: str) -> HookConversationRecord | None:
        normalized_id = _normalize_conversation_id(conversation_id)
        row = self._db.get_connection().execute(
            "SELECT * FROM hook_conversations WHERE conversation_id = ?",
            (normalized_id,),
        ).fetchone()
        if row is None:
            return None
        return HookConversationRecord(
            conversation_id=str(row["conversation_id"]),
            workspace_id=row["workspace_id"],
            started_at=float(row["started_at"]),
            updated_at=float(row["updated_at"]),
            last_ping_at=None if row["last_ping_at"] is None else float(row["last_ping_at"]),
            last_reminder_at=None if row["last_reminder_at"] is None else float(row["last_reminder_at"]),
            last_tool_name=row["last_tool_name"],
            last_payload=_decode_payload(row["last_payload_json"]),
            ended_at=None if row["ended_at"] is None else float(row["ended_at"]),
        )

    def get_active_client_count(self) -> int:
        conn = self._db.get_connection()
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM hook_conversations WHERE ended_at IS NULL"
        ).fetchone()
        return 0 if row is None else int(row["count"])


def _conversation_id_from_payload(payload: dict[str, Any]) -> str:
    raw = payload.get("conversation_id") or payload.get("sessionId") or payload.get("session_id")
    return _normalize_conversation_id(raw)


def _tool_name_from_payload(payload: dict[str, Any]) -> str:
    raw = payload.get("tool_name") or payload.get("tool") or payload.get("action") or "unknown"
    return str(raw).strip() or "unknown"


def _normalize_conversation_id(value: Any) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError("conversation_id_required")
    return normalized


def _payload_timestamp(payload: dict[str, Any] | None) -> float:
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


def _decode_payload(raw: object) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        decoded = json.loads(raw)
    except ValueError:
        return {}
    return decoded if isinstance(decoded, dict) else {}
