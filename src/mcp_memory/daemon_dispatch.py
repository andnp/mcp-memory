from __future__ import annotations

import re
from urllib.parse import parse_qs, urlsplit

from mcp_memory.management.scope_policy import resolve_workspace_id_for_policy, scope_policy_for_endpoint


_DEFAULT_LIST_LIMIT = 20
_DEFAULT_LOG_LIMIT = 50
_MAX_LIST_LIMIT = 200
_MEMORY_DETAIL_PATH_RE = re.compile(r"^/api/memories/(?P<memory_id>[^/]+)$")
_TASK_CANCEL_PATH_RE = re.compile(r"^/api/admin/tasks/(?P<task_id>[^/]+)/cancel$")


def dispatch_management_request(routes, metadata, path: str, payload: dict[str, object]) -> dict:
    if path == "/api/health":
        response = routes.service.get_health().model_dump()
        response["pid"] = metadata.pid
        response["status"] = metadata.status
        response["daemon_scope"] = metadata.daemon_scope
        response["binary_path"] = metadata.binary_path
        response["version"] = metadata.version
        response["transport"] = metadata.transport
        return response
    if path == "/api/overview":
        return routes.service.get_overview().model_dump()
    if path == "/api/metrics/nerd":
        effective_workspace_id = _resolve_endpoint_workspace_id(routes, path, payload)
        return routes.service.get_nerd_metrics(
            scope="global" if effective_workspace_id is None else None,
            workspace_id=effective_workspace_id,
            window_hours=optional_int(payload, "window_hours", default=24, minimum=1, maximum=24 * 30) or 24,
            bucket_minutes=optional_int(payload, "bucket_minutes", default=60, minimum=1, maximum=24 * 60) or 60,
            now=optional_float(payload, "now"),
        ).model_dump()
    if path == "/api/selector-stats":
        effective_workspace_id = _resolve_endpoint_workspace_id(routes, path, payload)
        return routes.service.get_selector_stats(
            scope="global" if effective_workspace_id is None else None,
            workspace_id=effective_workspace_id,
            window_hours=optional_int(payload, "window_hours", default=24, minimum=1, maximum=24 * 30) or 24,
            limit=optional_int(payload, "limit", default=200, minimum=1, maximum=500) or 200,
            now=optional_float(payload, "now"),
        ).model_dump()
    if path == "/api/tasks":
        effective_workspace_id = _resolve_endpoint_workspace_id(routes, path, payload)
        return routes.service.list_tasks(
            status=optional_str(payload, "status"),
            workspace_id=effective_workspace_id,
            limit=optional_int(payload, "limit", default=_DEFAULT_LIST_LIMIT, minimum=1, maximum=_MAX_LIST_LIMIT),
        ).model_dump()
    if path == "/api/record-thought":
        return serialize_payload(
            routes.service.record_thought(
                required_str(payload, "content"),
            )
        )
    if path == "/api/memories":
        effective_workspace_id = _resolve_endpoint_workspace_id(routes, path, payload)
        return routes.service.list_memories(
            workspace_id=effective_workspace_id,
            memory_type=optional_str(payload, "memory_type"),
            status=optional_str(payload, "status"),
            limit=optional_int(payload, "limit", default=_DEFAULT_LIST_LIMIT, minimum=1, maximum=_MAX_LIST_LIMIT),
        ).model_dump()
    if path == "/api/memories/search":
        effective_workspace_id = _resolve_endpoint_workspace_id(routes, path, payload)
        return routes.service.search_memories(
            query=required_str(payload, "query"),
            workspace_id=effective_workspace_id,
            memory_type=optional_str(payload, "memory_type"),
            status=optional_str(payload, "status"),
            include_superseded=bool_value(payload, "include_superseded", default=False),
            limit=optional_int(payload, "limit", default=_DEFAULT_LIST_LIMIT, minimum=1, maximum=_MAX_LIST_LIMIT) or _DEFAULT_LIST_LIMIT,
            debug=bool_value(payload, "debug", default=False),
        ).model_dump()
    if path == "/api/logs":
        effective_workspace_id = _resolve_endpoint_workspace_id(routes, path, payload)
        return routes.service.list_logs(
            workspace_id=effective_workspace_id,
            level=optional_str(payload, "level"),
            logger_name=optional_str(payload, "logger_name"),
            source=optional_str(payload, "source"),
            query=optional_str(payload, "q"),
            after=optional_float(payload, "after"),
            before=optional_float(payload, "before"),
            limit=optional_int(payload, "limit", default=_DEFAULT_LOG_LIMIT, minimum=1, maximum=_MAX_LIST_LIMIT),
        ).model_dump()
    if path == "/api/logs/summary":
        effective_workspace_id = _resolve_endpoint_workspace_id(routes, path, payload)
        return routes.service.summarize_logs(
            workspace_id=effective_workspace_id,
            level=optional_str(payload, "level"),
            logger_name=optional_str(payload, "logger_name"),
            source=optional_str(payload, "source"),
            query=optional_str(payload, "q"),
            after=optional_float(payload, "after"),
            before=optional_float(payload, "before"),
        ).model_dump()
    if path == "/api/ai-conversations":
        effective_workspace_id = _resolve_endpoint_workspace_id(routes, path, payload)
        return routes.service.list_ai_conversations(
            workspace_id=effective_workspace_id,
            request_id=optional_str(payload, "request_id"),
            task_name=optional_str(payload, "task_name"),
            status=optional_str(payload, "status"),
            limit=optional_int(payload, "limit", default=_DEFAULT_LOG_LIMIT, minimum=1, maximum=_MAX_LIST_LIMIT),
        ).model_dump()
    if path == "/api/admin/agents/run":
        return routes.service.enqueue_background_task(
            required_str(payload, "task_name"),
            force=bool_value(payload, "force", default=False),
        )
    if path == "/api/admin/agents/run-all":
        return {
            "results": routes.service.enqueue_all_background_tasks(
                force=bool_value(payload, "force", default=False)
            )
        }
    if path == "/api/admin/logs/prune":
        effective_workspace_id = _resolve_endpoint_workspace_id(routes, path, payload)
        return routes.service.prune_logs(
            workspace_id=effective_workspace_id,
            max_runtime_logs=optional_int(payload, "max_runtime_logs"),
            max_log_age_days=optional_int(payload, "max_log_age_days"),
        ).model_dump()
    if path == "/api/admin/search/repair":
        return serialize_payload(routes.service.repair_search_index())
    if path == "/api/admin/links":
        return serialize_payload(
            routes.service.create_memory_link(
                source_id=required_str(payload, "source_id"),
                target_id=required_str(payload, "target_id"),
                link_type=required_str(payload, "link_type"),
                context=required_str(payload, "context"),
            )
        )
    if path == "/api/admin/links/delete":
        return serialize_payload(
            routes.service.delete_memory_link(
                source_id=required_str(payload, "source_id"),
                target_id=required_str(payload, "target_id"),
                link_type=required_str(payload, "link_type"),
            )
        )

    memory_match = _MEMORY_DETAIL_PATH_RE.fullmatch(path)
    if memory_match is not None:
        return routes.service.get_memory_detail(memory_match.group("memory_id")).model_dump()

    cancel_match = _TASK_CANCEL_PATH_RE.fullmatch(path)
    if cancel_match is not None:
        return serialize_payload(
            routes.service.cancel_task(
                cancel_match.group("task_id"),
                cancelled_by=optional_str(payload, "cancelled_by") or "cli",
                reason=optional_str(payload, "reason") or "cancelled_by_user",
            )
        )

    return {"status": "error", "error": "unknown_transport_path", "path": path}


def serialize_tool(tool) -> dict[str, object]:
    return {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.inputSchema,
    }


def serialize_tool_response(response) -> dict[str, object]:
    return {
        "contents": [
            {
                "type": content.type,
                "text": content.text,
            }
            for content in response
        ]
    }


def normalize_request(path: str, payload: dict | None) -> tuple[str, dict[str, object]]:
    split = urlsplit(path)
    request_payload = {} if payload is None else dict(payload)
    if split.query:
        for key, values in parse_qs(split.query, keep_blank_values=True).items():
            if key in request_payload:
                continue
            if len(values) == 1:
                request_payload[key] = values[0]
            else:
                request_payload[key] = values
    return split.path or path, request_payload


def _resolve_endpoint_workspace_id(routes, path: str, payload: dict[str, object]) -> str | None:
    return resolve_workspace_id_for_policy(
        scope_policy_for_endpoint(path),
        scope=optional_str(payload, "scope"),
        workspace_id=optional_str(payload, "workspace_id"),
        current_workspace_id=routes.service.workspace_id,
    )


def optional_str(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def required_str(payload: dict[str, object], key: str) -> str:
    value = optional_str(payload, key)
    if value is None:
        raise ValueError(f"{key}_required")
    return value


def optional_int(
    payload: dict[str, object],
    key: str,
    *,
    default: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    value = payload.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{key}_must_be_integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{key}_must_be_integer") from exc
    else:
        raise ValueError(f"{key}_must_be_integer")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{key}_out_of_range")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{key}_out_of_range")
    return parsed


def optional_float(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{key}_must_be_number")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{key}_must_be_number") from exc
    raise ValueError(f"{key}_must_be_number")


def bool_value(payload: dict[str, object], key: str, *, default: bool) -> bool:
    value = payload.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{key}_must_be_boolean")


def serialize_payload(payload) -> dict[str, object]:
    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        serialized = model_dump()
        if isinstance(serialized, dict):
            return serialized
        raise ValueError("service_response_must_be_object")
    if isinstance(payload, dict):
        return payload
    raise ValueError("service_response_must_be_object")


def error_payload(exc: ValueError) -> dict[str, object]:
    return {
        "status": "error",
        "error": str(exc),
    }
