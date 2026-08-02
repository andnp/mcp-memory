from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import shlex
from typing import Any, TextIO
from urllib.error import HTTPError, URLError

from mcp_memory.config import resolve_workspace_root
from mcp_memory.daemon import ensure_daemon_started
from mcp_memory.daemon_transport import request_daemon_json
from mcp_memory.utils.atomic_io import atomic_write_json


SUPPORTED_INSTALL_TOOLS = ("copilot", "claude", "gemini")
SUPPORTED_INSTALL_COMPONENTS = ("hooks", "mcp")
HOOK_EVENT_ENDPOINTS = {
    "SessionStart": "/api/hooks/session-start",
    "PostToolUse": "/api/hooks/post-tool-use",
    "Stop": "/api/hooks/session-end",
    "SessionEnd": "/api/hooks/session-end",
}
DEFAULT_HOOK_TIMEOUT_SECONDS = 10
DEFAULT_GEMINI_SERVER_NAME = "mcp-memory-internal"


@dataclass(slots=True)
class InstallAction:
    tool: str
    component: str
    path: Path
    status: str


@dataclass(slots=True)
class InstallResult:
    workspace_root: Path
    actions: list[InstallAction] = field(default_factory=list)


def install_integrations(
    *,
    tools: tuple[str, ...] | list[str],
    components: tuple[str, ...] | list[str],
    scope: str,
    workspace_root: str | None,
) -> InstallResult:
    normalized_tools = _normalize_requested_values(tools, SUPPORTED_INSTALL_TOOLS)
    normalized_components = _normalize_requested_values(components, SUPPORTED_INSTALL_COMPONENTS)
    if scope not in {"workspace", "user"}:
        raise ValueError("invalid_install_scope")
    if scope == "user" and "gemini" in normalized_tools and "mcp" in normalized_components:
        raise ValueError("gemini_mcp_requires_workspace_scope")

    target_workspace = resolve_workspace_root(workspace_root=workspace_root)
    repo_root = _resolve_repo_root()
    result = InstallResult(workspace_root=target_workspace)

    if "hooks" in normalized_components:
        if "copilot" in normalized_tools:
            result.actions.append(_install_copilot_hooks(repo_root, target_workspace, scope))
        if "claude" in normalized_tools:
            result.actions.append(_install_claude_hooks(repo_root, target_workspace, scope))

    if "mcp" in normalized_components and "gemini" in normalized_tools:
        result.actions.append(_install_gemini_mcp(repo_root, target_workspace))

    return result


def load_hook_payload(stream: TextIO) -> dict[str, Any]:
    raw = stream.read().strip()
    if not raw:
        return {}
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise ValueError("hook_payload_must_be_object")
    return decoded


def forward_hook_event(payload: dict[str, Any], workspace_root: str | None = None) -> dict[str, Any]:
    event_name = _hook_event_name(payload)
    endpoint = HOOK_EVENT_ENDPOINTS.get(event_name)
    if endpoint is None:
        return {}

    metadata = ensure_daemon_started()
    request_payload = dict(payload)
    if workspace_root is not None:
        request_payload.setdefault("workspace_root", workspace_root)
    decoded = request_daemon_json(metadata, endpoint, request_payload, timeout_seconds=5)
    if not isinstance(decoded, dict):
        raise ValueError("hook_response_must_be_object")
    return decoded


def safe_forward_hook_event(payload: dict[str, Any], workspace_root: str | None = None) -> tuple[dict[str, Any], str | None]:
    try:
        response = forward_hook_event(payload, workspace_root=workspace_root)
    except (ValueError, HTTPError, URLError, OSError, TimeoutError) as exc:
        return {}, str(exc)
    return response, None


def _install_copilot_hooks(repo_root: Path, workspace_root: Path, scope: str) -> InstallAction:
    hook_payload = _build_hook_configuration(repo_root, workspace_root)
    if scope == "workspace":
        path = workspace_root / ".github" / "hooks" / "mcp-memory.json"
        atomic_write_json(path, hook_payload)
        return InstallAction(tool="copilot", component="hooks", path=path, status="installed")

    path = Path.home() / ".claude" / "settings.json"
    _merge_hook_settings(path, hook_payload)
    return InstallAction(tool="copilot", component="hooks", path=path, status="updated")


def _install_claude_hooks(repo_root: Path, workspace_root: Path, scope: str) -> InstallAction:
    hook_payload = _build_hook_configuration(repo_root, workspace_root)
    path = (
        workspace_root / ".claude" / "settings.local.json"
        if scope == "workspace"
        else Path.home() / ".claude" / "settings.json"
    )
    _merge_hook_settings(path, hook_payload)
    return InstallAction(tool="claude", component="hooks", path=path, status="updated")


def _install_gemini_mcp(repo_root: Path, workspace_root: Path) -> InstallAction:
    path = workspace_root / ".gemini" / "settings.json"
    settings = _load_json_object(path)
    mcp_settings = settings.get("mcp")
    if not isinstance(mcp_settings, dict):
        mcp_settings = {}
    allowed = mcp_settings.get("allowed")
    normalized_allowed = [] if not isinstance(allowed, list) else [str(value) for value in allowed if str(value).strip()]
    if DEFAULT_GEMINI_SERVER_NAME not in normalized_allowed:
        normalized_allowed.append(DEFAULT_GEMINI_SERVER_NAME)
    mcp_settings["allowed"] = normalized_allowed
    settings["mcp"] = mcp_settings

    mcp_servers = settings.get("mcpServers")
    if not isinstance(mcp_servers, dict):
        mcp_servers = {}
    mcp_servers[DEFAULT_GEMINI_SERVER_NAME] = {
        "command": "uv",
        "args": _build_uv_args(repo_root, "internal-run", workspace_root),
        "timeout": 30000,
        "trust": True,
    }
    settings["mcpServers"] = mcp_servers
    atomic_write_json(path, settings)
    return InstallAction(tool="gemini", component="mcp", path=path, status="updated")


def _build_hook_configuration(repo_root: Path, workspace_root: Path) -> dict[str, Any]:
    command = _build_uv_command(repo_root, "hook-runner", workspace_root)
    hook_entry = {
        "type": "command",
        "command": command,
        "timeout": DEFAULT_HOOK_TIMEOUT_SECONDS,
    }
    return {
        "hooks": {
            "SessionStart": [hook_entry],
            "PostToolUse": [hook_entry],
            "Stop": [hook_entry],
        }
    }


def _merge_hook_settings(path: Path, hook_payload: dict[str, Any]) -> None:
    settings = _load_json_object(path)
    existing_hooks = settings.get("hooks")
    normalized_hooks = {} if not isinstance(existing_hooks, dict) else dict(existing_hooks)

    for event_name, entries in hook_payload["hooks"].items():
        current_entries = normalized_hooks.get(event_name)
        normalized_entries = [] if not isinstance(current_entries, list) else [entry for entry in current_entries if isinstance(entry, dict)]
        for entry in entries:
            if not any(_hook_entries_match(candidate, entry) for candidate in normalized_entries):
                normalized_entries.append(entry)
        normalized_hooks[event_name] = normalized_entries

    settings["hooks"] = normalized_hooks
    atomic_write_json(path, settings)


def _hook_entries_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        str(left.get("type", "")).strip() == str(right.get("type", "")).strip()
        and str(left.get("command", "")).strip() == str(right.get("command", "")).strip()
    )


def _normalize_requested_values(values: tuple[str, ...] | list[str], supported: tuple[str, ...]) -> list[str]:
    raw_values = [value.strip() for value in values if value.strip()]
    if not raw_values or "all" in raw_values:
        return list(supported)
    invalid = sorted(set(raw_values) - set(supported))
    if invalid:
        raise ValueError(f"unsupported_install_values:{','.join(invalid)}")
    return list(dict.fromkeys(raw_values))


def _build_uv_command(repo_root: Path, entrypoint: str, workspace_root: Path) -> str:
    args = ["uv", *_build_uv_args(repo_root, entrypoint, workspace_root)]
    return " ".join(shlex.quote(value) for value in args)


def _build_uv_args(repo_root: Path, entrypoint: str, workspace_root: Path) -> list[str]:
    return [
        "--directory",
        str(repo_root),
        "run",
        "mcp-memory",
        entrypoint,
        "--workspace-root",
        str(workspace_root),
    ]


def _hook_event_name(payload: dict[str, Any]) -> str:
    raw_value = payload.get("hookEventName") or payload.get("hook_event_name")
    return str(raw_value or "").strip()


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"json_settings_must_be_object:{path}")
    return decoded


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]