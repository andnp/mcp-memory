from __future__ import annotations

import json
import socket
import subprocess
import sys
from dataclasses import asdict
from dataclasses import fields
from pathlib import Path
from typing import Any, cast

from mcp_memory.daemon_transport import request_daemon_json
from mcp_memory.daemon_models import DaemonMetadata


def read_daemon_metadata(metadata_path: Path) -> DaemonMetadata | None:
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return DaemonMetadata(**cast(Any, _normalize_metadata_payload(payload)))
    except TypeError:
        return None


def spawn_daemon_process(workspace_root: Path, host: str, port: int) -> None:
    command = [
        sys.executable,
        "-m",
        "mcp_memory.cli",
        "daemon",
        "--workspace-root",
        str(workspace_root),
        "--host",
        host,
        "--port",
        str(port),
    ]
    subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def write_metadata(metadata_path: Path, metadata: DaemonMetadata) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(asdict(metadata), sort_keys=True), encoding="utf-8")


def remove_metadata(metadata_path: Path) -> None:
    try:
        metadata_path.unlink()
    except FileNotFoundError:
        return


def is_daemon_healthy(metadata: DaemonMetadata) -> bool:
    expected_socket_path = metadata.socket_path.strip() if isinstance(metadata.socket_path, str) else None
    if metadata.transport in {"zmq", "hybrid"} and not expected_socket_path:
        return False
    try:
        payload = request_daemon_json(metadata, "/internal/health", None, timeout_seconds=1)
        if payload.get("daemon_scope", "global") != metadata.daemon_scope:
            return False
        if payload.get("status") != "ready":
            return False
        if metadata.transport in {"zmq", "hybrid"}:
            live_socket_path = payload.get("socket_path")
            if not isinstance(live_socket_path, str) or not live_socket_path.strip():
                return False
            if live_socket_path != expected_socket_path:
                return False
        return True
    except (OSError, TimeoutError, json.JSONDecodeError, ValueError):
        return False


def _normalize_metadata_payload(payload: dict[str, object]) -> dict[str, object]:
    allowed_fields = {field.name for field in fields(DaemonMetadata)}
    normalized = {key: value for key, value in payload.items() if key in allowed_fields}
    normalized.setdefault("daemon_scope", "global")
    normalized.setdefault("transport", "http")
    normalized.setdefault("binary_path", None)
    normalized.setdefault("version", None)
    normalized.setdefault("socket_path", None)
    return normalized
