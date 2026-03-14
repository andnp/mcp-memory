from __future__ import annotations

import json
import socket
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from mcp_memory.daemon_models import DaemonMetadata


def read_daemon_metadata(metadata_path: Path) -> DaemonMetadata | None:
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return DaemonMetadata(**payload)
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
    try:
        request = Request(f"{metadata.base_url}/internal/health", method="GET")
        with urlopen(request, timeout=1) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload.get("workspace_id") == metadata.workspace_id and payload.get("status") == "ready"
    except (URLError, OSError, TimeoutError, json.JSONDecodeError):
        return False