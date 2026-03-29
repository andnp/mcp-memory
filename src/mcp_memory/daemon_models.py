from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, cast

from mcp_memory.management.service import ManagementService


@dataclass
class DaemonMetadata:
    host: str
    port: int
    pid: int
    started_at: float
    status: str
    daemon_scope: str = "global"
    binary_path: str | None = None
    version: str | None = None
    transport: str = "zmq"
    socket_path: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def transport_endpoint(self) -> str:
        if self.socket_path:
            return f"ipc://{self.socket_path}"
        return self.base_url


@dataclass
class DaemonRoutes:
    ctx: Any
    service: ManagementService
    hook_service: Any
    metadata_path: Path


@dataclass(frozen=True)
class DaemonControllerView:
    hook_service: Any | None = None
    transport_server: Any | None = None

    @property
    def has_runtime(self) -> bool:
        return True

    @property
    def client_count(self) -> int:
        if self.hook_service is None:
            return 0
        return int(self.hook_service.get_active_client_count())

    @property
    def transport_diagnostics(self) -> dict[str, object] | None:
        if self.transport_server is None:
            return None
        get_snapshot = getattr(self.transport_server, "get_diagnostics_snapshot", None)
        if not callable(get_snapshot):
            return None
        snapshot = get_snapshot()
        if snapshot is None:
            return None
        if isinstance(snapshot, dict):
            return snapshot
        if not is_dataclass(snapshot):
            return None
        return cast(dict[str, object], asdict(cast(Any, snapshot)))
