from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

    @property
    def has_runtime(self) -> bool:
        return True

    @property
    def client_count(self) -> int:
        if self.hook_service is None:
            return 0
        return int(self.hook_service.get_active_client_count())
