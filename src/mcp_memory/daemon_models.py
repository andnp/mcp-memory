from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import logging
from pathlib import Path
from typing import Any, cast

from mcp_memory.management.capabilities import ManagementCapabilities
from mcp_memory.integrations.federation_source import MemoryFederationSource
from mcp_memory.management.service import ManagementService


logger = logging.getLogger(__name__)


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
    management: ManagementCapabilities | None
    hook_service: Any
    metadata_path: Path
    federation_source: MemoryFederationSource | None = None


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
        try:
            return int(self.hook_service.get_active_client_count())
        except Exception as exc:
            logger.warning("Failed to read active daemon client count", exc_info=exc)
            return 0

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
