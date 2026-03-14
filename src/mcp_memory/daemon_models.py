from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_memory.management.service import ManagementService


@dataclass
class DaemonMetadata:
    workspace_id: str
    workspace_root: str
    host: str
    port: int
    pid: int
    started_at: float
    status: str

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class DaemonRoutes:
    ctx: Any
    service: ManagementService
    metadata_path: Path


class DaemonControllerView:
    @property
    def has_runtime(self) -> bool:
        return True

    @property
    def client_count(self) -> int:
        return 1