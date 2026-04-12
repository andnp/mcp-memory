from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp_memory.config import Config


@dataclass(frozen=True)
class SharedModeCacheState:
    enabled: bool = False
    backend_supported: bool = False
    mode: str | None = None
    active: bool = False
    read_cache: Any = None
    max_outbox_entries: int | None = None

    @property
    def writeback_active(self) -> bool:
        return self.mode == "writeback" and self.active and self.max_outbox_entries is not None

    @property
    def writeback_cache(self) -> Any | None:
        if not self.writeback_active:
            return None
        return self.read_cache


def resolve_shared_mode_cache_state(
    config: Config | None,
    *,
    storage_backend: str | None,
    read_cache: Any = None,
) -> SharedModeCacheState:
    backend = storage_backend or "sqlite"
    backend_supported = backend == "postgres"
    storage_config = None if config is None else getattr(config, "storage", None)
    cache_config = None if storage_config is None else getattr(storage_config, "cache", None)
    if cache_config is None or not cache_config.enabled:
        return SharedModeCacheState(
            enabled=False,
            backend_supported=backend_supported,
        )

    mode = getattr(cache_config, "mode", None)
    active = backend_supported and read_cache is not None
    max_outbox_entries = getattr(cache_config, "max_outbox_entries", None)
    if mode != "writeback":
        max_outbox_entries = None

    return SharedModeCacheState(
        enabled=True,
        backend_supported=backend_supported,
        mode=mode,
        active=active,
        read_cache=read_cache if active else None,
        max_outbox_entries=max_outbox_entries if backend_supported else None,
    )