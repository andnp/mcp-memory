from __future__ import annotations

from types import SimpleNamespace

from mcp_memory.config import Config
from mcp_memory.storage.shared_mode_cache import resolve_shared_mode_cache_state


def test_resolve_shared_mode_cache_state_disables_cache_when_not_configured() -> None:
    state = resolve_shared_mode_cache_state(None, storage_backend="sqlite", read_cache=None)

    assert state.enabled is False
    assert state.backend_supported is False
    assert state.mode is None
    assert state.active is False
    assert state.writeback_active is False
    assert state.writeback_cache is None


def test_resolve_shared_mode_cache_state_reports_active_readonly_postgres_cache() -> None:
    config = Config()
    config.storage.cache.enabled = True
    config.storage.cache.mode = "readonly"
    read_cache = object()

    state = resolve_shared_mode_cache_state(config, storage_backend="postgres", read_cache=read_cache)

    assert state.enabled is True
    assert state.backend_supported is True
    assert state.mode == "readonly"
    assert state.active is True
    assert state.read_cache is read_cache
    assert state.max_outbox_entries is None
    assert state.writeback_active is False
    assert state.writeback_cache is None


def test_resolve_shared_mode_cache_state_reports_active_writeback_postgres_cache() -> None:
    config = Config()
    config.storage.cache.enabled = True
    config.storage.cache.mode = "writeback"
    config.storage.cache.max_outbox_entries = 17
    read_cache = SimpleNamespace(name="cache")

    state = resolve_shared_mode_cache_state(config, storage_backend="postgres", read_cache=read_cache)

    assert state.enabled is True
    assert state.backend_supported is True
    assert state.mode == "writeback"
    assert state.active is True
    assert state.max_outbox_entries == 17
    assert state.writeback_active is True
    assert state.writeback_cache is read_cache