from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

import mcp_memory.core.journal_operations as journal_operations_module
import mcp_memory.daemon_app as daemon_app_module
from mcp_memory.config import Config, StorageCacheMode
from mcp_memory.core.journal import System1Journal
from mcp_memory.storage.shared_read_cache import SharedReadCache

pytestmark = pytest.mark.small


@pytest.mark.parametrize(
    ("enabled", "mode", "read_cache"),
    [
        (False, "writeback", object()),
        (True, "readonly", object()),
        (True, "writeback", None),
    ],
)
def test_flush_record_thought_writeback_once_noops_when_cache_mode_inactive(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    mode: StorageCacheMode,
    read_cache: object | None,
) -> None:
    config = Config()
    config.storage.cache.enabled = enabled
    config.storage.cache.mode = mode

    runtime = SimpleNamespace(
        config=config,
        storage_backend="postgres",
        journal=object(),
        read_cache=read_cache,
        task_queue=object(),
    )
    flush_calls: list[str] = []

    def _unexpected_flush(*args, **kwargs):
        del args, kwargs
        flush_calls.append("called")
        raise AssertionError("writeback helper should not run when cache mode is inactive")

    monkeypatch.setattr(daemon_app_module, "flush_record_thought_writeback_outbox", _unexpected_flush)

    assert daemon_app_module._flush_record_thought_writeback_once(runtime) == 0
    assert flush_calls == []


@pytest.mark.asyncio
async def test_record_thought_writeback_flush_loop_runs_immediately_then_polls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_times: list[float] = []
    first_call = threading.Event()
    second_call = threading.Event()

    def _fake_flush_once(_runtime) -> int:
        call_times.append(time.monotonic())
        if len(call_times) == 1:
            first_call.set()
        elif len(call_times) == 2:
            second_call.set()
        return 0

    monkeypatch.setattr(daemon_app_module, "_flush_record_thought_writeback_once", _fake_flush_once)

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="daemon-app-test")
    started_at = time.monotonic()
    task = asyncio.create_task(
        daemon_app_module._run_record_thought_writeback_flush_loop(
            SimpleNamespace(),
            executor=executor,
            poll_seconds=0.5,
        )
    )

    try:
        assert await asyncio.to_thread(first_call.wait, 0.2)
        assert second_call.is_set() is False
        assert call_times[0] - started_at < 0.2

        assert await asyncio.to_thread(second_call.wait, 1.0)
        assert call_times[1] - call_times[0] >= 0.35
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.to_thread(executor.shutdown, True)


@pytest.mark.asyncio
async def test_record_thought_writeback_flush_loop_shutdown_does_not_wait_for_timed_out_authoritative_thread(
    db_manager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config()
    config.storage.cache.enabled = True
    config.storage.cache.mode = "writeback"
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    assert cache.enqueue_record_thought_outbox_entry(
        content="queued thought",
        workspace_id="workspace-a",
        timestamp=10.0,
        max_entries=4,
    ) is not None

    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    class SlowJournal(System1Journal):
        def __init__(self) -> None:
            super().__init__(db_manager)

        def record_with_timestamp(
            self,
            content: str,
            workspace_id: str | None = None,
            *,
            timestamp: float | None = None,
        ):
            started.set()
            try:
                assert release.wait(timeout=1.0)
                return super().record_with_timestamp(content, workspace_id=workspace_id, timestamp=timestamp)
            finally:
                completed.set()

    runtime = SimpleNamespace(
        config=config,
        storage_backend="postgres",
        journal=SlowJournal(),
        read_cache=cache,
        task_queue=None,
    )
    monkeypatch.setattr(
        journal_operations_module,
        "_RECORD_THOUGHT_AUTHORITATIVE_TIMEOUT_SECONDS",
        0.01,
    )

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="daemon-app-shutdown-test")
    task = asyncio.create_task(
        daemon_app_module._run_record_thought_writeback_flush_loop(
            runtime,
            executor=executor,
            poll_seconds=60.0,
        )
    )

    try:
        assert await asyncio.to_thread(started.wait, 0.5)
        await asyncio.sleep(0.05)

        assert cache.count_record_thought_outbox_entries() == 1
        assert completed.is_set() is False

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        shutdown_started = time.monotonic()
        await asyncio.to_thread(executor.shutdown, True)
        assert (time.monotonic() - shutdown_started) < 0.2
        assert completed.is_set() is False
        assert cache.count_record_thought_outbox_entries() == 1
    finally:
        release.set()
        assert await asyncio.to_thread(completed.wait, 1.0)