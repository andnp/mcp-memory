from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from mcp_memory.config import resolve_backup_dir
from mcp_memory.core.journal_operations import flush_record_thought_writeback_outbox
from mcp_memory.management.frontend_build import ensure_dashboard_frontend_built
from mcp_memory.sqlite_backup import create_and_prune_sqlite_backup, log_shared_storage_risks
from mcp_memory.storage.shared_mode_cache import resolve_shared_mode_cache_state


logger = logging.getLogger(__name__)
RECORD_THOUGHT_WRITEBACK_FLUSH_INTERVAL_SECONDS = 5.0


def ensure_dashboard_frontend_ready(static_root: Path) -> None:
    result = ensure_dashboard_frontend_built(static_root=static_root)
    if result.status == "up_to_date":
        logger.debug("Dashboard frontend bundle is up to date")
        return
    if result.status == "built":
        logger.info("Dashboard frontend rebuilt at startup from %s", result.frontend_root)
        return
    logger.warning(
        "Dashboard frontend build step did not complete cleanly: status=%s returncode=%s message=%s",
        result.status,
        result.returncode,
        result.message,
    )


async def warm_embedding_model(embedder: Any) -> bool:
    if embedder is None:
        return False
    cache_model = getattr(embedder, "cache_model", None)
    if not callable(cache_model):
        return False
    return bool(await asyncio.to_thread(cache_model))


def consume_embedding_warmup_result(task: asyncio.Task[bool]) -> None:
    with suppress(asyncio.CancelledError):
        task.result()


async def run_periodic_backup_loop(runtime, *, backup_fn=create_and_prune_sqlite_backup) -> None:
    config = getattr(runtime, "config", None)
    db_manager = getattr(runtime, "db_manager", None)
    memory_path = getattr(runtime, "memory_path", None)
    storage_backend = getattr(runtime, "storage_backend", None) or "sqlite"
    if config is None or db_manager is None or memory_path is None:
        return

    backup_config = config.backups
    app_data_dir = Path(memory_path).parent
    if backup_config.warn_on_shared_storage:
        log_shared_storage_risks(app_data_dir)

    if storage_backend != "sqlite":
        logger.info("Skipping periodic SQLite backup loop for storage backend %s", storage_backend)
        return
    if not backup_config.enabled:
        return

    backup_dir = resolve_backup_dir()
    should_run_immediately = backup_config.create_startup_snapshot
    while True:
        if not should_run_immediately:
            await asyncio.sleep(backup_config.interval_seconds)
        should_run_immediately = False
        try:
            result = await asyncio.to_thread(
                backup_fn,
                Path(db_manager.db_path),
                backup_dir,
                max_snapshots=backup_config.max_snapshots,
            )
            logger.info(
                "SQLite backup snapshot created: %s pruned=%s",
                result.backup_path,
                len(result.pruned_paths),
            )
        except Exception as exc:
            logger.warning("SQLite backup snapshot failed: %s", exc)


@dataclass(frozen=True)
class RecordThoughtWritebackFlushContext:
    journal: Any
    task_queue: Any
    writeback_cache: Any
    suppression_config: Any = None


def resolve_record_thought_writeback_flush_context(runtime) -> RecordThoughtWritebackFlushContext | None:
    cache_state = resolve_shared_mode_cache_state(
        getattr(runtime, "config", None),
        storage_backend=getattr(runtime, "storage_backend", None),
        read_cache=getattr(runtime, "read_cache", None),
    )
    if not cache_state.writeback_active:
        return None
    journal = getattr(runtime, "journal", None)
    if journal is None or cache_state.writeback_cache is None:
        return None
    return RecordThoughtWritebackFlushContext(
        journal=journal,
        task_queue=getattr(runtime, "task_queue", None),
        writeback_cache=cache_state.writeback_cache,
        suppression_config=(
            None if getattr(runtime, "config", None) is None else runtime.config.ingest_suppression
        ),
    )


def flush_record_thought_writeback_once(runtime, *, flush_fn=flush_record_thought_writeback_outbox) -> int:
    flush_context = resolve_record_thought_writeback_flush_context(runtime)
    if flush_context is None:
        return 0
    result = flush_fn(
        flush_context.journal,
        task_queue=flush_context.task_queue,
        suppression_config=flush_context.suppression_config,
        writeback_cache=flush_context.writeback_cache,
    )
    if result.flushed_count > 0:
        logger.info(
            "Flushed %s queued record_thought writeback entr%s",
            result.flushed_count,
            "y" if result.flushed_count == 1 else "ies",
        )
    return result.flushed_count


async def run_record_thought_writeback_flush_loop(
    runtime,
    *,
    executor: ThreadPoolExecutor,
    poll_seconds: float = RECORD_THOUGHT_WRITEBACK_FLUSH_INTERVAL_SECONDS,
) -> None:
    loop = asyncio.get_running_loop()
    while True:
        try:
            await loop.run_in_executor(executor, flush_record_thought_writeback_once, runtime)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Record-thought writeback flush loop failed", exc_info=True)
        await asyncio.sleep(poll_seconds)
