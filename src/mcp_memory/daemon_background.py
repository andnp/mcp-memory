from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

from mcp_memory.config import resolve_backup_dir
from mcp_memory.core.journal_operations import flush_record_thought_writeback_outbox
from mcp_memory.daemon_ports import BackupPort, EmbeddingModelPort, WritebackDependencies, WritebackFlushPort
from mcp_memory.management.frontend_build import ensure_dashboard_frontend_built
from mcp_memory.mcp.runtime import DaemonCapabilityBundle
from mcp_memory.sqlite_backup import create_and_prune_sqlite_backup, log_shared_storage_risks

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


async def warm_embedding_model(daemon: DaemonCapabilityBundle) -> bool:
    embedder = daemon.resources.embedder
    if not isinstance(embedder, EmbeddingModelPort):
        return False
    return bool(await asyncio.to_thread(embedder.cache_model))


def consume_embedding_warmup_result(task: asyncio.Task[bool]) -> None:
    with suppress(asyncio.CancelledError):
        task.result()


async def run_periodic_backup_loop(
    daemon: DaemonCapabilityBundle,
    *,
    backup_fn: BackupPort = create_and_prune_sqlite_backup,
) -> None:
    config = daemon.memory.config
    db_manager = daemon.resources.storage.db_manager
    memory_path = daemon.memory.memory_path
    storage_backend = daemon.resources.storage.backend or "sqlite"
    if config is None or db_manager is None or memory_path is None:
        return

    backup_config = config.backups
    app_data_dir = memory_path.parent
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
                memory_path / "indices" / "memory.db",
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


def resolve_record_thought_writeback_flush_context(
    daemon: DaemonCapabilityBundle,
) -> WritebackDependencies | None:
    return daemon.writeback


def flush_record_thought_writeback_once(
    daemon: DaemonCapabilityBundle,
    *,
    flush_fn: WritebackFlushPort = flush_record_thought_writeback_outbox,
) -> int:
    writeback = resolve_record_thought_writeback_flush_context(daemon)
    if writeback is None:
        return 0
    result = flush_fn(
        writeback.journal,
        task_queue=writeback.task_queue,
        suppression_config=writeback.suppression_config,
        writeback_cache=writeback.writeback_cache,
    )
    if result.flushed_count > 0:
        logger.info(
            "Flushed %s queued record_thought writeback entr%s",
            result.flushed_count,
            "y" if result.flushed_count == 1 else "ies",
        )
    return result.flushed_count


async def run_record_thought_writeback_flush_loop(
    daemon: DaemonCapabilityBundle,
    *,
    executor: ThreadPoolExecutor,
    poll_seconds: float = RECORD_THOUGHT_WRITEBACK_FLUSH_INTERVAL_SECONDS,
) -> None:
    loop = asyncio.get_running_loop()
    while True:
        try:
            await loop.run_in_executor(executor, flush_record_thought_writeback_once, daemon)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Record-thought writeback flush loop failed", exc_info=True)
        await asyncio.sleep(poll_seconds)
