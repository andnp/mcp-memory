from __future__ import annotations

from dataclasses import dataclass
import logging
from threading import Event, Lock, Thread
import time
from typing import cast

from mcp_memory.core.maintenance_idle import resume_paused_recurring_maintenance
from mcp_memory.core.system1_scheduling import System1IngestScheduleResult, schedule_system1_ingest
from mcp_memory.core.journal import JournalEntry, System1Journal
from mcp_memory.core.ports.tasks import TaskQueue, TaskRecord
from mcp_memory.storage.shared_read_cache import SharedReadCache


logger = logging.getLogger(__name__)
_RECORD_THOUGHT_AUTHORITATIVE_TIMEOUT_SECONDS = 5.0
_RECORD_THOUGHT_OUTBOX_FLUSH_LIMIT = 8
_RECORD_THOUGHT_WRITE_SLOT = Lock()


class _AuthoritativeRecordBusyError(RuntimeError):
    pass


@dataclass(frozen=True)
class RecordThoughtWritebackFlushResult:
    flushed_count: int = 0
    flushed_workspace_ids: tuple[str | None, ...] = ()


def _normalize_record_thought_content(content: str) -> str:
    if not content or not content.strip():
        raise ValueError("Journal content cannot be empty")
    return content.strip()


def _record_journal_entry(
    journal: System1Journal,
    *,
    content: str,
    workspace_id: str | None,
    timestamp: float,
) -> JournalEntry:
    record_with_timestamp = getattr(journal, "record_with_timestamp", None)
    if callable(record_with_timestamp):
        return cast(
            JournalEntry,
            record_with_timestamp(content, workspace_id=workspace_id, timestamp=timestamp),
        )
    return journal.record(content, workspace_id=workspace_id)


def _run_bounded_authoritative_record(
    journal: System1Journal,
    *,
    content: str,
    workspace_id: str | None,
    timestamp: float,
    timeout_seconds: float,
) -> JournalEntry:
    if not _RECORD_THOUGHT_WRITE_SLOT.acquire(blocking=False):
        raise _AuthoritativeRecordBusyError("authoritative_record_in_progress")

    result: dict[str, object] = {}
    completed = Event()

    def _worker() -> None:
        try:
            result["entry"] = _record_journal_entry(
                journal,
                content=content,
                workspace_id=workspace_id,
                timestamp=timestamp,
            )
        except Exception as exc:  # pragma: no cover - exercised through caller assertions
            result["error"] = exc
        finally:
            _RECORD_THOUGHT_WRITE_SLOT.release()
            completed.set()

    Thread(target=_worker, daemon=True, name="record-thought-authoritative-write").start()

    if not completed.wait(timeout_seconds):
        raise TimeoutError("authoritative_record_timed_out")
    error = result.get("error")
    if isinstance(error, Exception):
        raise error
    return cast(JournalEntry, result["entry"])


def _is_writeback_fallback_error(exc: Exception) -> bool:
    if isinstance(exc, (_AuthoritativeRecordBusyError, ConnectionError, OSError, TimeoutError)):
        return True
    return isinstance(exc, RuntimeError) and str(exc) == "journal_unavailable"


def _ordered_workspace_ids(workspace_ids: list[str | None]) -> list[str | None]:
    ordered: list[str | None] = []
    seen: set[str | None] = set()
    for workspace_id in workspace_ids:
        if workspace_id in seen:
            continue
        seen.add(workspace_id)
        ordered.append(workspace_id)
    return ordered


def _schedule_record_thought_follow_up(
    *,
    journal: System1Journal,
    task_queue: TaskQueue | None,
    workspace_ids: list[str | None],
    suppression_config=None,
    resumed_now: float | None = None,
) -> tuple[list[TaskRecord], list[System1IngestScheduleResult]]:
    if task_queue is None:
        return [], []

    resumed_tasks = resume_paused_recurring_maintenance(task_queue, now=resumed_now)
    scheduled_ingests: list[System1IngestScheduleResult] = []
    for workspace_id in _ordered_workspace_ids(workspace_ids):
        scheduled = schedule_system1_ingest(
            task_queue,
            journal,
            workspace_id,
            suppression_config=suppression_config,
        )
        if scheduled is not None:
            scheduled_ingests.append(scheduled)
    return resumed_tasks, scheduled_ingests


def flush_record_thought_writeback_outbox(
    journal: System1Journal,
    *,
    task_queue: TaskQueue | None,
    suppression_config,
    writeback_cache: SharedReadCache | None,
    flush_limit: int = _RECORD_THOUGHT_OUTBOX_FLUSH_LIMIT,
) -> RecordThoughtWritebackFlushResult:
    if writeback_cache is None or flush_limit < 1:
        return RecordThoughtWritebackFlushResult()

    flushed_workspace_ids: list[str | None] = []
    queued_entries = writeback_cache.list_record_thought_outbox_entries(limit=flush_limit)
    for queued_entry in queued_entries:
        try:
            _run_bounded_authoritative_record(
                journal,
                content=queued_entry.content,
                workspace_id=queued_entry.workspace_id,
                timestamp=queued_entry.timestamp,
                timeout_seconds=_RECORD_THOUGHT_AUTHORITATIVE_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            if _is_writeback_fallback_error(exc):
                logger.warning("Writeback outbox flush paused after authoritative write failure", exc_info=True)
                break
            logger.warning("Writeback outbox flush failed unexpectedly", exc_info=True)
            break
        writeback_cache.delete_record_thought_outbox_entries([queued_entry.outbox_id])
        flushed_workspace_ids.append(queued_entry.workspace_id)

    if not flushed_workspace_ids:
        return RecordThoughtWritebackFlushResult()

    ordered_workspace_ids = _ordered_workspace_ids(flushed_workspace_ids)
    _schedule_record_thought_follow_up(
        journal=journal,
        task_queue=task_queue,
        workspace_ids=ordered_workspace_ids,
        suppression_config=suppression_config,
        resumed_now=time.time(),
    )
    return RecordThoughtWritebackFlushResult(
        flushed_count=len(flushed_workspace_ids),
        flushed_workspace_ids=tuple(ordered_workspace_ids),
    )


class RecordThoughtOperation:
    def __init__(
        self,
        journal: System1Journal,
        task_queue: TaskQueue | None,
        workspace_id: str | None,
        suppression_config=None,
        writeback_cache: SharedReadCache | None = None,
        max_outbox_entries: int | None = None,
    ) -> None:
        self._journal = journal
        self._task_queue = task_queue
        self._workspace_id = workspace_id
        self._suppression_config = suppression_config
        self._writeback_cache = writeback_cache
        self._max_outbox_entries = max_outbox_entries

    def execute(self, content: str) -> dict:
        normalized_content = _normalize_record_thought_content(content)
        recorded_at = time.time()
        entry = self._record_entry_with_writeback(normalized_content, recorded_at=recorded_at)
        payload: dict[str, object] = {
            "status": "recorded",
            "entry": entry.to_dict(),
        }

        if entry.status == "queued_writeback":
            payload["degraded"] = True
            payload["cache_status"] = "writeback_queued"
            if self._writeback_cache is not None:
                payload["writeback_queue_depth"] = self._writeback_cache.count_record_thought_outbox_entries()
            return payload

        resumed_tasks, scheduled_ingests = _schedule_record_thought_follow_up(
            journal=self._journal,
            task_queue=self._task_queue,
            workspace_ids=[self._workspace_id],
            suppression_config=self._suppression_config,
            resumed_now=entry.timestamp,
        )
        if resumed_tasks:
            payload["resumed_maintenance_tasks"] = [
                {
                    "id": task.id,
                    "status": task.status,
                    "task_name": task.task_name,
                    "workspace_id": task.workspace_id,
                }
                for task in resumed_tasks
            ]
        scheduled_threshold_ingest = next(
            (scheduled for scheduled in scheduled_ingests if scheduled.trigger == "system1_threshold"),
            None,
        )
        if scheduled_threshold_ingest is not None:
            ingest_task = scheduled_threshold_ingest.task
            payload["ingest_task"] = {
                "id": ingest_task.id,
                "status": ingest_task.status,
                "task_name": ingest_task.task_name,
                "workspace_id": ingest_task.workspace_id,
                "created": scheduled_threshold_ingest.created,
            }

        return payload

    def _record_entry_with_writeback(self, content: str, *, recorded_at: float) -> JournalEntry:
        if self._writeback_cache is None or self._max_outbox_entries is None:
            return _record_journal_entry(
                self._journal,
                content=content,
                workspace_id=self._workspace_id,
                timestamp=recorded_at,
            )
        try:
            return _run_bounded_authoritative_record(
                self._journal,
                content=content,
                workspace_id=self._workspace_id,
                timestamp=recorded_at,
                timeout_seconds=_RECORD_THOUGHT_AUTHORITATIVE_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            if not _is_writeback_fallback_error(exc):
                raise
            queued_entry = self._writeback_cache.enqueue_record_thought_outbox_entry(
                content=content,
                workspace_id=self._workspace_id,
                timestamp=recorded_at,
                max_entries=self._max_outbox_entries,
            )
            if queued_entry is None:
                raise
            return JournalEntry(
                id=queued_entry.outbox_id,
                content=queued_entry.content,
                workspace_id=queued_entry.workspace_id,
                timestamp=queued_entry.timestamp,
                status="queued_writeback",
            )
