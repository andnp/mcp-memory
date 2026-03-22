from __future__ import annotations

import os
from pathlib import Path
import time

from mcp_memory.context import ApplicationContext
from mcp_memory.core import MemoryPipeline
from mcp_memory.core.journal_operations import RecordThoughtOperation
from mcp_memory.core.task_handlers import TRIGGERABLE_BACKGROUND_TASK_NAMES
from mcp_memory.core.task_handlers import task_priority
from mcp_memory.management.agent_run_reporting import build_recent_agent_runs
from mcp_memory.management.analytics_reporting import build_nerd_metrics
from mcp_memory.management.health_reporting import build_embedding_status, build_search_health
from mcp_memory.management.overview_reporting import build_overview
from mcp_memory.management.models import (
    AgentRunHistoryListPayload,
    AIConversationListPayload,
    AIConversationPayload,
    HealthPayload,
    MemoryListPayload,
    MemoryDetailPayload,
    MemorySearchResultPayload,
    MemorySearchPayload,
    NerdMetricsPayload,
    RuntimeLogListPayload,
    RuntimeLogPrunePayload,
    RuntimeLogPayload,
    RuntimeLogSummaryPayload,
    TaskDetailPayload,
    TaskListPayload,
)
from mcp_memory.provider_usage_store import ProviderUsageRepository
from mcp_memory.process_termination import send_process_signal as _send_process_signal
from mcp_memory.process_termination import terminate_process as _terminate_process_with_scope
from mcp_memory.process_termination import wait_for_process_exit as _wait_for_process_exit
from mcp_memory.runtime_log_store import RuntimeLogRepository
from mcp_memory.serialization import (
    compact_memory_record_payload,
    link_payload,
    memory_record_payload,
    search_result_payload,
    task_payload,
)


class ManagementService:
    def __init__(self, ctx: ApplicationContext, controller) -> None:
        pipeline = MemoryPipeline.from_context(ctx, controller)
        self._controller = controller
        self._db_manager = ctx.db_manager
        self._workspace_id = ctx.workspace_id
        self._runtime_info = pipeline.runtime_info
        self._journal = pipeline.journal
        self._task_queue = pipeline.task_queue
        self._memory_queries = pipeline.memory_queries
        self._repository = ctx.repository
        self._provider_usage = ProviderUsageRepository(ctx.db_manager, workspace_id=self._workspace_id)
        self._runtime_logs = RuntimeLogRepository(
            ctx.db_manager,
            workspace_id=self._workspace_id,
            config=None if ctx.config is None else ctx.config.logging,
        )
        self._embedder = ctx.embedder
        self._relational_search = ctx.relational_search
        self._config = ctx.config
        self._ai_json_provider = getattr(ctx, "ai_json_provider", None) or getattr(ctx, "ai_provider", None)
        self._ai_agent_provider = getattr(ctx, "ai_agent_provider", None)
        self._ai_provider_registry = getattr(ctx, "ai_provider_registry", None) or {}
        self._dashboard_static_root = Path(__file__).with_name("static")
        self._dashboard_static_path = self._dashboard_static_root / "index.html"
        self._dashboard_dist_path = self._dashboard_static_root / "dist" / "index.html"
        self._dashboard_asset_root = self._dashboard_static_root / "dist" / "assets"

    def get_health(self):
        embedder_status = build_embedding_status(self._embedder)
        return HealthPayload(
            status="ok",
            workspace_id=self._runtime_info.workspace_id,
            workspace_root=str(self._runtime_info.workspace_root) if self._runtime_info.workspace_root is not None else None,
            memory_path=str(self._runtime_info.memory_path) if self._runtime_info.memory_path is not None else None,
            db_path=str(self._runtime_info.db_path) if self._runtime_info.db_path is not None else None,
            runtime_active=bool(getattr(self._controller, "has_runtime", self._runtime_info.runtime_active)),
            client_count=int(getattr(self._controller, "client_count", self._runtime_info.client_count)),
            task_queue_enabled=self._runtime_info.task_queue_enabled,
            embeddings=embedder_status,
            search=build_search_health(self._relational_search),
        )

    def get_overview(
        self,
        *,
        scope: str | None = None,
        workspace_id: str | None = None,
        recent_limit: int = 10,
        failed_limit: int = 10,
    ):
        effective_workspace_id = self._resolve_scoped_workspace_id(
            scope=scope,
            workspace_id=workspace_id,
        )
        return build_overview(
            memory_queries=self._memory_queries,
            repository=self._repository,
            task_queue=self._task_queue,
            runtime_info=self._runtime_info,
            db_manager=self._db_manager,
            workspace_id=effective_workspace_id,
            provider_usage_repo=self._provider_usage,
            runtime_logs_repo=self._runtime_logs,
            embedder=self._embedder,
            relational_search=self._relational_search,
            recent_limit=recent_limit,
            failed_limit=failed_limit,
        )

    def repair_search_index(self) -> dict[str, int | bool | str | None]:
        if self._relational_search is None:
            raise ValueError("search_not_initialized")
        return self._relational_search.rebuild_semantic_index()

    def enqueue_background_task(
        self,
        task_name: str,
        *,
        force: bool = False,
    ) -> dict:
        if task_name not in TRIGGERABLE_BACKGROUND_TASK_NAMES:
            raise ValueError(f"unknown_background_task:{task_name}")

        payload = {"workspace_id": None}
        if force:
            task = self._task_queue.enqueue(
                task_name=task_name,
                workspace_id=None,
                data=payload,
                priority=task_priority(task_name),
            )
            return {"status": "enqueued", "created": True, "task": task_payload(task)}

        task, created = self._task_queue.enqueue_unique(
            task_name=task_name,
            workspace_id=None,
            data=payload,
            priority=task_priority(task_name),
        )
        return {
            "status": "enqueued" if created else "already_pending",
            "created": created,
            "task": task_payload(task),
        }

    def enqueue_all_background_tasks(self, *, force: bool = False) -> list[dict]:
        return [
            self.enqueue_background_task(task_name, force=force)
            for task_name in TRIGGERABLE_BACKGROUND_TASK_NAMES
        ]

    def cancel_task(
        self,
        task_id: str,
        *,
        cancelled_by: str = "cli",
        reason: str = "cancelled_by_user",
    ) -> dict:
        task = self._task_queue.request_cancel(
            task_id,
            cancelled_by=cancelled_by,
            reason=reason,
        )
        signal_sent = False
        if task.status == "running" and task.subprocess_pid is not None:
            signal_sent = _terminate_process(task.subprocess_pid)
        return {
            "status": "cancelled" if task.status == "cancelled" else "cancellation_requested",
            "signal_sent": signal_sent,
            "task": task_payload(self._task_queue.get_task(task_id)),
        }

    def list_ai_conversations(
        self,
        *,
        request_id: str | None = None,
        task_name: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> AIConversationListPayload:
        return AIConversationListPayload(
            conversations=[
                AIConversationPayload(
                    id=record.id,
                    request_id=record.request_id,
                    attempt=record.attempt,
                    workspace_id=record.workspace_id,
                    task_name=record.task_name,
                    task_id=record.task_id,
                    provider_key=record.provider_key,
                    provider_name=record.provider_name,
                    model_name=record.model_name,
                    subprocess_pid=record.subprocess_pid,
                    prompt_text=record.prompt_text,
                    response_text=record.response_text,
                    parsed=record.parsed,
                    status=record.status,
                    error_text=record.error_text,
                    reason_category=record.reason_category,
                    reason_code=record.reason_code,
                    retry_delay_seconds=record.retry_delay_seconds,
                    started_at=record.started_at,
                    completed_at=record.completed_at,
                    duration_seconds=record.duration_seconds,
                )
                for record in self._provider_usage.list_conversations(
                    request_id=request_id,
                    task_name=task_name,
                    status=status,
                    limit=limit,
                )
            ]
        )

    def get_memory_detail(self, memory_id: str):
        if self._memory_queries is None:
            raise ValueError("repository_not_initialized")

        record = self._memory_queries.get_memory(memory_id)
        if record is None:
            raise ValueError("memory_not_found")

        outgoing = self._memory_queries.get_links(memory_id, direction="outgoing")
        incoming = self._memory_queries.get_links(memory_id, direction="incoming")
        superseded = [
            memory_record_payload(target)
            for target in self._memory_queries.get_superseded_records(memory_id)
        ]

        return MemoryDetailPayload(
            record=memory_record_payload(record),
            relationships={
                "incoming": [link_payload(link) for link in incoming],
                "outgoing": [link_payload(link) for link in outgoing],
            },
            superseded=superseded,
        )

    def create_memory_link(
        self,
        *,
        source_id: str,
        target_id: str,
        link_type: str,
        context: str = "",
    ) -> dict:
        if self._memory_queries is None:
            raise ValueError("repository_not_initialized")

        source = self._memory_queries.get_memory(source_id)
        if source is None:
            raise ValueError("source_memory_not_found")
        if not target_id.startswith("ext:") and self._memory_queries.get_memory(target_id) is None:
            raise ValueError("target_memory_not_found")

        assert self._repository is not None
        link = self._repository.add_link(
            source_id=source_id,
            target_id=target_id,
            link_type=link_type,
            context=context,
        )
        return {"status": "created", "link": link_payload(link)}

    def delete_memory_link(
        self,
        *,
        source_id: str,
        target_id: str,
        link_type: str,
    ) -> dict:
        if self._memory_queries is None:
            raise ValueError("repository_not_initialized")

        assert self._repository is not None
        deleted = self._repository.remove_link(
            source_id=source_id,
            target_id=target_id,
            link_type=link_type,
        )
        if not deleted:
            raise ValueError("link_not_found")
        return {"status": "deleted"}

    def list_tasks(
        self,
        status: str | None = None,
        workspace_id: str | None = None,
        limit: int = 20,
    ) -> TaskListPayload:
        tasks = self._task_queue.list_tasks(
            status=status,
            workspace_id=workspace_id,
            limit=limit,
        )
        return TaskListPayload(tasks=[task_payload(task) for task in tasks])

    def list_recent_agent_runs(self, *, limit: int = 20, detail_level: str = "compact") -> AgentRunHistoryListPayload:
        return AgentRunHistoryListPayload(
            runs=build_recent_agent_runs(
                self._db_manager,
                self._workspace_id,
                limit=limit,
                detail_level=detail_level,
            )
        )

    def get_task_detail(self, task_id: str) -> TaskDetailPayload:
        task = self._task_queue.get_task(task_id)
        runs = self._task_queue.list_task_runs(task_id=task_id, limit=50)
        from mcp_memory.management.agent_run_reporting import build_agent_run_history_payload

        return TaskDetailPayload(
            task=task_payload(task),
            runs=[
                build_agent_run_history_payload(
                    task_id=run.task_id,
                    task_name=run.task_name,
                    status=run.status,
                    started_at=run.started_at,
                    completed_at=run.completed_at,
                    duration_seconds=run.duration_seconds,
                    error_text=run.error_text,
                    result=run.result,
                    detail_level="full",
                )
                for run in runs
            ],
        )

    def get_nerd_metrics(
        self,
        *,
        scope: str | None = None,
        workspace_id: str | None = None,
        window_hours: int = 24,
        bucket_minutes: int = 60,
        now: float | None = None,
    ) -> NerdMetricsPayload:
        effective_workspace_id = self._resolve_scoped_workspace_id(
            scope=scope,
            workspace_id=workspace_id,
        )
        return build_nerd_metrics(
            db_manager=self._db_manager,
            workspace_id=effective_workspace_id,
            task_queue=self._task_queue,
            provider_usage_repo=ProviderUsageRepository(
                self._db_manager,
                workspace_id=effective_workspace_id,
            ),
            config=self._config,
            ai_json_provider=self._ai_json_provider,
            ai_agent_provider=self._ai_agent_provider,
            ai_provider_registry=self._ai_provider_registry,
            relational_search=self._relational_search,
            window_hours=window_hours,
            bucket_minutes=bucket_minutes,
            now=now,
        )

    def _resolve_scoped_workspace_id(
        self,
        *,
        scope: str | None,
        workspace_id: str | None,
    ) -> str | None:
        if workspace_id is not None:
            return workspace_id
        if scope is None:
            return self._workspace_id
        if scope == "global":
            return None
        if scope == "workspace":
            if self._workspace_id is None:
                raise ValueError("workspace_id_required_for_workspace_scope")
            return self._workspace_id
        raise ValueError("scope_must_be_global_or_workspace")

    def record_thought(self, content: str) -> dict[str, object]:
        if self._journal.journal is None:
            raise ValueError("journal_not_initialized")
        return RecordThoughtOperation(
            self._journal.journal,
            self._task_queue.task_queue,
            self._runtime_info.workspace_id,
        ).execute(content)

    def list_memories(
        self,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> MemoryListPayload:
        if self._memory_queries is None:
            return MemoryListPayload()
        records = self._memory_queries.list_memories(
            workspace_id=workspace_id,
            memory_type=memory_type,
            status=status,
            limit=limit,
        )
        return MemoryListPayload(
            records=[compact_memory_record_payload(record) for record in records]
        )

    def search_memories(
        self,
        *,
        query: str,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
        limit: int = 10,
        debug: bool = False,
    ) -> MemorySearchPayload:
        if self._relational_search is None:
            return MemorySearchPayload()
        return MemorySearchPayload(
            results=[
                MemorySearchResultPayload(**search_result_payload(result))
                for result in self._relational_search.search_memories(
                    query,
                    workspace_id=workspace_id,
                    limit=limit,
                    memory_type=memory_type,
                    status=status,
                    include_superseded=include_superseded,
                    debug=debug,
                )
            ]
        )

    def list_logs(
        self,
        *,
        level: str | None = None,
        logger_name: str | None = None,
        source: str | None = None,
        query: str | None = None,
        after: float | None = None,
        before: float | None = None,
        limit: int = 50,
    ) -> RuntimeLogListPayload:
        if self._db_manager is None:
            return RuntimeLogListPayload()
        return RuntimeLogListPayload(
            logs=[
                RuntimeLogPayload(
                    id=record.id,
                    created_at=record.created_at,
                    level=record.level,
                    logger_name=record.logger_name,
                    source=record.source,
                    message=record.message,
                    data=record.data,
                )
                for record in self._runtime_logs.list_logs(
                    level=level,
                    logger_name=logger_name,
                    source=source,
                    query=query,
                    after=after,
                    before=before,
                    limit=limit,
                )
            ]
        )

    def summarize_logs(
        self,
        *,
        level: str | None = None,
        logger_name: str | None = None,
        source: str | None = None,
        query: str | None = None,
        after: float | None = None,
        before: float | None = None,
    ) -> RuntimeLogSummaryPayload:
        if self._db_manager is None:
            return RuntimeLogSummaryPayload()
        summary = self._runtime_logs.summarize_logs(
            level=level,
            logger_name=logger_name,
            source=source,
            query=query,
            after=after,
            before=before,
        )
        return RuntimeLogSummaryPayload(
            total=summary.total,
            by_level=summary.by_level,
            by_source=summary.by_source,
        )

    def prune_logs(
        self,
        *,
        max_runtime_logs: int | None = None,
        max_log_age_days: int | None = None,
    ) -> RuntimeLogPrunePayload:
        config = self._runtime_logs.retention_policy
        deleted = self._runtime_logs.prune_logs(
            max_runtime_logs=max_runtime_logs,
            max_age_days=max_log_age_days,
        )
        resolved_max_runtime_logs = config.max_runtime_logs if max_runtime_logs is None else max_runtime_logs
        resolved_max_log_age_days = config.max_log_age_days if max_log_age_days is None else max_log_age_days
        return RuntimeLogPrunePayload(
            deleted=deleted,
            max_runtime_logs=resolved_max_runtime_logs,
            max_log_age_days=resolved_max_log_age_days,
        )

    def load_dashboard_html(self):
        html_path = self._dashboard_dist_path if self._dashboard_dist_path.exists() else self._dashboard_static_path
        return html_path.read_text(encoding="utf-8")

    def resolve_dashboard_asset_path(self, asset_path: str) -> Path | None:
        if not asset_path.strip() or not self._dashboard_asset_root.exists():
            return None
        candidate = (self._dashboard_asset_root / asset_path).resolve()
        asset_root = self._dashboard_asset_root.resolve()
        if asset_root not in candidate.parents and candidate != asset_root:
            return None
        if not candidate.is_file():
            return None
        return candidate



def _terminate_process(pid: int) -> bool:
    termination = _terminate_process_with_scope(
        pid,
        deadline=time.monotonic() + 1.0,
        poll_interval_seconds=0.05,
        is_process_running=_is_process_alive,
        send_signal=lambda process_id, sig: _send_process_signal(
            process_id,
            sig,
            scope="pid",
            suppress_permission_errors=True,
        ),
        wait_for_exit=lambda process_id, *, deadline, poll_interval_seconds: _wait_for_process_exit(
            process_id,
            deadline=deadline,
            poll_interval_seconds=poll_interval_seconds,
            is_process_running=_is_process_alive,
        ),
    )
    return termination.signal_sent


def _is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
