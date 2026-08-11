from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import time
from time import perf_counter
from typing import Any, cast
from uuid import uuid4

from mcp_memory.application.ports import MemorySearchPort
from mcp_memory.core.journal_operations import RecordThoughtOperation
from mcp_memory.integrations.memory_retrieval import (
    MemoryRetrievalPort,
    MemorySearchRequest,
    SearchExecutionDiagnostics,
)
from mcp_memory.management.models import (
    AIConversationListPayload,
    AIConversationPayload,
    MemoryDetailPayload,
    MemoryListPayload,
    MemorySearchPayload,
    MemorySearchResultPayload,
)
from mcp_memory.mcp.telemetry import record_search_diagnostics
from mcp_memory.relational.search import _to_relational_search_result
from mcp_memory.runtime_log_store import _AllWorkspacesSentinel
from mcp_memory.serialization import (
    compact_memory_record_payload,
    link_payload,
    memory_record_payload,
    search_result_payload,
)
from mcp_memory.storage.shared_mode_cache import resolve_shared_mode_cache_state


_USE_SERVICE_WORKSPACE = object()
_SLOW_MEMORY_TOOL_WARNING_MS = 2_000.0


def _resolve_log_workspace_id(
    service_workspace_id: str | None,
    workspace_id: str | None | object,
) -> str | None | _AllWorkspacesSentinel:
    if workspace_id is _USE_SERVICE_WORKSPACE:
        return service_workspace_id
    return cast(str | None | _AllWorkspacesSentinel, workspace_id)


def _resolve_service_workspace_id(
    service_workspace_id: str | None,
    workspace_id: str | None | object,
) -> str | None:
    if workspace_id is _USE_SERVICE_WORKSPACE:
        return service_workspace_id
    return cast(str | None, workspace_id)


@dataclass(frozen=True)
class MemoryServiceDependencies:
    workspace_id: str | None
    journal: Any
    task_queue: Any
    config: Any
    storage_backend: str
    read_cache: Any
    memory_queries: Any
    repository: Any
    relational_search: MemorySearchPort | None
    retrieval: MemoryRetrievalPort | None
    provider_usage_repo: Any
    retrieval_telemetry: Any
    runtime_logs: Any


class MemoryService:
    def __init__(self, dependencies: MemoryServiceDependencies) -> None:
        self._dependencies = dependencies

    def record_thought(
        self,
        content: str,
        *,
        workspace_id: str | None | object = _USE_SERVICE_WORKSPACE,
    ) -> dict[str, object]:
        dependencies = self._dependencies
        if dependencies.journal.journal is None:
            raise ValueError("journal_not_initialized")
        effective_workspace_id = _resolve_service_workspace_id(dependencies.workspace_id, workspace_id)
        suppression_config = None if dependencies.config is None else dependencies.config.ingest_suppression
        cache_state = resolve_shared_mode_cache_state(
            dependencies.config,
            storage_backend=dependencies.storage_backend,
            read_cache=dependencies.read_cache,
        )
        return RecordThoughtOperation(
            dependencies.journal.journal,
            dependencies.task_queue.task_queue,
            effective_workspace_id,
            suppression_config,
            writeback_cache=cache_state.writeback_cache,
            max_outbox_entries=cache_state.max_outbox_entries,
        ).execute(content)

    def list_memories(
        self,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> MemoryListPayload:
        dependencies = self._dependencies
        if dependencies.memory_queries is None:
            return MemoryListPayload()
        records = dependencies.memory_queries.list_memories(
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
        dependencies = self._dependencies
        if dependencies.relational_search is None or dependencies.retrieval is None:
            return MemorySearchPayload()
        started_at = perf_counter()
        request = MemorySearchRequest(
            query=query,
            # Management search is global; workspace context must not filter results.
            workspace_id=None,
            limit=limit,
            adaptive_limit=False,
            ranking_workspace_id=workspace_id,
            memory_type=memory_type,
            status=status,
            include_superseded=include_superseded,
        )
        diagnostics: SearchExecutionDiagnostics | None = None
        if debug:
            outcome, diagnostics = dependencies.retrieval.search_sync_with_diagnostics(
                request,
                debug=True,
            )
            results = [_to_relational_search_result(result) for result in outcome.results]
            for result in results:
                if result.ranking_debug is not None:
                    result.ranking_debug["final_duplicate"] = result.memory_id in (
                        diagnostics.final_duplicate_ids or []
                    )
        else:
            outcome = dependencies.retrieval.search_sync(request)
            results = [_to_relational_search_result(result) for result in outcome.results]
        surfaced_memory_ids = [result.memory_id for result in results]
        touch_last_surfaced = getattr(dependencies.repository, "touch_last_surfaced", None)
        if surfaced_memory_ids and callable(touch_last_surfaced):
            touch_last_surfaced(
                surfaced_memory_ids,
                datetime.now(UTC).isoformat(),
                best_effort=True,
            )
        duration_ms = (perf_counter() - started_at) * 1000.0
        invocation_id = str(uuid4())
        dependencies.retrieval_telemetry.record_search(
            invocation_id=invocation_id,
            caller_kind="operator",
            query=query,
            surfaced_memory_ids=[result.memory_id for result in results],
            duration_ms=duration_ms,
        )
        if diagnostics is not None:
            record_search_diagnostics(
                dependencies.retrieval_telemetry,
                invocation_id=invocation_id,
                diagnostics=diagnostics,
            )
        self._log_slow_memory_tool_operation(
            tool_name="management.search_memories",
            duration_ms=duration_ms,
            data={
                "query": query,
                "result_count": len(results),
                "storage_backend": dependencies.storage_backend,
                "workspace_id": workspace_id,
            },
        )
        return MemorySearchPayload(
            results=[
                MemorySearchResultPayload(**search_result_payload(result))
                for result in results
            ]
        )

    def get_memory_detail(self, memory_id: str):
        dependencies = self._dependencies
        if dependencies.memory_queries is None:
            raise ValueError("repository_not_initialized")

        record = dependencies.memory_queries.get_memory(memory_id)
        if record is None:
            raise ValueError("memory_not_found")

        outgoing = dependencies.memory_queries.get_links(memory_id, direction="outgoing")
        incoming = dependencies.memory_queries.get_links(memory_id, direction="incoming")
        superseded = [
            memory_record_payload(target)
            for target in dependencies.memory_queries.get_superseded_records(memory_id)
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
        dependencies = self._dependencies
        if dependencies.memory_queries is None:
            raise ValueError("repository_not_initialized")

        source = dependencies.memory_queries.get_memory(source_id)
        if source is None:
            raise ValueError("source_memory_not_found")
        if not target_id.startswith("ext:") and dependencies.memory_queries.get_memory(target_id) is None:
            raise ValueError("target_memory_not_found")

        assert dependencies.repository is not None
        link = dependencies.repository.add_link(
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
        dependencies = self._dependencies
        if dependencies.memory_queries is None:
            raise ValueError("repository_not_initialized")

        assert dependencies.repository is not None
        deleted = dependencies.repository.remove_link(
            source_id=source_id,
            target_id=target_id,
            link_type=link_type,
        )
        if not deleted:
            raise ValueError("link_not_found")
        return {"status": "deleted"}

    def list_ai_conversations(
        self,
        *,
        workspace_id: str | None | object = _USE_SERVICE_WORKSPACE,
        request_id: str | None = None,
        task_name: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> AIConversationListPayload:
        dependencies = self._dependencies
        effective_workspace_id = _resolve_log_workspace_id(dependencies.workspace_id, workspace_id)
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
                    input_tokens=record.input_tokens,
                    output_tokens=record.output_tokens,
                    cached_input_tokens=record.cached_input_tokens,
                    cache_write_tokens=record.cache_write_tokens,
                    reasoning_tokens=record.reasoning_tokens,
                    total_tokens=record.total_tokens,
                    token_usage_source=record.token_usage_source,
                )
                for record in dependencies.provider_usage_repo.list_conversations(
                    workspace_id=effective_workspace_id,
                    request_id=request_id,
                    task_name=task_name,
                    status=status,
                    limit=limit,
                )
            ]
        )

    def _log_slow_memory_tool_operation(
        self,
        *,
        tool_name: str,
        duration_ms: float,
        data: dict[str, object],
    ) -> None:
        if duration_ms < _SLOW_MEMORY_TOOL_WARNING_MS:
            return
        self._dependencies.runtime_logs.write_log(
            source="memory-tool",
            logger_name="mcp_memory.management.service",
            level="WARNING",
            message=f"Slow {tool_name} operation",
            created_at=time.time(),
            data={"tool_name": tool_name, "duration_ms": round(duration_ms, 3)} | data,
        )
