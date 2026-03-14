from __future__ import annotations

from mcp_memory.context import ApplicationContext
from mcp_memory.core.journal_operations import (
    GetPendingThoughtsOperation,
    RecordThoughtOperation,
)
from mcp_memory.core.runtime_operations import GetMemoryStatsOperation
from mcp_memory.mcp.validation import (
    optional_bool,
    optional_object,
    optional_positive_int,
    optional_string,
    require_string,
    string_list,
)
from mcp_memory.relational.operations import (
    CreateMemoryRecordOperation,
    GetMemoryRecordOperation,
    ImportMarkdownMemoryFileOperation,
    ListMemoryRecordsOperation,
    ReadMemoryRecordOperation,
    SearchMemoryRecordsOperation,
)
from mcp_memory.relational.queries import RelationalMemoryQueries
from mcp_memory.serialization import (
    link_payload,
    memory_record_payload,
    search_result_payload,
)


def record_thought_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.journal is None:
        return {"status": "error", "error": "journal_not_initialized"}

    operation = RecordThoughtOperation(ctx.journal, ctx.task_queue, ctx.workspace_id)
    return operation.execute(require_string(arguments, "content"))


def get_pending_thoughts_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.journal is None:
        return {"status": "error", "error": "journal_not_initialized"}

    operation = GetPendingThoughtsOperation(ctx.journal)
    return operation.execute(optional_positive_int(arguments, "limit", 10))


def get_memory_stats_service(ctx: ApplicationContext) -> dict:
    if ctx.db_manager is None:
        return {"status": "error", "error": "db_not_initialized"}

    operation = GetMemoryStatsOperation(
        ctx.db_manager,
        ctx.journal,
        ctx.config,
        ctx.workspace_id,
        ctx.workspace_root,
        ctx.memory_path,
    )
    return operation.execute()


def create_memory_record_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    operation = CreateMemoryRecordOperation(ctx.repository, ctx.workspace_id)
    record = operation.execute(
        title=require_string(arguments, "title"),
        content=require_string(arguments, "content"),
        workspace_ids=string_list(arguments, "workspace_ids"),
        tags=string_list(arguments, "tags"),
        summary=optional_string(arguments, "summary"),
        memory_type=optional_string(arguments, "memory_type") or "journal",
        status=optional_string(arguments, "status") or "active",
        metadata=optional_object(arguments, "metadata") or {},
    )
    return {"status": "created", "record": memory_record_payload(record)}


def get_memory_record_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    operation = GetMemoryRecordOperation(RelationalMemoryQueries(ctx.repository))
    memory_id = require_string(arguments, "memory_id")
    record = operation.execute(memory_id)
    if record is None:
        return {"status": "error", "error": "memory_not_found"}
    return {"status": "ok", "record": memory_record_payload(record)}


def list_memory_records_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    operation = ListMemoryRecordsOperation(RelationalMemoryQueries(ctx.repository))
    records = operation.execute(
        workspace_id=optional_string(arguments, "workspace_id"),
        memory_type=optional_string(arguments, "memory_type"),
        status=optional_string(arguments, "status"),
        limit=optional_positive_int(arguments, "limit", 100),
    )
    return {
        "status": "ok",
        "records": [memory_record_payload(record) for record in records],
    }


def search_memory_records_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.relational_search is None:
        return {"status": "error", "error": "relational_search_not_initialized"}

    operation = SearchMemoryRecordsOperation(ctx.relational_search)
    results = operation.execute(
        query=require_string(arguments, "query"),
        workspace_id=optional_string(arguments, "workspace_id"),
        limit=optional_positive_int(arguments, "limit", 5),
        memory_type=optional_string(arguments, "memory_type"),
        status=optional_string(arguments, "status"),
        include_superseded=optional_bool(arguments, "include_superseded", False),
    )
    return {
        "status": "ok",
        "results": [search_result_payload(result) for result in results],
    }


def read_memory_record_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.relational_search is None:
        return {"status": "error", "error": "relational_search_not_initialized"}

    operation = ReadMemoryRecordOperation(ctx.relational_search)
    memory_id = require_string(arguments, "memory_id")
    result = operation.execute(memory_id)
    if result is None:
        return {"status": "error", "error": "memory_not_found"}

    return {
        "status": "ok",
        "record": memory_record_payload(result.record),
        "relationships": {
            direction: [link_payload(link) for link in links]
            for direction, links in result.relationships.items()
        },
        "superseded": [memory_record_payload(record) for record in result.superseded],
    }


def import_markdown_memory_file_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    operation = ImportMarkdownMemoryFileOperation(ctx.repository, ctx.workspace_id)
    imported = operation.execute(
        require_string(arguments, "file_path"),
        string_list(arguments, "workspace_ids"),
    )
    return {
        "status": "imported",
        "record": memory_record_payload(imported),
    }