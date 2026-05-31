from __future__ import annotations

from mcp_memory.context import ApplicationContext
from mcp_memory.mcp.services import read_memory_record_service, search_memory_records_service


def internal_search_memory_records_service(ctx: ApplicationContext, arguments: dict) -> dict:
    return search_memory_records_service(ctx, arguments, caller_kind="internal")


def internal_read_memory_record_service(ctx: ApplicationContext, arguments: dict) -> dict:
    return read_memory_record_service(ctx, arguments, caller_kind="internal")