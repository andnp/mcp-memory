"""Inventory of internal search-tool consumers and registration boundaries."""

from dataclasses import dataclass


INTERNAL_SEARCH_TOOL_NAME = "internal_search_memory_records"


@dataclass(frozen=True)
class InternalSearchConsumer:
    """A source-level consumer contract for the internal search tool."""

    name: str
    category: str
    files: tuple[str, ...]
    symbols: tuple[str, ...]
    contract: str


INTERNAL_SEARCH_CONSUMERS = (
    InternalSearchConsumer(
        name="tool_registration",
        category="registration",
        files=("src/mcp_memory/mcp/internal_tools.py", "src/mcp_memory/mcp/transport.py"),
        symbols=("get_internal_maintenance_tools", "internal_tool_services"),
        contract="register the tool schema and resolve it to the internal async service",
    ),
    InternalSearchConsumer(
        name="call_tracking",
        category="tracking",
        files=("src/mcp_memory/internal_tool_call_tracking.py",),
        symbols=("_READ_ONLY_INTERNAL_TOOL_NAMES",),
        contract="classify the tool as read-only for maintenance telemetry",
    ),
    InternalSearchConsumer(
        name="provider_allowlist",
        category="provider",
        files=("src/mcp_memory/core/providers/copilot_sdk.py",),
        symbols=("_INTERNAL_TOOL_NAMES",),
        contract="allow the provider to expose the internal tool",
    ),
    InternalSearchConsumer(
        name="task_handlers",
        category="prompt_and_tools",
        files=(
            "src/mcp_memory/core/task_handlers/ingest.py",
            "src/mcp_memory/core/task_handlers/taxonomist_support.py",
            "src/mcp_memory/core/task_handlers/curator_support.py",
            "src/mcp_memory/core/task_handlers/deduplicator_handlers.py",
            "src/mcp_memory/core/task_handlers/relationship_proposals.py",
        ),
        symbols=(
            "INGEST_ALLOWED_TOOLS",
            "TAXONOMIST_ALLOWED_TOOLS",
            "CURATOR_ALLOWED_TOOLS",
            "DEDUPLICATOR_ALLOWED_TOOLS",
            "RELATIONSHIP_ALLOWED_TOOLS",
        ),
        contract="prompt or allow the tool for ingest, taxonomy, curation, deduplication, and relationship work",
    ),
    InternalSearchConsumer(
        name="analytics_and_cli",
        category="consumer_verification",
        files=("tests/small/test_retrieval_analytics.py", "tests/small/test_cli_taxonomy.py"),
        symbols=("internal_search_memory_records_service", "fake_search_memory_records_service"),
        contract="exercise internal search through analytics and taxonomy CLI paths",
    ),
)


INTERNAL_SEARCH_CONSUMER_NAMES = frozenset(consumer.name for consumer in INTERNAL_SEARCH_CONSUMERS)

