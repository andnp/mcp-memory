"""Verify the inventory of internal search-tool contracts."""

from mcp_memory.mcp.internal_search_contract import (
    INTERNAL_SEARCH_CONSUMER_NAMES,
    INTERNAL_SEARCH_CONSUMERS,
    INTERNAL_SEARCH_TOOL_NAME,
)
from mcp_memory.mcp.internal_tools import get_internal_maintenance_tools
from mcp_memory.mcp.transport import internal_tool_services


def test_internal_search_inventory_covers_verified_consumer_boundaries() -> None:
    """The inventory names every verified registration and consumer boundary."""
    assert {
        "tool_registration",
        "call_tracking",
        "provider_allowlist",
        "task_handlers",
        "analytics_and_cli",
    } == INTERNAL_SEARCH_CONSUMER_NAMES
    assert all(consumer.files and consumer.symbols and consumer.contract for consumer in INTERNAL_SEARCH_CONSUMERS)


def test_internal_search_inventory_matches_live_registration_boundaries() -> None:
    """The registered schema and dispatcher retain the inventoried tool name."""
    tools = get_internal_maintenance_tools()
    services = internal_tool_services()

    assert any(tool.name == INTERNAL_SEARCH_TOOL_NAME for tool in tools)
    assert INTERNAL_SEARCH_TOOL_NAME in services

