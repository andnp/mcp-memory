from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import click


type AdminSearchHealthAction = Callable[[str | None, bool], None]
type AdminSearchRepairAction = Callable[[str | None, bool], None]
type AdminSearchDebugAction = Callable[[str | None, str, int, bool], None]


@dataclass(frozen=True)
class AdminSearchCommandFamily:
    group: click.Group
    health_command: click.Command
    repair_command: click.Command
    debug_command: click.Command


def build_admin_search_command_family(
    workspace_root_option,
    *,
    show_search_health: AdminSearchHealthAction,
    repair_search_index: AdminSearchRepairAction,
    show_search_debug: AdminSearchDebugAction,
) -> AdminSearchCommandFamily:
    @click.group(name="search")
    def admin_search_group() -> None:
        """Canonical semantic search operator commands."""

    @admin_search_group.command(name="health")
    @workspace_root_option
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of a table")
    def search_health_command(workspace_root: str | None, json_output: bool) -> None:
        """Show semantic search health for the current runtime context."""
        show_search_health(workspace_root, json_output)

    @admin_search_group.command(name="repair")
    @workspace_root_option
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of human-readable output")
    def search_repair_command(workspace_root: str | None, json_output: bool) -> None:
        """Rebuild semantic search embeddings for the current model."""
        repair_search_index(workspace_root, json_output)

    @admin_search_group.command(name="debug")
    @workspace_root_option
    @click.argument("query_parts", nargs=-1, required=True)
    @click.option("--limit", default=5, show_default=True, type=int, help="Maximum number of search results to inspect")
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of human-readable output")
    def search_debug_command(
        workspace_root: str | None,
        query_parts: tuple[str, ...],
        limit: int,
        json_output: bool,
    ) -> None:
        """Run one search query and print timing/strategy diagnostics."""
        query = " ".join(part for part in query_parts if part.strip()).strip()
        if not query:
            raise click.UsageError("Provide a non-empty query.")
        show_search_debug(workspace_root, query, limit, json_output)

    return AdminSearchCommandFamily(
        group=admin_search_group,
        health_command=search_health_command,
        repair_command=search_repair_command,
        debug_command=search_debug_command,
    )