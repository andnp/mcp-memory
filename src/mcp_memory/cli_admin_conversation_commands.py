from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import click


type AdminConversationListAction = Callable[[str | None, str, str | None, str | None, str | None, int, bool], None]
type AdminConversationShowAction = Callable[[str, str | None, str, str | None, bool], None]


@dataclass(frozen=True)
class AdminConversationCommandFamily:
    group: click.Group
    list_command: click.Command
    show_command: click.Command


def build_admin_conversation_command_family(
    workspace_root_option,
    *,
    list_conversations: AdminConversationListAction,
    show_conversation: AdminConversationShowAction,
) -> AdminConversationCommandFamily:
    @click.group(name="conversation")
    def admin_conversation_group() -> None:
        """Canonical AI conversation operator commands."""

    @admin_conversation_group.command(name="list")
    @workspace_root_option
    @click.option(
        "--scope",
        type=click.Choice(["global", "workspace"]),
        default="global",
        show_default=True,
        help="Read conversations across the shared runtime or only the active workspace context.",
    )
    @click.option("--workspace-id", help="Explicit workspace ID override for conversation filtering")
    @click.option("--task-name", help="Filter by task name")
    @click.option("--status", help="Filter by conversation status")
    @click.option("--limit", default=20, show_default=True, type=int, help="Maximum number of conversation rows to print")
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of a table")
    def list_conversations_command(
        workspace_root: str | None,
        scope: str,
        workspace_id: str | None,
        task_name: str | None,
        status: str | None,
        limit: int,
        json_output: bool,
    ) -> None:
        """List recorded AI conversations."""
        list_conversations(workspace_root, scope, workspace_id, task_name, status, limit, json_output)

    @admin_conversation_group.command(name="show")
    @workspace_root_option
    @click.argument("request_id")
    @click.option(
        "--scope",
        type=click.Choice(["global", "workspace"]),
        default="global",
        show_default=True,
        help="Read matching conversation attempts across the shared runtime or only the active workspace context.",
    )
    @click.option("--workspace-id", help="Explicit workspace ID override for conversation filtering")
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of human-readable output")
    def show_conversation_command(
        request_id: str,
        workspace_root: str | None,
        scope: str,
        workspace_id: str | None,
        json_output: bool,
    ) -> None:
        """Show all recorded attempts for one AI request ID."""
        show_conversation(request_id, workspace_root, scope, workspace_id, json_output)

    return AdminConversationCommandFamily(
        group=admin_conversation_group,
        list_command=list_conversations_command,
        show_command=show_conversation_command,
    )