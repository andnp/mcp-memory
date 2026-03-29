from __future__ import annotations

from collections.abc import Callable

import click


def build_memory_group(
    workspace_root_option,
    stash_command_action: Callable[[str | None, tuple[str, ...]], None],
) -> click.Group:
    @click.group(name="memory")
    def memory_group() -> None:
        """Canonical memory commands."""

    @memory_group.command(name="stash")
    @workspace_root_option
    @click.argument("text", nargs=-1)
    def memory_stash_command(workspace_root: str | None, text: tuple[str, ...]) -> None:
        """Record one raw thought into the System 1 journal."""
        stash_command_action(workspace_root, text)

    return memory_group