from __future__ import annotations

from collections.abc import Callable

import click


type StashCommandAction = Callable[[str | None, tuple[str, ...]], None]
type ImportMarkdownCommandAction = Callable[[tuple[str, ...], str | None, tuple[str, ...], bool], None]


def build_memory_group(
    workspace_root_option,
    stash_command_action: StashCommandAction,
    *,
    import_markdown_command_action: ImportMarkdownCommandAction | None = None,
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

    if import_markdown_command_action is not None:
        @memory_group.command(name="import-markdown")
        @click.argument("file_paths", nargs=-1, type=str)
        @workspace_root_option
        @click.option(
            "--workspace-id",
            "workspace_ids",
            multiple=True,
            help="Attach the imported memory to explicit workspace IDs (defaults to the active workspace)",
        )
        @click.option(
            "--thought",
            is_flag=True,
            help="Import as a thought into the thought buffer instead of directly into storage (for intelligent processing by agents)",
        )
        def import_markdown(
            file_paths: tuple[str, ...],
            workspace_root: str | None,
            workspace_ids: tuple[str, ...],
            thought: bool,
        ) -> None:
            """Import one or more markdown memory files into the relational store or thought buffer."""
            import_markdown_command_action(file_paths, workspace_root, workspace_ids, thought)

    return memory_group
