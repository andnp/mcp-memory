from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import click


type AdminInstallAction = Callable[[tuple[str, ...], tuple[str, ...], str, str | None], None]
type AdminPrefetchModelAction = Callable[[str | None], None]
type AdminMigrateSqliteToPostgresAction = Callable[[Path | None, str | None, bool, bool, bool], None]


@dataclass(frozen=True)
class AdminUtilityCommandCluster:
    install_command: click.Command
    prefetch_model_command: click.Command
    migrate_sqlite_to_postgres_command: click.Command


def build_admin_utility_command_cluster(
    workspace_root_option,
    *,
    install_command_action: AdminInstallAction,
    prefetch_model: AdminPrefetchModelAction,
    migrate_sqlite_to_postgres: AdminMigrateSqliteToPostgresAction,
) -> AdminUtilityCommandCluster:
    @click.command(name="install")
    @click.option(
        "--tool",
        "tools",
        multiple=True,
        type=click.Choice(["all", "copilot", "claude", "gemini"]),
        help="Install for selected tool integrations (defaults to all).",
    )
    @click.option(
        "--component",
        "components",
        multiple=True,
        type=click.Choice(["all", "hooks", "mcp"]),
        help="Install only selected integration components (defaults to all).",
    )
    @click.option(
        "--scope",
        type=click.Choice(["workspace", "user"]),
        default="workspace",
        show_default=True,
        help="Target workspace-local or user-level config files where supported.",
    )
    @click.option("--workspace-root", help="Override the target workspace root")
    def install_command(
        tools: tuple[str, ...],
        components: tuple[str, ...],
        scope: str,
        workspace_root: str | None,
    ) -> None:
        """Install hook and MCP integration config for supported tools."""
        install_command_action(tools, components, scope, workspace_root)

    @click.command(name="prefetch-model")
    @workspace_root_option
    def prefetch_model_command(workspace_root: str | None) -> None:
        """Download and cache the configured local embedding model in the foreground."""
        prefetch_model(workspace_root)

    @click.command(name="migrate-sqlite-to-postgres")
    @click.option(
        "--sqlite-path",
        type=click.Path(path_type=Path, dir_okay=False, resolve_path=True),
        help="Source SQLite database path (defaults to the current local memory.db path).",
    )
    @click.option(
        "--postgres-dsn",
        help="Target Postgres DSN (defaults to storage.postgres.dsn from config).",
    )
    @click.option("--dry-run", is_flag=True, help="Read and summarize the source without importing rows.")
    @click.option(
        "--allow-non-empty-target",
        is_flag=True,
        help="Allow importing into a non-empty Postgres target for controlled reruns.",
    )
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of human-readable output")
    def migrate_sqlite_to_postgres_command(
        sqlite_path: Path | None,
        postgres_dsn: str | None,
        dry_run: bool,
        allow_non_empty_target: bool,
        json_output: bool,
    ) -> None:
        """Import the current SQLite memory graph into a Postgres backend."""
        migrate_sqlite_to_postgres(
            sqlite_path,
            postgres_dsn,
            dry_run,
            allow_non_empty_target,
            json_output,
        )

    return AdminUtilityCommandCluster(
        install_command=install_command,
        prefetch_model_command=prefetch_model_command,
        migrate_sqlite_to_postgres_command=migrate_sqlite_to_postgres_command,
    )