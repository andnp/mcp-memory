from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import click

# ---------------------------------------------------------------------------
# Agent commands
# ---------------------------------------------------------------------------

type AdminAgentRunAction = Callable[[str | None, str | None, bool, bool], None]


@dataclass(frozen=True)
class AdminAgentCommandFamily:
    group: click.Group
    run_command: click.Command


def build_admin_agent_command_family(
    workspace_root_option,
    *,
    agent_names: tuple[str, ...],
    run_agent: AdminAgentRunAction,
) -> AdminAgentCommandFamily:
    @click.group(name="agent")
    def admin_agent_group() -> None:
        """Canonical background agent operator commands."""

    @admin_agent_group.command(name="run")
    @workspace_root_option
    @click.argument("agent_name", required=False, type=click.Choice(agent_names))
    @click.option("--all", "run_all", is_flag=True, help="Trigger all background agents")
    @click.option("--force", is_flag=True, help="Enqueue new tasks even if matching tasks are already open")
    def admin_run_agent_command(
        agent_name: str | None,
        workspace_root: str | None,
        run_all: bool,
        force: bool,
    ) -> None:
        """Trigger one or all background agents for the active workspace."""
        run_agent(agent_name, workspace_root, run_all, force)

    return AdminAgentCommandFamily(
        group=admin_agent_group,
        run_command=admin_run_agent_command,
    )


# ---------------------------------------------------------------------------
# Conversation commands
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dashboard commands
# ---------------------------------------------------------------------------

type AdminDashboardOpenAction = Callable[[str | None], None]
type AdminDashboardBuildAction = Callable[[], None]


@dataclass(frozen=True)
class AdminDashboardCommandFamily:
    group: click.Group
    open_command: click.Command
    build_command: click.Command


def build_admin_dashboard_command_family(
    workspace_root_option,
    *,
    open_dashboard: AdminDashboardOpenAction,
    build_dashboard_frontend: AdminDashboardBuildAction,
) -> AdminDashboardCommandFamily:
    @click.group(name="dashboard")
    def admin_dashboard_group() -> None:
        """Canonical operator dashboard commands."""

    @admin_dashboard_group.command(name="open")
    @workspace_root_option
    def admin_dashboard_open_command(workspace_root: str | None) -> None:
        """Ensure the daemon is running and open the operator dashboard."""
        open_dashboard(workspace_root)

    @admin_dashboard_group.command(name="build")
    def admin_dashboard_build_command() -> None:
        """Build the operator dashboard frontend bundle."""
        build_dashboard_frontend()

    return AdminDashboardCommandFamily(
        group=admin_dashboard_group,
        open_command=admin_dashboard_open_command,
        build_command=admin_dashboard_build_command,
    )


# ---------------------------------------------------------------------------
# Log commands
# ---------------------------------------------------------------------------

type AdminLogListAction = Callable[
    [
        str | None,
        str,
        str | None,
        int,
        str | None,
        str | None,
        str | None,
        str | None,
        float | None,
        float | None,
        bool,
    ],
    None,
]
type AdminLogSummaryAction = Callable[
    [
        str | None,
        str,
        str | None,
        str | None,
        str | None,
        str | None,
        str | None,
        float | None,
        float | None,
        bool,
    ],
    None,
]
type AdminLogPruneAction = Callable[[str | None, str, str | None, int | None, int | None, bool], None]


@dataclass(frozen=True)
class AdminLogCommandFamily:
    group: click.Group
    list_command: click.Command
    summary_command: click.Command
    prune_command: click.Command


def build_admin_log_command_family(
    workspace_root_option,
    *,
    show_logs: AdminLogListAction,
    summarize_logs: AdminLogSummaryAction,
    prune_logs: AdminLogPruneAction,
) -> AdminLogCommandFamily:
    @click.group(name="log")
    def admin_log_group() -> None:
        """Canonical structured runtime log commands."""

    @admin_log_group.command(name="list")
    @workspace_root_option
    @click.option(
        "--scope",
        type=click.Choice(["global", "workspace"]),
        default="global",
        show_default=True,
        help="Read logs across the shared runtime or only the active workspace context.",
    )
    @click.option("--workspace-id", help="Explicit workspace ID override for log filtering")
    @click.option("--limit", default=20, show_default=True, type=int, help="Maximum number of log rows to print")
    @click.option(
        "--level",
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
        help="Filter by log level.",
    )
    @click.option("--logger", "logger_name", help="Filter by logger name")
    @click.option("--source", help="Filter by log source (for example: daemon, stdio)")
    @click.option("--query", help="Case-insensitive text filter across message, logger, and source")
    @click.option("--after", type=float, help="Only include logs at or after this UNIX timestamp")
    @click.option("--before", type=float, help="Only include logs at or before this UNIX timestamp")
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of a table")
    def admin_logs(
        workspace_root: str | None,
        scope: str,
        workspace_id: str | None,
        limit: int,
        level: str | None,
        logger_name: str | None,
        source: str | None,
        query: str | None,
        after: float | None,
        before: float | None,
        json_output: bool,
    ) -> None:
        """Print recent structured runtime logs from SQLite."""
        show_logs(workspace_root, scope, workspace_id, limit, level, logger_name, source, query, after, before, json_output)

    @admin_log_group.command(name="summary")
    @workspace_root_option
    @click.option(
        "--scope",
        type=click.Choice(["global", "workspace"]),
        default="global",
        show_default=True,
        help="Summarize logs across the shared runtime or only the active workspace context.",
    )
    @click.option("--workspace-id", help="Explicit workspace ID override for log filtering")
    @click.option(
        "--level",
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
        help="Filter by log level.",
    )
    @click.option("--logger", "logger_name", help="Filter by logger name")
    @click.option("--source", help="Filter by log source (for example: daemon, stdio)")
    @click.option("--query", help="Case-insensitive text filter across message, logger, and source")
    @click.option("--after", type=float, help="Only include logs at or after this UNIX timestamp")
    @click.option("--before", type=float, help="Only include logs at or before this UNIX timestamp")
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of tables")
    def admin_log_summary(
        workspace_root: str | None,
        scope: str,
        workspace_id: str | None,
        level: str | None,
        logger_name: str | None,
        source: str | None,
        query: str | None,
        after: float | None,
        before: float | None,
        json_output: bool,
    ) -> None:
        """Print aggregated runtime log counts."""
        summarize_logs(workspace_root, scope, workspace_id, level, logger_name, source, query, after, before, json_output)

    @admin_log_group.command(name="prune")
    @workspace_root_option
    @click.option(
        "--scope",
        type=click.Choice(["global", "workspace"]),
        default="global",
        show_default=True,
        help="Prune logs across the shared runtime or only the active workspace context.",
    )
    @click.option("--workspace-id", help="Explicit workspace ID override for log pruning")
    @click.option("--max-runtime-logs", type=int, help="Keep at most this many recent runtime logs")
    @click.option("--max-log-age-days", type=int, help="Delete runtime logs older than this many days")
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of human-readable output")
    def admin_log_prune(
        workspace_root: str | None,
        scope: str,
        workspace_id: str | None,
        max_runtime_logs: int | None,
        max_log_age_days: int | None,
        json_output: bool,
    ) -> None:
        """Prune runtime logs using explicit or configured retention limits."""
        prune_logs(workspace_root, scope, workspace_id, max_runtime_logs, max_log_age_days, json_output)

    return AdminLogCommandFamily(
        group=admin_log_group,
        list_command=admin_logs,
        summary_command=admin_log_summary,
        prune_command=admin_log_prune,
    )


# ---------------------------------------------------------------------------
# Operator commands
# ---------------------------------------------------------------------------

type AdminOverviewAction = Callable[[str | None, bool, float, bool], None]
type AdminHealthAction = Callable[[str | None, bool], None]
type AdminMonitorAction = Callable[[str | None, float], None]
type AdminQualityCleanupAction = Callable[[str | None, int, int, int, bool], None]


@dataclass(frozen=True)
class AdminOperatorCommandCluster:
    overview_command: click.Command
    health_command: click.Command
    monitor_command: click.Command
    quality_cleanup_command: click.Command


def build_admin_operator_command_cluster(
    workspace_root_option,
    *,
    show_overview: AdminOverviewAction,
    show_health: AdminHealthAction,
    open_monitor: AdminMonitorAction,
    show_quality_cleanup: AdminQualityCleanupAction,
) -> AdminOperatorCommandCluster:
    @click.command(name="overview")
    @workspace_root_option
    @click.option("--watch", is_flag=True, help="Refresh the stats view continuously")
    @click.option(
        "--interval",
        default=2.0,
        show_default=True,
        type=click.FloatRange(min=0.1),
        help="Seconds between watch refreshes",
    )
    @click.option("--verbose", is_flag=True, help="Show detailed recent agent status lines")
    def admin_overview_command(workspace_root: str | None, watch: bool, interval: float, verbose: bool) -> None:
        """Print background task and memory statistics."""
        show_overview(workspace_root, watch, interval, verbose)

    @click.command(name="health")
    @workspace_root_option
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of human-readable output")
    def admin_health_command(workspace_root: str | None, json_output: bool) -> None:
        """Print an AI-friendly operator health snapshot."""
        show_health(workspace_root, json_output)

    @click.command(name="monitor")
    @workspace_root_option
    @click.option(
        "--interval",
        default=2.0,
        show_default=True,
        type=click.FloatRange(min=0.1),
        help="Seconds between automatic refreshes",
    )
    def admin_monitor_command(workspace_root: str | None, interval: float) -> None:
        """Open the live operations TUI."""
        open_monitor(workspace_root, interval)

    @click.command(name="quality-cleanup")
    @workspace_root_option
    @click.option("--window-hours", default=24, show_default=True, type=int, help="Analytics window to inspect")
    @click.option("--bucket-minutes", default=60, show_default=True, type=int, help="Bucket size used for supporting analytics")
    @click.option("--limit", default=20, show_default=True, type=int, help="Maximum number of cleanup candidates to show")
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of a table")
    def admin_quality_cleanup_command(
        workspace_root: str | None,
        window_hours: int,
        bucket_minutes: int,
        limit: int,
        json_output: bool,
    ) -> None:
        """Show prioritized memory-quality cleanup candidates."""
        show_quality_cleanup(workspace_root, window_hours, bucket_minutes, limit, json_output)

    return AdminOperatorCommandCluster(
        overview_command=admin_overview_command,
        health_command=admin_health_command,
        monitor_command=admin_monitor_command,
        quality_cleanup_command=admin_quality_cleanup_command,
    )


# ---------------------------------------------------------------------------
# Search commands
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Task commands
# ---------------------------------------------------------------------------

type AdminTaskListAction = Callable[[str | None, str | None, int, bool], None]
type AdminTaskRecentRunsAction = Callable[[str | None, int, bool], None]
type AdminTaskSamplingSummaryAction = Callable[[str | None, int, bool], None]
type AdminTaskCancelAction = Callable[[str, str | None, str, bool], None]
type AdminTaskShowAction = Callable[[str, str | None, bool], None]


@dataclass(frozen=True)
class AdminTaskCommandFamily:
    group: click.Group
    list_command: click.Command
    recent_runs_command: click.Command
    sampling_summary_command: click.Command
    cancel_command: click.Command
    show_command: click.Command


def build_admin_task_command_family(
    workspace_root_option,
    *,
    list_tasks: AdminTaskListAction,
    list_recent_task_runs: AdminTaskRecentRunsAction,
    show_task_sampling_summary: AdminTaskSamplingSummaryAction,
    cancel_task: AdminTaskCancelAction,
    show_task: AdminTaskShowAction,
) -> AdminTaskCommandFamily:
    @click.group(name="task")
    def admin_task_group() -> None:
        """Canonical background task operator commands."""

    @admin_task_group.command(name="list")
    @workspace_root_option
    @click.option("--status", help="Filter by task status")
    @click.option("--limit", default=20, show_default=True, type=int, help="Maximum number of task rows to print")
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of a table")
    def admin_list_tasks_command(workspace_root: str | None, status: str | None, limit: int, json_output: bool) -> None:
        """List queued or running tasks."""
        list_tasks(workspace_root, status, limit, json_output)

    @admin_task_group.command(name="recent-runs")
    @workspace_root_option
    @click.option("--limit", default=20, show_default=True, type=int, help="Maximum number of recent runs to print")
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of a table")
    def admin_recent_task_runs_command(workspace_root: str | None, limit: int, json_output: bool) -> None:
        """Show recent completed background task runs and their sampling metadata."""
        list_recent_task_runs(workspace_root, limit, json_output)

    @admin_task_group.command(name="sampling-summary")
    @workspace_root_option
    @click.option("--limit", default=50, show_default=True, type=int, help="Maximum number of recent runs to summarize")
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of tables")
    def admin_task_sampling_summary_command(workspace_root: str | None, limit: int, json_output: bool) -> None:
        """Summarize recent selection and ingest grouping strategy usage."""
        show_task_sampling_summary(workspace_root, limit, json_output)

    @admin_task_group.command(name="cancel")
    @workspace_root_option
    @click.argument("task_id")
    @click.option("--reason", default="cancelled_by_user", show_default=True, help="Cancellation reason")
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of human-readable output")
    def admin_cancel_task_command(task_id: str, workspace_root: str | None, reason: str, json_output: bool) -> None:
        """Cancel a pending or running task."""
        cancel_task(task_id, workspace_root, reason, json_output)

    @admin_task_group.command(name="show")
    @workspace_root_option
    @click.argument("task_id")
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of human-readable output")
    def admin_show_task_command(task_id: str, workspace_root: str | None, json_output: bool) -> None:
        """Show one task and its persisted run result details."""
        show_task(task_id, workspace_root, json_output)

    return AdminTaskCommandFamily(
        group=admin_task_group,
        list_command=admin_list_tasks_command,
        recent_runs_command=admin_recent_task_runs_command,
        sampling_summary_command=admin_task_sampling_summary_command,
        cancel_command=admin_cancel_task_command,
        show_command=admin_show_task_command,
    )


# ---------------------------------------------------------------------------
# Utility commands
# ---------------------------------------------------------------------------

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
