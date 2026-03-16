from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime
import json
import socket
import sys
from types import SimpleNamespace

import click
from rich.console import Console
from rich.table import Table
import uvicorn

from mcp_memory.core.agent_runtime import bootstrap_background_tasks
from mcp_memory.core.task_handlers import TRIGGERABLE_BACKGROUND_TASK_NAMES
from mcp_memory.daemon import create_daemon_app, ensure_daemon_started, inspect_daemon, stop_daemon
from mcp_memory.embeddings import describe_embedder
from mcp_memory.installer import install_integrations, load_hook_payload, safe_forward_hook_event
from mcp_memory.management.service import ManagementService
from mcp_memory.mcp.runtime import create_runtime
from mcp_memory.relational.importer import (
    import_markdown_memory_paths,
    record_markdown_memory_paths_as_thoughts,
)
from mcp_memory.runtime_logging import configure_cli_logging, configure_workspace_logging
from mcp_memory.server import MCPServer

console = Console()
workspace_root_option = click.option("--workspace-root", help="Override the active workspace root")


def _exit_cli_error(exc: Exception) -> None:
    console.print(f"[red]Error:[/] {exc}")
    sys.exit(1)


def _run_stdio_proxy(
    debug_enabled: bool,
    workspace_root: str | None,
    source: str,
    server_name: str,
    tool_path_prefix: str,
) -> None:
    configure_workspace_logging(
        debug_enabled,
        workspace_root_override=workspace_root,
        console_output=False,
        source=source,
    )
    server = MCPServer(
        workspace_root=workspace_root,
        server_name=server_name,
        tool_path_prefix=tool_path_prefix,
    )
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        return
    except Exception as exc:
        click.echo(f"mcp-memory: {exc}", err=True)
        sys.exit(1)


def _resolve_daemon_port(host: str, port: int) -> int:
    if port != 0:
        return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _start_daemon(debug_enabled: bool, workspace_root: str | None, host: str, port: int) -> None:
    configure_workspace_logging(
        debug_enabled,
        workspace_root_override=workspace_root,
        console_output=True,
        source="daemon",
    )
    daemon_port = _resolve_daemon_port(host, port)
    app = create_daemon_app(
        workspace_root_override=workspace_root,
        host=host,
        port=daemon_port,
        enable_idle_shutdown=True,
    )
    uvicorn.run(app, host=host, port=daemon_port, log_level="info")


def _print_daemon_status(workspace_root: str | None) -> None:
    workspace_id, metadata, healthy = inspect_daemon(workspace_root, None)
    if metadata is None:
        console.print("[yellow]Global daemon not registered[/]")
        return
    console.print("[bold]Daemon scope:[/] global")
    console.print(f"[bold]Workspace context:[/] {workspace_id}")
    console.print(f"[bold]Status:[/] {'running' if healthy else 'stale'}")
    console.print(f"[bold]PID:[/] {metadata.pid}")
    console.print(f"[bold]URL:[/] {metadata.base_url}")
    console.print(f"[bold]Started:[/] {_format_timestamp(metadata.started_at)}")


def _stop_daemon_command(workspace_root: str | None) -> None:
    try:
        metadata = stop_daemon(workspace_root, None)
    except Exception as exc:
        _exit_cli_error(exc)
    if metadata is None:
        console.print("[yellow]No daemon metadata found.[/]")
        return
    console.print(f"[green]Daemon stopped:[/] pid={metadata.pid} scope=global")


def _restart_daemon_command(workspace_root: str | None) -> None:
    try:
        stop_daemon(workspace_root, None)
        metadata = ensure_daemon_started(workspace_root, None)
    except Exception as exc:
        _exit_cli_error(exc)
    console.print(f"[green]Daemon restarted:[/] {metadata.base_url}/ (pid={metadata.pid})")


def _print_dashboard_url(workspace_root: str | None) -> None:
    try:
        metadata = ensure_daemon_started(workspace_root, None)
    except Exception as exc:
        _exit_cli_error(exc)
    console.print(f"[green]Dashboard ready:[/] {metadata.base_url}/")


def _render_logs_table(payload) -> None:
    table = Table(title="Runtime Logs")
    table.add_column("Time", no_wrap=True)
    table.add_column("Level", no_wrap=True)
    table.add_column("Source", no_wrap=True)
    table.add_column("Logger", no_wrap=True)
    table.add_column("Message")
    if not payload.logs:
        table.add_row("-", "-", "-", "-", "No logs found")
    for entry in payload.logs:
        table.add_row(
            _format_timestamp(entry.created_at),
            entry.level,
            entry.source,
            entry.logger_name,
            entry.message,
        )
    console.print(table)


def _render_log_summary(payload) -> None:
    console.print(f"[bold]Matching logs:[/] {payload.total}")

    by_level = Table(title="By Level")
    by_level.add_column("Level")
    by_level.add_column("Count", justify="right")
    if not payload.by_level:
        by_level.add_row("-", "0")
    for level_name, count in sorted(payload.by_level.items()):
        by_level.add_row(level_name, str(count))
    console.print(by_level)

    by_source = Table(title="By Source")
    by_source.add_column("Source")
    by_source.add_column("Count", justify="right")
    if not payload.by_source:
        by_source.add_row("-", "0")
    for source_name, count in sorted(payload.by_source.items()):
        by_source.add_row(source_name, str(count))
    console.print(by_source)


def _show_logs(
    workspace_root: str | None,
    limit: int,
    level: str | None,
    logger_name: str | None,
    source: str | None,
    query: str | None,
    after: float | None,
    before: float | None,
    json_output: bool,
) -> None:
    runtime = create_runtime(workspace_root_override=workspace_root)
    try:
        payload = _build_management_service(runtime).list_logs(
            level=None if level is None else level.upper(),
            logger_name=logger_name,
            source=source,
            query=query,
            after=after,
            before=before,
            limit=limit,
        )
        if json_output:
            click.echo(json.dumps(payload.model_dump(), sort_keys=True))
            return
        _render_logs_table(payload)
    finally:
        runtime.close()


def _summarize_logs(
    workspace_root: str | None,
    level: str | None,
    logger_name: str | None,
    source: str | None,
    query: str | None,
    after: float | None,
    before: float | None,
    json_output: bool,
) -> None:
    runtime = create_runtime(workspace_root_override=workspace_root)
    try:
        payload = _build_management_service(runtime).summarize_logs(
            level=None if level is None else level.upper(),
            logger_name=logger_name,
            source=source,
            query=query,
            after=after,
            before=before,
        )
        if json_output:
            click.echo(json.dumps(payload.model_dump(), sort_keys=True))
            return
        _render_log_summary(payload)
    finally:
        runtime.close()


def _prune_logs(
    workspace_root: str | None,
    max_runtime_logs: int | None,
    max_log_age_days: int | None,
    json_output: bool,
) -> None:
    runtime = create_runtime(workspace_root_override=workspace_root)
    try:
        payload = _build_management_service(runtime).prune_logs(
            max_runtime_logs=max_runtime_logs,
            max_log_age_days=max_log_age_days,
        )
        if json_output:
            click.echo(json.dumps(payload.model_dump(), sort_keys=True))
            return
        console.print(f"[green]Pruned logs:[/] {payload.deleted}")
        console.print(
            f"policy max_runtime_logs={payload.max_runtime_logs} max_log_age_days={payload.max_log_age_days}"
        )
    finally:
        runtime.close()


def _render_memory_metrics_table(overview, journal_counts: dict[str, int]) -> None:
    metrics = Table(title="Memory Metrics")
    metrics.add_column("Metric")
    metrics.add_column("Value", justify="right")
    metrics.add_row("Total memories", str(overview.memory_metrics.total_memories))
    metrics.add_row("Total memory lines", str(overview.memory_metrics.total_memory_lines))
    metrics.add_row("Total summary lines", str(overview.memory_metrics.total_summary_lines))
    metrics.add_row("Total lines compressed", str(overview.memory_metrics.total_lines_compressed))
    metrics.add_row("Pending journal entries", str(journal_counts.get("pending", 0)))
    metrics.add_row("Processed journal entries", str(journal_counts.get("processed", 0)))
    metrics.add_row("Archived journal entries", str(journal_counts.get("archived", 0)))
    metrics.add_row("Pending journal lines", str(overview.memory_metrics.thought_buffer_lines))
    console.print(metrics)


def _render_agent_table(overview) -> None:
    agent_table = Table(title="Background Agents")
    agent_table.add_column("Agent", no_wrap=True)
    agent_table.add_column("Running", justify="right")
    agent_table.add_column("Age")
    agent_table.add_column("Next")
    agent_table.add_column("Last status")
    agent_table.add_column("Runs", justify="right")
    agent_table.add_column("Failures", justify="right")
    agent_table.add_column("Avg duration", justify="right")
    agent_table.add_column("Compressed", justify="right")
    agent_table.add_column("Last result")
    for agent in overview.agent_runs:
        agent_table.add_row(
            agent.task_name,
            str(agent.running_count),
            _format_age(agent.seconds_since_last_completion),
            _format_age(agent.seconds_until_next_run),
            agent.last_status or "never",
            str(agent.total_runs),
            str(agent.failed_runs),
            f"{agent.avg_duration_seconds:.2f}s",
            str(agent.total_lines_compressed),
            agent.last_result_summary or "-",
        )
    console.print(agent_table)


def _render_provider_usage_table(overview) -> None:
    provider_table = Table(title="AI Provider Usage")
    provider_table.add_column("Task")
    provider_table.add_column("Provider", no_wrap=True)
    provider_table.add_column("Model")
    provider_table.add_column("Calls (1h)", justify="right")
    provider_table.add_column("Calls (24h)", justify="right")
    provider_table.add_column("Failures (1h)", justify="right")
    provider_table.add_column("Failures (24h)", justify="right")
    provider_table.add_column("Avg Duration (1h)", justify="right")
    provider_table.add_column("Avg Duration (24h)", justify="right")
    if not overview.provider_usage:
        provider_table.add_row("-", "-", "-", "0", "0", "0", "0", "0.00s", "0.00s")
    for usage in overview.provider_usage:
        provider_table.add_row(
            usage.task_name or "-",
            usage.provider_key,
            usage.model_name,
            str(usage.calls_last_hour),
            str(usage.calls_last_day),
            str(usage.failures_last_hour),
            str(usage.failures_last_day),
            f"{usage.avg_duration_last_hour:.2f}s",
            f"{usage.avg_duration_last_day:.2f}s",
        )
    console.print(provider_table)


def _render_top_reads_table(overview) -> None:
    top_reads_table = Table(title="Top Read Memories")
    top_reads_table.add_column("Reads", justify="right", no_wrap=True)
    top_reads_table.add_column("Type", no_wrap=True)
    top_reads_table.add_column("Title")
    if not overview.top_read_memories:
        top_reads_table.add_row("0", "-", "No memories have been read yet")
    for record in overview.top_read_memories:
        top_reads_table.add_row(
            str(record.read_count),
            record.type,
            record.title,
        )
    console.print(top_reads_table)


def _print_agent_details(overview) -> None:
    console.print("[bold]Agent Details[/]")
    for agent in overview.agent_runs:
        console.print(
            "- "
            f"{agent.task_name}: "
            f"running={agent.running_count} "
            f"next={_format_age(agent.seconds_until_next_run)} "
            f"last_status={agent.last_status or 'never'} "
            f"last_result={agent.last_result_summary or '-'}"
        )


def _print_recent_agent_runs(overview) -> None:
    console.print("[bold]Recent Agent Runs[/]")
    for run in overview.recent_agent_runs:
        console.print(
            "- "
            f"{run.task_name}: "
            f"status={run.status} "
            f"duration={run.duration_seconds:.2f}s "
            f"result={run.result_summary or '-'} "
            f"error={run.error_text or '-'}"
        )


def _render_task_table(payload) -> None:
    table = Table(title="Tasks")
    table.add_column("Task ID")
    table.add_column("Task")
    table.add_column("Status", no_wrap=True)
    table.add_column("Workspace")
    table.add_column("PID", justify="right")
    table.add_column("Request")
    table.add_column("Updated", no_wrap=True)
    table.add_column("Error")
    if not payload.tasks:
        table.add_row("-", "-", "-", "-", "-", "-", "-", "No tasks found")
    for task in payload.tasks:
        table.add_row(
            task["id"],
            task["task_name"],
            task["status"],
            task["workspace_id"] or "-",
            "-" if task.get("subprocess_pid") is None else str(task["subprocess_pid"]),
            task.get("active_request_id") or "-",
            _format_timestamp(task.get("updated_at")),
            task.get("last_error") or task.get("cancellation_reason") or "-",
        )
    console.print(table)


def _render_ai_conversation_table(payload) -> None:
    table = Table(title="AI Conversations")
    table.add_column("Request")
    table.add_column("Attempt", justify="right")
    table.add_column("Task")
    table.add_column("PID", justify="right")
    table.add_column("Status", no_wrap=True)
    table.add_column("Duration", justify="right")
    table.add_column("Completed", no_wrap=True)
    if not payload.conversations:
        table.add_row("-", "-", "-", "-", "-", "-", "-")
    for conversation in payload.conversations:
        table.add_row(
            conversation.request_id,
            str(conversation.attempt),
            conversation.task_name or "-",
            "-" if conversation.subprocess_pid is None else str(conversation.subprocess_pid),
            conversation.status,
            f"{conversation.duration_seconds:.2f}s",
            _format_timestamp(conversation.completed_at),
        )
    console.print(table)


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx: click.Context, debug: bool) -> None:
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    configure_cli_logging(debug)


@main.command()
@workspace_root_option
@click.pass_context
def run(ctx: click.Context, workspace_root: str | None) -> None:
    """Run the MCP stdio proxy, auto-starting the global daemon when needed."""
    _run_stdio_proxy(
        bool(ctx.obj.get("debug", False)),
        workspace_root,
        "stdio",
        "mcp-memory",
        "/internal/tools",
    )


@main.command(name="internal-run", hidden=True)
@workspace_root_option
@click.pass_context
def internal_run(ctx: click.Context, workspace_root: str | None) -> None:
    """Run the internal maintenance MCP stdio proxy for trusted tool-using agents."""
    _run_stdio_proxy(
        bool(ctx.obj.get("debug", False)),
        workspace_root,
        "internal-stdio",
        "mcp-memory-internal",
        "/internal/maintenance/tools",
    )


@main.group(name="daemon", invoke_without_command=True)
@workspace_root_option
@click.option("--host", default="127.0.0.1", show_default=True, help="Daemon bind host")
@click.option("--port", default=0, show_default=True, type=int, help="Daemon bind port")
@click.pass_context
def daemon_group(ctx: click.Context, workspace_root: str | None, host: str, port: int) -> None:
    """Run and manage the global daemon."""
    if ctx.invoked_subcommand is not None:
        return
    _start_daemon(bool(ctx.obj.get("debug", False)), workspace_root, host, port)


@daemon_group.command(name="status")
@workspace_root_option
def daemon_status(workspace_root: str | None) -> None:
    """Print the current global daemon status."""
    _print_daemon_status(workspace_root)


@daemon_group.command(name="stop")
@workspace_root_option
def daemon_stop(workspace_root: str | None) -> None:
    """Stop the global daemon if it is running."""
    _stop_daemon_command(workspace_root)


@daemon_group.command(name="restart")
@workspace_root_option
def daemon_restart(workspace_root: str | None) -> None:
    """Restart the workspace daemon and print the fresh dashboard URL."""
    _restart_daemon_command(workspace_root)


@daemon_group.command(name="dashboard")
@workspace_root_option
def dashboard(workspace_root: str | None) -> None:
    """Ensure the daemon is running and print the dashboard URL."""
    _print_dashboard_url(workspace_root)


@main.command(name="daemon-status", hidden=True)
@workspace_root_option
def daemon_status_alias(workspace_root: str | None) -> None:
    _print_daemon_status(workspace_root)


@main.command(name="daemon-stop", hidden=True)
@workspace_root_option
def daemon_stop_alias(workspace_root: str | None) -> None:
    _stop_daemon_command(workspace_root)


@main.command(name="daemon-restart", hidden=True)
@workspace_root_option
def daemon_restart_alias(workspace_root: str | None) -> None:
    _restart_daemon_command(workspace_root)


@main.command(name="dashboard", hidden=True)
@workspace_root_option
def dashboard_alias(workspace_root: str | None) -> None:
    _print_dashboard_url(workspace_root)


@main.group(name="log")
def log_group() -> None:
    """Inspect and manage structured runtime logs."""


@log_group.command(name="show")
@workspace_root_option
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
def logs(
    workspace_root: str | None,
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
    _show_logs(workspace_root, limit, level, logger_name, source, query, after, before, json_output)


@log_group.command(name="summary")
@workspace_root_option
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
def log_summary(
    workspace_root: str | None,
    level: str | None,
    logger_name: str | None,
    source: str | None,
    query: str | None,
    after: float | None,
    before: float | None,
    json_output: bool,
) -> None:
    """Print aggregated runtime log counts."""
    _summarize_logs(workspace_root, level, logger_name, source, query, after, before, json_output)


@log_group.command(name="prune")
@workspace_root_option
@click.option("--max-runtime-logs", type=int, help="Keep at most this many recent runtime logs")
@click.option("--max-log-age-days", type=int, help="Delete runtime logs older than this many days")
@click.option("--json", "json_output", is_flag=True, help="Print JSON instead of human-readable output")
def log_prune(
    workspace_root: str | None,
    max_runtime_logs: int | None,
    max_log_age_days: int | None,
    json_output: bool,
) -> None:
    """Prune runtime logs using explicit or configured retention limits."""
    _prune_logs(workspace_root, max_runtime_logs, max_log_age_days, json_output)


@main.command(name="logs", hidden=True)
@workspace_root_option
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
def logs_alias(
    workspace_root: str | None,
    limit: int,
    level: str | None,
    logger_name: str | None,
    source: str | None,
    query: str | None,
    after: float | None,
    before: float | None,
    json_output: bool,
) -> None:
    _show_logs(workspace_root, limit, level, logger_name, source, query, after, before, json_output)


@main.command(name="log-summary", hidden=True)
@workspace_root_option
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
def log_summary_alias(
    workspace_root: str | None,
    level: str | None,
    logger_name: str | None,
    source: str | None,
    query: str | None,
    after: float | None,
    before: float | None,
    json_output: bool,
) -> None:
    _summarize_logs(workspace_root, level, logger_name, source, query, after, before, json_output)


@main.command(name="log-prune", hidden=True)
@workspace_root_option
@click.option("--max-runtime-logs", type=int, help="Keep at most this many recent runtime logs")
@click.option("--max-log-age-days", type=int, help="Delete runtime logs older than this many days")
@click.option("--json", "json_output", is_flag=True, help="Print JSON instead of human-readable output")
def log_prune_alias(
    workspace_root: str | None,
    max_runtime_logs: int | None,
    max_log_age_days: int | None,
    json_output: bool,
) -> None:
    _prune_logs(workspace_root, max_runtime_logs, max_log_age_days, json_output)


@main.command(name="install")
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
def install(tools: tuple[str, ...], components: tuple[str, ...], scope: str, workspace_root: str | None) -> None:
    """Install hook and MCP integration config for supported tools."""
    try:
        result = install_integrations(
            tools=tools,
            components=components,
            scope=scope,
            workspace_root=workspace_root,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)

    console.print(f"[green]Install target:[/] {result.workspace_root}")
    for action in result.actions:
        console.print(
            f"- {action.tool}/{action.component}: {action.status} → {action.path}"
        )


@main.command(name="hook-runner", hidden=True)
@click.option("--workspace-root", help="Override the target workspace root")
def hook_runner(workspace_root: str | None) -> None:
    """Forward VS Code hook payloads into the global daemon."""
    try:
        payload = load_hook_payload(sys.stdin)
    except (json.JSONDecodeError, ValueError) as exc:
        click.echo(f"mcp-memory hook-runner: {exc}", err=True)
        click.echo("{}")
        return

    response, error_message = safe_forward_hook_event(payload, workspace_root=workspace_root)
    if error_message is not None:
        click.echo(f"mcp-memory hook-runner: {error_message}", err=True)
    click.echo(json.dumps(response, sort_keys=True))


@main.command(name="prefetch-model")
@workspace_root_option
def prefetch_model(workspace_root: str | None) -> None:
    """Download and cache the configured local embedding model in the foreground."""
    runtime = create_runtime(workspace_root_override=workspace_root)
    try:
        embedder = runtime.embedder
        if embedder is None:
            raise RuntimeError("embedder_not_initialized")
        cache_model = getattr(embedder, "cache_model", None)
        if not callable(cache_model):
            raise RuntimeError("embedder_cache_not_supported")
        cached = bool(cache_model())
        status = describe_embedder(embedder)
        if not cached:
            raise RuntimeError("embedding_model_cache_failed")
        console.print(f"[green]Embedding model cached:[/] {status.model_name if status is not None else 'unknown'}")
        if status is not None:
            console.print(f"backend={status.backend} cached={status.model_cached}")
    except Exception as exc:
        _exit_cli_error(exc)
    finally:
        runtime.close()


@main.group(name="agents")
def agents() -> None:
    """Trigger and inspect background agents."""


@main.group(name="task")
def task_group() -> None:
    """Inspect and control queued background tasks."""


@task_group.command(name="list")
@workspace_root_option
@click.option("--status", help="Filter by task status")
@click.option("--limit", default=20, show_default=True, type=int, help="Maximum number of task rows to print")
@click.option("--json", "json_output", is_flag=True, help="Print JSON instead of a table")
def list_tasks_command(workspace_root: str | None, status: str | None, limit: int, json_output: bool) -> None:
    """List queued or running tasks."""
    runtime = create_runtime(workspace_root_override=workspace_root)
    try:
        payload = _build_management_service(runtime).list_tasks(
            status=status,
            workspace_id=runtime.workspace_id,
            limit=limit,
        )
        if json_output:
            click.echo(json.dumps(payload.model_dump(), sort_keys=True))
            return
        _render_task_table(payload)
    finally:
        runtime.close()


@task_group.command(name="cancel")
@workspace_root_option
@click.argument("task_id")
@click.option("--reason", default="cancelled_by_user", show_default=True, help="Cancellation reason")
@click.option("--json", "json_output", is_flag=True, help="Print JSON instead of human-readable output")
def cancel_task_command(task_id: str, workspace_root: str | None, reason: str, json_output: bool) -> None:
    """Cancel a pending or running task."""
    runtime = create_runtime(workspace_root_override=workspace_root)
    try:
        payload = _build_management_service(runtime).cancel_task(task_id, cancelled_by="cli", reason=reason)
        if json_output:
            click.echo(json.dumps(payload, sort_keys=True))
            return
        console.print(f"[green]{payload['status']}[/]: {task_id}")
        console.print(f"signal_sent={payload['signal_sent']} status={payload['task']['status']}")
    except Exception as exc:
        _exit_cli_error(exc)
    finally:
        runtime.close()


@main.group(name="conversation")
def conversation_group() -> None:
    """Inspect recorded AI provider conversations."""


@conversation_group.command(name="list")
@workspace_root_option
@click.option("--task-name", help="Filter by task name")
@click.option("--status", help="Filter by conversation status")
@click.option("--limit", default=20, show_default=True, type=int, help="Maximum number of conversation rows to print")
@click.option("--json", "json_output", is_flag=True, help="Print JSON instead of a table")
def list_conversations_command(
    workspace_root: str | None,
    task_name: str | None,
    status: str | None,
    limit: int,
    json_output: bool,
) -> None:
    """List recorded AI conversations."""
    runtime = create_runtime(workspace_root_override=workspace_root)
    try:
        payload = _build_management_service(runtime).list_ai_conversations(
            task_name=task_name,
            status=status,
            limit=limit,
        )
        if json_output:
            click.echo(json.dumps(payload.model_dump(), sort_keys=True))
            return
        _render_ai_conversation_table(payload)
    finally:
        runtime.close()


@conversation_group.command(name="show")
@workspace_root_option
@click.argument("request_id")
@click.option("--json", "json_output", is_flag=True, help="Print JSON instead of human-readable output")
def show_conversation_command(request_id: str, workspace_root: str | None, json_output: bool) -> None:
    """Show all recorded attempts for one AI request ID."""
    runtime = create_runtime(workspace_root_override=workspace_root)
    try:
        payload = _build_management_service(runtime).list_ai_conversations(request_id=request_id, limit=200)
        if json_output:
            click.echo(json.dumps(payload.model_dump(), sort_keys=True))
            return
        _render_ai_conversation_table(payload)
        for conversation in payload.conversations:
            console.print(f"\n[bold]Request {conversation.request_id} attempt {conversation.attempt}[/]")
            console.print(f"status={conversation.status} pid={conversation.subprocess_pid or '-'} task={conversation.task_name or '-'}")
            console.print("[bold]Prompt[/]")
            console.print(conversation.prompt_text or "-")
            console.print("[bold]Response[/]")
            console.print(conversation.response_text or "-")
            if conversation.error_text:
                console.print(f"[red]Error:[/] {conversation.error_text}")
    finally:
        runtime.close()


@agents.command(name="run")
@click.argument("agent_name", type=click.Choice(TRIGGERABLE_BACKGROUND_TASK_NAMES))
@workspace_root_option
@click.option("--force", is_flag=True, help="Enqueue a new task even if one is already open")
def run_agent(agent_name: str, workspace_root: str | None, force: bool) -> None:
    """Trigger one background agent for the active workspace."""
    ensure_daemon_started(workspace_root, None)
    runtime = create_runtime(workspace_root_override=workspace_root)
    try:
        payload = _build_management_service(runtime).enqueue_background_task(agent_name, force=force)
        console.print(f"[green]{payload['status']}[/]: {agent_name}")
        console.print(f"task_id={payload['task']['id']} status={payload['task']['status']}")
    except Exception as exc:
        _exit_cli_error(exc)
    finally:
        runtime.close()


@agents.command(name="run-all")
@workspace_root_option
@click.option("--force", is_flag=True, help="Enqueue new tasks even if matching tasks are already open")
def run_all_agents(workspace_root: str | None, force: bool) -> None:
    """Trigger all background agents for the active workspace."""
    ensure_daemon_started(workspace_root, None)
    runtime = create_runtime(workspace_root_override=workspace_root)
    try:
        service = _build_management_service(runtime)
        results = [service.enqueue_background_task(task_name, force=force) for task_name in TRIGGERABLE_BACKGROUND_TASK_NAMES]
        created_count = sum(1 for result in results if result["created"])
        console.print(f"[green]Agents queued:[/] {created_count}/{len(results)} newly created")
        for result in results:
            console.print(f"- {result['task']['task_name']}: {result['status']}")
    except Exception as exc:
        _exit_cli_error(exc)
    finally:
        runtime.close()


@main.command(name="stats")
@workspace_root_option
def stats(workspace_root: str | None) -> None:
    """Print background task and memory statistics."""
    ensure_daemon_started(workspace_root, None)
    runtime = create_runtime(workspace_root_override=workspace_root)
    try:
        bootstrap_background_tasks(runtime)
        workspace_service = _build_management_service(runtime)
        global_service = _build_management_service(runtime, workspace_id=None)
        health = workspace_service.get_health()
        overview = global_service.get_overview()
        journal_counts = {} if runtime.journal is None else runtime.journal.count_by_status()

        console.print(f"[bold]Workspace:[/] {health.workspace_id}")
        console.print("[bold]Stats scope:[/] global")
        console.print(f"[bold]Database:[/] {health.db_path}")

        _render_memory_metrics_table(overview, journal_counts)
        _render_agent_table(overview)
        _render_provider_usage_table(overview)
        _render_top_reads_table(overview)
        _print_agent_details(overview)
        _print_recent_agent_runs(overview)
    except Exception as exc:
        _exit_cli_error(exc)
    finally:
        runtime.close()


@main.command(name="import-markdown")
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
    if not file_paths:
        raise click.UsageError("Provide at least one markdown file path or glob pattern.")

    runtime = create_runtime(workspace_root_override=workspace_root)
    try:
        resolved_workspace_id = None
        if workspace_ids:
            resolved_workspace_id = next(
                (wid.strip() for wid in workspace_ids if wid.strip()),
                None
            )
        elif runtime.workspace_id is not None:
            resolved_workspace_id = runtime.workspace_id

        if thought:
            if runtime.journal is None:
                raise RuntimeError("journal_not_initialized")

            recorded = record_markdown_memory_paths_as_thoughts(
                runtime.journal,
                list(file_paths),
                workspace_id=resolved_workspace_id,
            )
            for entry_id, file_name in recorded:
                console.print(f"[green]Recorded thought:[/] {file_name} (entry {entry_id})")
            console.print(f"[green]Recorded total:[/] {len(recorded)} thoughts → buffer")
        else:
            if runtime.repository is None:
                raise RuntimeError("repository_not_initialized")

            resolved_workspace_ids = [workspace_id.strip() for workspace_id in workspace_ids if workspace_id.strip()]
            if not resolved_workspace_ids and runtime.workspace_id is not None:
                resolved_workspace_ids = [runtime.workspace_id]

            imported_records = import_markdown_memory_paths(
                runtime.repository,
                list(file_paths),
                resolved_workspace_ids,
            )
            for imported in imported_records:
                console.print(f"[green]Imported memory:[/] {imported.id} — {imported.title}")
            console.print(f"[green]Imported total:[/] {len(imported_records)}")
    except Exception as exc:
        _exit_cli_error(exc)
    finally:
        runtime.close()


def _build_management_service(runtime, workspace_id: str | None | object = ... ) -> ManagementService:
    ctx = runtime if workspace_id is ... else replace(runtime, workspace_id=workspace_id)
    return ManagementService(ctx, SimpleNamespace(has_runtime=True, client_count=1))


def _format_timestamp(value: float | None) -> str:
    if value is None:
        return "never"
    return datetime.fromtimestamp(value).isoformat(timespec="seconds")


def _format_age(value: float | None) -> str:
    if value is None:
        return "never"
    if value < 60:
        return f"{int(value)}s"
    if value < 3600:
        return f"{int(value // 60)}m"
    if value < 86400:
        return f"{int(value // 3600)}h"
    return f"{int(value // 86400)}d"


if __name__ == "__main__":
    main()
