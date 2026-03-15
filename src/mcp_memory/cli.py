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


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx: click.Context, debug: bool) -> None:
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    configure_cli_logging(debug)


@main.command()
@click.option("--workspace-root", help="Override the active workspace root")
@click.pass_context
def run(ctx: click.Context, workspace_root: str | None) -> None:
    """Run the MCP stdio proxy, auto-starting the workspace daemon when needed."""
    configure_workspace_logging(
        bool(ctx.obj.get("debug", False)),
        workspace_root_override=workspace_root,
        console_output=False,
        source="stdio",
    )
    server = MCPServer(workspace_root=workspace_root)
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        return
    except Exception as exc:
        click.echo(f"mcp-memory: {exc}", err=True)
        sys.exit(1)


@main.command(name="internal-run", hidden=True)
@click.option("--workspace-root", help="Override the active workspace root")
@click.pass_context
def internal_run(ctx: click.Context, workspace_root: str | None) -> None:
    """Run the internal maintenance MCP stdio proxy for trusted tool-using agents."""
    configure_workspace_logging(
        bool(ctx.obj.get("debug", False)),
        workspace_root_override=workspace_root,
        console_output=False,
        source="internal-stdio",
    )
    server = MCPServer(
        workspace_root=workspace_root,
        server_name="mcp-memory-internal",
        tool_path_prefix="/internal/maintenance/tools",
    )
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        return
    except Exception as exc:
        click.echo(f"mcp-memory: {exc}", err=True)
        sys.exit(1)


@main.command()
@click.option("--workspace-root", help="Override the active workspace root")
@click.option("--host", default="127.0.0.1", show_default=True, help="Daemon bind host")
@click.option("--port", default=0, show_default=True, type=int, help="Daemon bind port")
@click.pass_context
def daemon(ctx: click.Context, workspace_root: str | None, host: str, port: int) -> None:
    """Run the workspace daemon backend."""
    configure_workspace_logging(
        bool(ctx.obj.get("debug", False)),
        workspace_root_override=workspace_root,
        console_output=True,
        source="daemon",
    )
    daemon_port = port
    if daemon_port == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            daemon_port = int(sock.getsockname()[1])
    app = create_daemon_app(
        workspace_root_override=workspace_root,
        host=host,
        port=daemon_port,
        enable_idle_shutdown=True,
    )
    uvicorn.run(app, host=host, port=daemon_port, log_level="info")


@main.command(name="daemon-status")
@click.option("--workspace-root", help="Override the active workspace root")
def daemon_status(workspace_root: str | None) -> None:
    """Print the current daemon status for the active workspace."""
    workspace_id, metadata, healthy = inspect_daemon(workspace_root, None)
    if metadata is None:
        console.print(f"[yellow]Daemon not registered[/] for {workspace_id}")
        return
    console.print(f"[bold]Workspace:[/] {workspace_id}")
    console.print(f"[bold]Status:[/] {'running' if healthy else 'stale'}")
    console.print(f"[bold]PID:[/] {metadata.pid}")
    console.print(f"[bold]URL:[/] {metadata.base_url}")
    console.print(f"[bold]Started:[/] {_format_timestamp(metadata.started_at)}")


@main.command(name="daemon-stop")
@click.option("--workspace-root", help="Override the active workspace root")
def daemon_stop(workspace_root: str | None) -> None:
    """Stop the workspace daemon if it is running."""
    try:
        metadata = stop_daemon(workspace_root, None)
    except Exception as exc:
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)
    if metadata is None:
        console.print("[yellow]No daemon metadata found.[/]")
        return
    console.print(f"[green]Daemon stopped:[/] pid={metadata.pid} workspace={metadata.workspace_id}")


@main.command(name="daemon-restart")
@click.option("--workspace-root", help="Override the active workspace root")
def daemon_restart(workspace_root: str | None) -> None:
    """Restart the workspace daemon and print the fresh dashboard URL."""
    try:
        stop_daemon(workspace_root, None)
        metadata = ensure_daemon_started(workspace_root, None)
    except Exception as exc:
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)
    console.print(f"[green]Daemon restarted:[/] {metadata.base_url}/ (pid={metadata.pid})")


@main.command(name="dashboard")
@click.option("--workspace-root", help="Override the active workspace root")
def dashboard(workspace_root: str | None) -> None:
    """Ensure the daemon is running and print the dashboard URL."""
    try:
        metadata = ensure_daemon_started(workspace_root, None)
    except Exception as exc:
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)
    console.print(f"[green]Dashboard ready:[/] {metadata.base_url}/")


@main.command(name="logs")
@click.option("--workspace-root", help="Override the active workspace root")
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
    finally:
        runtime.close()


@main.command(name="log-summary")
@click.option("--workspace-root", help="Override the active workspace root")
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
    finally:
        runtime.close()


@main.command(name="log-prune")
@click.option("--workspace-root", help="Override the active workspace root")
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
    """Forward VS Code hook payloads into the workspace daemon."""
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
@click.option("--workspace-root", help="Override the active workspace root")
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
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)
    finally:
        runtime.close()


@main.group(name="agents")
def agents() -> None:
    """Trigger and inspect background agents."""


@agents.command(name="run")
@click.argument("agent_name", type=click.Choice(TRIGGERABLE_BACKGROUND_TASK_NAMES))
@click.option("--workspace-root", help="Override the active workspace root")
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
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)
    finally:
        runtime.close()


@agents.command(name="run-all")
@click.option("--workspace-root", help="Override the active workspace root")
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
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)
    finally:
        runtime.close()


@main.command(name="stats")
@click.option("--workspace-root", help="Override the active workspace root")
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

        provider_table = Table(title="AI Provider Usage")
        provider_table.add_column("Provider", no_wrap=True)
        provider_table.add_column("Model")
        provider_table.add_column("Calls (1h)", justify="right")
        provider_table.add_column("Calls (24h)", justify="right")
        provider_table.add_column("Failures (1h)", justify="right")
        provider_table.add_column("Failures (24h)", justify="right")
        provider_table.add_column("Avg Duration (1h)", justify="right")
        provider_table.add_column("Avg Duration (24h)", justify="right")
        if not overview.provider_usage:
            provider_table.add_row("-", "-", "0", "0", "0", "0", "0.00s", "0.00s")
        for usage in overview.provider_usage:
            provider_table.add_row(
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
    except Exception as exc:
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)
    finally:
        runtime.close()


@main.command(name="import-markdown")
@click.argument("file_paths", nargs=-1, type=str)
@click.option("--workspace-root", help="Override the active workspace root")
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
            # When --thought is used with multiple workspace IDs, use the first one
            resolved_workspace_id = next(
                (wid.strip() for wid in workspace_ids if wid.strip()), 
                None
            )
        elif runtime.workspace_id is not None:
            resolved_workspace_id = runtime.workspace_id

        if thought:
            # Import as thoughts into the journal buffer
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
            # Import directly to relational storage (existing behavior)
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
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)
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
