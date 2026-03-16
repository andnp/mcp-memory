from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import DataTable, Footer, Header, Static, TabPane, TabbedContent

from mcp_memory.management.service import ManagementService
from mcp_memory.mcp.runtime import create_runtime


def run_monitor_tui(workspace_root: str | None, interval_seconds: float) -> None:
    runtime = create_runtime(workspace_root_override=workspace_root)
    try:
        MemoryMonitorApp(runtime=runtime, interval_seconds=interval_seconds).run()
    finally:
        runtime.close()


class MemoryMonitorApp(App):
    TITLE = "MCP Memory Monitor"
    CSS = """
    Screen {
        layout: vertical;
    }

    #summary {
        height: auto;
        margin: 0 1;
        padding: 1 2;
        border: round $accent;
    }

    TabbedContent {
        height: 1fr;
        margin: 0 1 1 1;
    }

    .pane-scroll {
        height: 1fr;
    }

    DataTable {
        height: auto;
        margin: 1 0;
    }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(self, *, runtime, interval_seconds: float) -> None:
        super().__init__()
        self._runtime = runtime
        self._interval_seconds = interval_seconds

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="summary")
        with TabbedContent(initial="overview"):
            with TabPane("Overview", id="overview"):
                with VerticalScroll(classes="pane-scroll"):
                    yield DataTable(id="memory-metrics")
                    yield DataTable(id="agent-runs")
            with TabPane("Providers", id="providers"):
                with VerticalScroll(classes="pane-scroll"):
                    yield DataTable(id="provider-usage")
            with TabPane("Tasks", id="tasks"):
                with VerticalScroll(classes="pane-scroll"):
                    yield DataTable(id="task-list")
            with TabPane("Logs", id="logs"):
                with VerticalScroll(classes="pane-scroll"):
                    yield DataTable(id="recent-logs")
            with TabPane("Activity", id="activity"):
                with VerticalScroll(classes="pane-scroll"):
                    yield DataTable(id="recent-agent-runs")
                    yield DataTable(id="top-read-memories")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"refresh {self._interval_seconds:.1f}s"
        self._configure_tables()
        self.refresh_snapshot()
        self.set_interval(self._interval_seconds, self.refresh_snapshot)

    def action_refresh(self) -> None:
        self.refresh_snapshot()

    def refresh_snapshot(self) -> None:
        try:
            health, overview, journal_counts, tasks_payload, logs_payload = _fetch_monitor_snapshot(self._runtime)
        except Exception as exc:
            self.query_one("#summary", Static).update(f"[red]Error:[/] {exc}")
            return

        self.query_one("#summary", Static).update(
            "\n".join(
                [
                    f"[bold]Workspace:[/] {health.workspace_id}    [bold]Scope:[/] global",
                    f"[bold]Database:[/] {health.db_path}",
                    (
                        f"[bold]Refreshed:[/] {_format_timestamp(datetime.now().timestamp())}    "
                        f"[bold]Search:[/] {_describe_search_status(health.search)}    "
                        f"[bold]Semantic:[/] {health.search.semantic_enabled}"
                    ),
                ]
            )
        )

        _populate_memory_metrics_table(self.query_one("#memory-metrics", DataTable), overview, journal_counts)
        _populate_agent_runs_table(self.query_one("#agent-runs", DataTable), overview)
        _populate_provider_usage_table(self.query_one("#provider-usage", DataTable), overview)
        _populate_task_list_table(self.query_one("#task-list", DataTable), tasks_payload)
        _populate_recent_logs_table(self.query_one("#recent-logs", DataTable), logs_payload)
        _populate_recent_runs_table(self.query_one("#recent-agent-runs", DataTable), overview)
        _populate_top_reads_table(self.query_one("#top-read-memories", DataTable), overview)

    def _configure_tables(self) -> None:
        memory_metrics = self.query_one("#memory-metrics", DataTable)
        memory_metrics.cursor_type = "row"
        memory_metrics.zebra_stripes = True
        memory_metrics.add_columns("Metric", "Value")

        agent_runs = self.query_one("#agent-runs", DataTable)
        agent_runs.cursor_type = "row"
        agent_runs.zebra_stripes = True
        agent_runs.add_columns("Agent", "Running", "Age", "Next", "Status", "Failures")

        provider_usage = self.query_one("#provider-usage", DataTable)
        provider_usage.cursor_type = "row"
        provider_usage.zebra_stripes = True
        provider_usage.add_columns("Task", "Provider", "Model", "Calls 1h", "Failures 1h", "Avg 1h")

        task_list = self.query_one("#task-list", DataTable)
        task_list.cursor_type = "row"
        task_list.zebra_stripes = True
        task_list.add_columns("Task", "Status", "Workspace", "PID", "Updated", "Error")

        recent_logs = self.query_one("#recent-logs", DataTable)
        recent_logs.cursor_type = "row"
        recent_logs.zebra_stripes = True
        recent_logs.add_columns("Time", "Level", "Source", "Logger", "Message")

        recent_runs = self.query_one("#recent-agent-runs", DataTable)
        recent_runs.cursor_type = "row"
        recent_runs.zebra_stripes = True
        recent_runs.add_columns("Agent", "Status", "Completed", "Duration", "Result")

        top_reads = self.query_one("#top-read-memories", DataTable)
        top_reads.cursor_type = "row"
        top_reads.zebra_stripes = True
        top_reads.add_columns("Reads", "Type", "Title")


def _build_management_service(runtime, workspace_id: str | None | object = ...) -> ManagementService:
    ctx = runtime if workspace_id is ... else replace(runtime, workspace_id=workspace_id)
    return ManagementService(ctx, SimpleNamespace(has_runtime=True, client_count=1))


def _fetch_monitor_snapshot(runtime):
    workspace_service = _build_management_service(runtime)
    global_service = _build_management_service(runtime, workspace_id=None)
    health = workspace_service.get_health()
    overview = global_service.get_overview()
    journal_counts = {} if runtime.journal is None else runtime.journal.count_by_status()
    tasks_payload = global_service.list_tasks(limit=20)
    logs_payload = global_service.list_logs(limit=20)
    return health, overview, journal_counts, tasks_payload, logs_payload


def _describe_search_status(search_health) -> str:
    if not search_health.available:
        return "offline"
    if search_health.degraded:
        return "degraded"
    return "healthy"


def _populate_memory_metrics_table(table: DataTable, overview, journal_counts: dict[str, int]) -> None:
    table.clear(columns=False)
    rows = [
        ("Total memories", str(overview.memory_metrics.total_memories)),
        ("Total memory lines", str(overview.memory_metrics.total_memory_lines)),
        ("Total summary lines", str(overview.memory_metrics.total_summary_lines)),
        ("Lines compressed", str(overview.memory_metrics.total_lines_compressed)),
        ("Pending journal entries", str(journal_counts.get("pending", 0))),
        ("Processed journal entries", str(journal_counts.get("processed", 0))),
        ("Archived journal entries", str(journal_counts.get("archived", 0))),
        ("Pending journal lines", str(overview.memory_metrics.thought_buffer_lines)),
    ]
    for row in rows:
        table.add_row(*row)


def _populate_agent_runs_table(table: DataTable, overview) -> None:
    table.clear(columns=False)
    if not overview.agent_runs:
        table.add_row("-", "0", "never", "never", "never", "0")
        return
    for agent in overview.agent_runs:
        table.add_row(
            agent.task_name,
            str(agent.running_count),
            _format_age(agent.seconds_since_last_completion),
            _format_age(agent.seconds_until_next_run),
            agent.last_status or "never",
            str(agent.failed_runs),
        )


def _populate_provider_usage_table(table: DataTable, overview) -> None:
    table.clear(columns=False)
    if not overview.provider_usage:
        table.add_row("-", "-", "-", "0", "0", "0.00s")
        return
    for usage in overview.provider_usage:
        table.add_row(
            usage.task_name or "-",
            usage.provider_key,
            usage.model_name,
            str(usage.calls_last_hour),
            str(usage.failures_last_hour),
            f"{usage.avg_duration_last_hour:.2f}s",
        )


def _populate_task_list_table(table: DataTable, payload) -> None:
    table.clear(columns=False)
    if not payload.tasks:
        table.add_row("-", "-", "-", "-", "never", "No tasks found")
        return
    for task in payload.tasks:
        table.add_row(
            task["task_name"],
            task["status"],
            task["workspace_id"] or "-",
            "-" if task.get("subprocess_pid") is None else str(task["subprocess_pid"]),
            _format_timestamp(task.get("updated_at")),
            task.get("last_error") or task.get("cancellation_reason") or "-",
        )


def _populate_recent_logs_table(table: DataTable, payload) -> None:
    table.clear(columns=False)
    if not payload.logs:
        table.add_row("never", "-", "-", "-", "No logs found")
        return
    for entry in payload.logs:
        table.add_row(
            _format_timestamp(entry.created_at),
            entry.level,
            entry.source,
            entry.logger_name,
            entry.message,
        )


def _populate_recent_runs_table(table: DataTable, overview) -> None:
    table.clear(columns=False)
    if not overview.recent_agent_runs:
        table.add_row("-", "-", "never", "0.00s", "No recent agent runs")
        return
    for run in overview.recent_agent_runs:
        table.add_row(
            run.task_name,
            run.status,
            _format_timestamp(run.completed_at),
            f"{run.duration_seconds:.2f}s",
            run.result_summary or run.error_text or "-",
        )


def _populate_top_reads_table(table: DataTable, overview) -> None:
    table.clear(columns=False)
    if not overview.top_read_memories:
        table.add_row("0", "-", "No memories have been read yet")
        return
    for record in overview.top_read_memories:
        table.add_row(str(record.read_count), record.type, record.title)


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