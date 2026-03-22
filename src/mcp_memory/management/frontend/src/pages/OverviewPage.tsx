import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';

import { CommandBar } from '../components/CommandBar';
import { PulseOverview } from '../components/PulseOverview';
import { fetchOverview } from '../lib/api';

function formatTimestamp(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleString();
}

function formatAge(seconds: number): string {
  if (seconds < 60) {
    return `${Math.floor(seconds)}s`;
  }
  if (seconds < 3600) {
    return `${Math.floor(seconds / 60)}m`;
  }
  if (seconds < 86400) {
    return `${Math.floor(seconds / 3600)}h`;
  }
  return `${Math.floor(seconds / 86400)}d`;
}

function statusTone(status: string): string {
  switch (status) {
    case 'failed':
      return 'text-danger';
    case 'completed':
      return 'text-success';
    case 'pending':
    case 'scheduled':
      return 'text-amber';
    default:
      return 'text-text';
  }
}

function queueTimingText(readyInSeconds: number, overdueSeconds: number): string {
  if (overdueSeconds > 0) {
    return `overdue ${formatAge(overdueSeconds)}`;
  }
  return `ready in ${formatAge(readyInSeconds)}`;
}

function providerReasonSummary(usage: {
  top_failure_reason_last_day: string | null;
  top_skip_reason_last_day: string | null;
  active_admission_reason: string | null;
}): string {
  if (usage.active_admission_reason) {
    return `active ${usage.active_admission_reason}`;
  }
  if (usage.top_skip_reason_last_day) {
    return `skip ${usage.top_skip_reason_last_day}`;
  }
  if (usage.top_failure_reason_last_day) {
    return `fail ${usage.top_failure_reason_last_day}`;
  }
  return 'healthy';
}

export function OverviewPage() {
  const [commandResult, setCommandResult] = useState<string | null>(null);
  const overviewQuery = useQuery({
    queryKey: ['overview'],
    queryFn: fetchOverview,
    refetchInterval: 5000,
  });

  if (overviewQuery.isError) {
    return (
      <section className="panel p-6 text-danger">
        Failed to load the daemon overview. This page expects the management API to be reachable at <code>/api</code>.
      </section>
    );
  }

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-border bg-panel/80 px-3 py-2 text-xs text-muted shadow-panel">
        Status: {overviewQuery.isLoading ? 'Loading pulse…' : 'Pulse online'}
      </div>

      <CommandBar onResult={setCommandResult} />

      {commandResult ? (
        <section className="panel p-3">
          <p className="panel-title">Command Result</p>
          <pre className="mt-2 overflow-x-auto whitespace-pre-wrap break-words text-xs text-text">{commandResult}</pre>
        </section>
      ) : null}

      {overviewQuery.data ? (
        <>
          <PulseOverview overview={overviewQuery.data} />

          <section className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
            <div className="table-shell">
              <div className="border-b border-border px-4 py-4">
                <p className="panel-title">Background agents</p>
                <h2 className="mt-2 text-lg font-semibold text-text">Current pulse</h2>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>Agent</th>
                    <th className="hidden md:table-cell">Running</th>
                    <th className="hidden lg:table-cell">Runs</th>
                    <th className="hidden lg:table-cell">Failures</th>
                    <th>Last status</th>
                    <th>Last result</th>
                  </tr>
                </thead>
                <tbody>
                  {overviewQuery.data.agent_runs.slice(0, 8).map((agent) => (
                    <tr key={agent.task_name}>
                      <td>{agent.task_name}</td>
                      <td className="hidden md:table-cell">{agent.running_count}</td>
                      <td className="hidden lg:table-cell">{agent.total_runs}</td>
                      <td className="hidden lg:table-cell">{agent.failed_runs}</td>
                      <td>{agent.last_status ?? 'never'}</td>
                      <td>{agent.last_result_summary ?? '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="space-y-6">
              <section className="table-shell">
                <div className="border-b border-border px-4 py-4">
                  <p className="panel-title">Recent runs</p>
                  <h2 className="mt-2 text-lg font-semibold text-text">Agent activity</h2>
                </div>
                <table>
                  <thead>
                    <tr>
                      <th>Agent</th>
                      <th>Status</th>
                      <th>Completed</th>
                      <th>Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {overviewQuery.data.recent_agent_runs.slice(0, 6).map((run, index) => (
                      <tr key={`${run.task_name}-${run.completed_at}-${index}`}>
                        <td>{run.task_name}</td>
                        <td className={statusTone(run.status)}>{run.status}</td>
                        <td>{formatTimestamp(run.completed_at)}</td>
                        <td>{run.result_summary ?? run.error_text ?? '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>

              <section className="panel p-5">
                <p className="panel-title">Focused pages</p>
                <ul className="mt-3 space-y-2 text-sm leading-6 text-muted">
                  <li>• Use <Link className="text-accent" to="/search">Search</Link> for memory lookup and drill-down.</li>
                  <li>• Use <Link className="text-accent" to="/activity">Activity</Link> for audit context and conversations.</li>
                  <li>• Use <Link className="text-accent" to="/logs">Logs</Link> for filtered daemon diagnostics.</li>
                </ul>
              </section>
            </div>
          </section>

          <section className="grid gap-6 xl:grid-cols-2">
            <div className="table-shell">
              <div className="border-b border-border px-4 py-4">
                <p className="panel-title">Recent memories</p>
                <h2 className="mt-2 text-lg font-semibold text-text">Latest memory movement</h2>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>Title</th>
                    <th className="hidden md:table-cell">Type</th>
                    <th>Status</th>
                    <th className="hidden lg:table-cell">Updated</th>
                  </tr>
                </thead>
                <tbody>
                  {overviewQuery.data.recent_memories.slice(0, 6).map((record) => (
                    <tr key={record.id}>
                      <td>
                        <Link className="font-medium text-accent" to={`/memory/${record.id}`}>{record.title}</Link>
                        {record.summary ? <div className="mt-1 max-w-[26rem] truncate text-[11px] text-muted" title={record.summary}>{record.summary}</div> : null}
                        {record.tags.length ? <div className="mt-1 max-w-[26rem] truncate text-[10px] text-muted" title={record.tags.join(', ')}>{record.tags.join(', ')}</div> : null}
                      </td>
                      <td className="hidden md:table-cell">{record.type}</td>
                      <td className={statusTone(record.status)}>{record.status}</td>
                      <td className="hidden lg:table-cell">{record.updated_at}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="table-shell">
              <div className="border-b border-border px-4 py-4">
                <p className="panel-title">Provider usage</p>
                <h2 className="mt-2 text-lg font-semibold text-text">Who burned the tokens</h2>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>Task</th>
                    <th>Provider</th>
                    <th>Calls 24h</th>
                    <th className="hidden md:table-cell">Failures / skips</th>
                  </tr>
                </thead>
                <tbody>
                  {overviewQuery.data.provider_usage.slice(0, 8).map((usage, index) => (
                    <tr key={`${usage.provider_key}-${usage.task_name ?? 'global'}-${index}`}>
                      <td>{usage.task_name ?? '-'}</td>
                      <td>
                        <div className="truncate" title={usage.provider_key}>{usage.provider_key}</div>
                        <div className="mt-1 truncate text-[10px] text-muted" title={usage.model_name}>{usage.model_name}</div>
                        <div className="mt-1 truncate text-[10px] text-muted" title={providerReasonSummary(usage)}>{providerReasonSummary(usage)}</div>
                      </td>
                      <td>{usage.calls_last_day}</td>
                      <td className="hidden md:table-cell">{usage.failures_last_day} / {usage.skips_last_day}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
            <div className="table-shell">
              <div className="border-b border-border px-4 py-4">
                <p className="panel-title">Failed tasks</p>
                <h2 className="mt-2 text-lg font-semibold text-text">Needs operator attention</h2>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>Task</th>
                    <th>Retries</th>
                    <th>Error</th>
                  </tr>
                </thead>
                <tbody>
                  {overviewQuery.data.failed_tasks.length ? overviewQuery.data.failed_tasks.slice(0, 6).map((task) => (
                    <tr key={task.id}>
                      <td>
                        <div>{task.task_name}</div>
                        <div className={`mt-1 text-xs ${statusTone(task.status)}`}>{task.status}</div>
                      </td>
                      <td>{task.retries_count}/{task.max_retries}</td>
                      <td className="max-w-[20rem] truncate" title={task.last_error ?? ''}>{task.last_error ?? '-'}</td>
                    </tr>
                  )) : (
                    <tr>
                      <td colSpan={3} className="text-muted">No failed tasks. A rare and beautiful sight.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            <div className="table-shell">
              <div className="border-b border-border px-4 py-4">
                <p className="panel-title">Recent logs</p>
                <h2 className="mt-2 text-lg font-semibold text-text">Daemon chatter</h2>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Level</th>
                    <th className="hidden md:table-cell">Source</th>
                    <th>Message</th>
                  </tr>
                </thead>
                <tbody>
                  {overviewQuery.data.recent_logs.slice(0, 8).map((entry) => (
                    <tr key={entry.id}>
                      <td>{formatTimestamp(entry.created_at)}</td>
                      <td className={statusTone(entry.level.toLowerCase() === 'error' ? 'failed' : entry.level.toLowerCase())}>{entry.level}</td>
                      <td className="hidden md:table-cell">
                        <div>{entry.source}</div>
                        <div className="mt-1 text-xs text-muted">{entry.logger_name}</div>
                      </td>
                      <td className="max-w-[28rem] truncate" title={entry.message}>{entry.message}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="grid gap-6 xl:grid-cols-2">
            <div className="table-shell">
              <div className="border-b border-border px-4 py-4">
                <p className="panel-title">Queue diagnostics</p>
                <h2 className="mt-2 text-lg font-semibold text-text">What is waiting next</h2>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>Task</th>
                    <th>State</th>
                    <th className="hidden md:table-cell">Priority</th>
                    <th>Timing</th>
                  </tr>
                </thead>
                <tbody>
                  {overviewQuery.data.queue_diagnostics.length ? overviewQuery.data.queue_diagnostics.slice(0, 8).map((task, index) => (
                    <tr key={`${task.task_name}-${task.workspace_id ?? 'global'}-${index}`}>
                      <td>
                        <div className="truncate" title={task.task_name}>{task.task_name}</div>
                        <div className="mt-1 truncate text-[10px] text-muted" title={`${task.trigger ?? '-'} · ${task.workspace_id ?? 'global'}`}>{task.trigger ?? '-'} · {task.workspace_id ?? 'global'}</div>
                      </td>
                      <td className={statusTone(task.pending_state)}>{task.pending_state}</td>
                      <td className="hidden md:table-cell">{task.priority}</td>
                      <td>{queueTimingText(task.ready_in_seconds, task.overdue_seconds)}</td>
                    </tr>
                  )) : (
                    <tr>
                      <td colSpan={4} className="text-muted">No queued tasks.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            <div className="table-shell">
              <div className="border-b border-border px-4 py-4">
                <p className="panel-title">Active top read memories</p>
                <h2 className="mt-2 text-lg font-semibold text-text">What the system keeps revisiting right now</h2>
                <p className="mt-2 text-sm text-muted">Active memories only, so archived lineage parents stay out of the limelight.</p>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>Reads</th>
                    <th className="hidden md:table-cell">Type</th>
                    <th>Title</th>
                  </tr>
                </thead>
                <tbody>
                  {overviewQuery.data.top_read_memories_active.length ? overviewQuery.data.top_read_memories_active.slice(0, 8).map((record) => (
                    <tr key={record.id}>
                      <td>{record.read_count}</td>
                      <td className="hidden md:table-cell">{record.type}</td>
                      <td><Link className="text-accent" to={`/memory/${record.id}`}>{record.title}</Link></td>
                    </tr>
                  )) : (
                    <tr>
                      <td colSpan={3} className="text-muted">No read hotspots yet.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>
        </>
      ) : null}
    </div>
  );
}