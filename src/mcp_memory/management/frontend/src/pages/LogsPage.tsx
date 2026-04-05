import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';

import type { ManagementScope } from '../lib/api';
import { fetchLogs, fetchLogSummary } from '../lib/api';

function formatTimestamp(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleString();
}

export function LogsPage() {
  const [query, setQuery] = useState('');
  const [level, setLevel] = useState('');
  const [source, setSource] = useState('');
  const [scope, setScope] = useState<'global' | 'workspace'>('global');
  const [workspaceId, setWorkspaceId] = useState('');
  const filters = useMemo(
    (): {
      q?: string;
      level?: string;
      source?: string;
      scope?: ManagementScope;
      workspace_id?: string;
      limit: number;
    } => ({
      q: query || undefined,
      level: level || undefined,
      source: source || undefined,
      scope: scope === 'workspace' ? 'workspace' : undefined,
      workspace_id: scope === 'workspace' ? (workspaceId.trim() || undefined) : undefined,
      limit: 20,
    }),
    [level, query, scope, source, workspaceId],
  );
  const logsQuery = useQuery({
    queryKey: ['logs', filters],
    queryFn: () => fetchLogs(filters),
    refetchInterval: 15000,
    refetchOnWindowFocus: false,
  });
  const summaryQuery = useQuery({
    queryKey: ['log-summary', filters],
    queryFn: () => fetchLogSummary(filters),
    refetchInterval: 15000,
    refetchOnWindowFocus: false,
  });
  const logs = logsQuery.data?.logs ?? [];

  let summaryText = 'Loading log summary…';
  if (summaryQuery.isError) {
    summaryText = 'Unable to load log summary.';
  } else if (summaryQuery.data) {
    summaryText = `global by default; add workspace narrowing when you need it. workspace-id override only applies when workspace scope is selected. matching=${summaryQuery.data.total} · levels=${Object.entries(summaryQuery.data.by_level).map(([name, count]) => `${name}:${count}`).join(' • ') || 'none'}`;
  }

  let tableState = 'Loading logs…';
  if (logsQuery.isError) {
    tableState = 'Unable to load logs.';
  } else if (logsQuery.isSuccess && logs.length === 0) {
    tableState = 'No matching logs.';
  }

  return (
    <div className="space-y-6">
      <section className="panel p-4">
        <p className="panel-title">Logs</p>
        <h2 className="mt-1 text-lg font-semibold text-text">Filtered daemon diagnostics</h2>
        <div className="mt-3 grid gap-2 lg:grid-cols-5">
          <input value={query} onChange={(event) => setQuery(event.target.value)} className="rounded-lg border border-border bg-ink px-3 py-2 text-xs text-text outline-none focus:border-accent" placeholder="message or logger" />
          <select value={level} onChange={(event) => setLevel(event.target.value)} className="rounded-lg border border-border bg-ink px-3 py-2 text-xs text-text outline-none focus:border-accent">
            <option value="">all levels</option>
            <option value="DEBUG">DEBUG</option>
            <option value="INFO">INFO</option>
            <option value="WARNING">WARNING</option>
            <option value="ERROR">ERROR</option>
            <option value="CRITICAL">CRITICAL</option>
          </select>
          <input value={source} onChange={(event) => setSource(event.target.value)} className="rounded-lg border border-border bg-ink px-3 py-2 text-xs text-text outline-none focus:border-accent" placeholder="source" />
          <select value={scope} onChange={(event) => setScope(event.target.value as 'global' | 'workspace')} className="rounded-lg border border-border bg-ink px-3 py-2 text-xs text-text outline-none focus:border-accent">
            <option value="global">global logs</option>
            <option value="workspace">current workspace logs</option>
          </select>
          <input value={workspaceId} onChange={(event) => setWorkspaceId(event.target.value)} className="rounded-lg border border-border bg-ink px-3 py-2 text-xs text-text outline-none focus:border-accent" placeholder="optional workspace-id override" />
        </div>
        <p className="mt-2 text-xs text-muted">
          {summaryText}
        </p>
      </section>

      <section className="table-shell">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Level</th>
              <th>Source</th>
              <th>Message</th>
            </tr>
          </thead>
          <tbody>
            {logs.length ? logs.map((entry) => (
              <tr key={entry.id}>
                <td>{formatTimestamp(entry.created_at)}</td>
                <td>{entry.level}</td>
                <td>
                  <div className="truncate" title={entry.source}>{entry.source}</div>
                  <div className="mt-1 truncate text-[10px] text-muted" title={entry.logger_name}>{entry.logger_name}</div>
                </td>
                <td className="max-w-[34rem] truncate" title={entry.message}>{entry.message}</td>
              </tr>
            )) : (
              <tr>
                <td colSpan={4} className="text-muted">{tableState}</td>
              </tr>
            )}
          </tbody>
        </table>
      </section>
    </div>
  );
}