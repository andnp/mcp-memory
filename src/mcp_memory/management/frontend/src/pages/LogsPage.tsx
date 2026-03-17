import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';

import { fetchLogs, fetchLogSummary } from '../lib/api';

function formatTimestamp(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleString();
}

export function LogsPage() {
  const [query, setQuery] = useState('');
  const [level, setLevel] = useState('');
  const [source, setSource] = useState('');
  const filters = useMemo(() => ({ q: query || undefined, level: level || undefined, source: source || undefined, limit: 20 }), [level, query, source]);
  const logsQuery = useQuery({ queryKey: ['logs', filters], queryFn: () => fetchLogs(filters), refetchInterval: 5000 });
  const summaryQuery = useQuery({ queryKey: ['log-summary', filters], queryFn: () => fetchLogSummary(filters), refetchInterval: 5000 });

  return (
    <div className="space-y-6">
      <section className="panel p-4">
        <p className="panel-title">Logs</p>
        <h2 className="mt-1 text-lg font-semibold text-text">Filtered daemon diagnostics</h2>
        <div className="mt-3 grid gap-2 md:grid-cols-3">
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
        </div>
        <p className="mt-2 text-xs text-muted">
          matching={summaryQuery.data?.total ?? 0} · levels={Object.entries(summaryQuery.data?.by_level ?? {}).map(([name, count]) => `${name}:${count}`).join(' • ') || 'none'}
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
            {logsQuery.data?.logs.map((entry) => (
              <tr key={entry.id}>
                <td>{formatTimestamp(entry.created_at)}</td>
                <td>{entry.level}</td>
                <td>
                  <div>{entry.source}</div>
                  <div className="mt-1 text-xs text-muted">{entry.logger_name}</div>
                </td>
                <td>{entry.message}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}