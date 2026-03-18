import { useEffect, useMemo, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import * as Plot from '@observablehq/plot';

import { ApiError, fetchNerdMetrics } from '../lib/api';

function PlotFigure({ chart }: { chart: HTMLElement | SVGElement | null }) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!ref.current || !chart) {
      return;
    }
    ref.current.replaceChildren(chart);
    return () => {
      chart.remove();
    };
  }, [chart]);

  return <div ref={ref} className="overflow-x-auto" />;
}

function formatStatValue(value: number, unit: string | null): string {
  if (unit === 's') {
    return `${value.toFixed(2)}s`;
  }
  if (unit === 'pct') {
    return `${(value * 100).toFixed(1)}%`;
  }
  if (unit === 'B') {
    if (value >= 1024 * 1024) {
      return `${(value / (1024 * 1024)).toFixed(1)} MiB`;
    }
    if (value >= 1024) {
      return `${(value / 1024).toFixed(1)} KiB`;
    }
    return `${Math.round(value)} B`;
  }
  return `${Math.round(value)}${unit ? ` ${unit}` : ''}`;
}

function severityClass(severity: 'info' | 'warning' | 'error'): string {
  if (severity === 'error') {
    return 'border-danger/40 bg-danger/10 text-danger';
  }
  if (severity === 'warning') {
    return 'border-yellow-500/40 bg-yellow-500/10 text-yellow-200';
  }
  return 'border-blue-500/40 bg-blue-500/10 text-blue-200';
}

function KeyValueTable({
  title,
  rows,
}: {
  title: string;
  rows: Array<{ label: string; value: string }>;
}) {
  return (
    <section className="table-shell">
      <div className="border-b border-border px-3 py-2">
        <p className="panel-title">{title}</p>
      </div>
      <table>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label}>
              <td>{row.label}</td>
              <td>{row.value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export function NerdPage() {
  const nerdQuery = useQuery({
    queryKey: ['nerd-metrics'],
    queryFn: () => fetchNerdMetrics({ window_hours: 24, bucket_minutes: 60 }),
    refetchInterval: 5000,
    retry: false,
  });

  const throughputChart = useMemo(() => {
    if (!nerdQuery.data?.agent_throughput.length) {
      return null;
    }
    return Plot.plot({
      height: 240,
      marginLeft: 48,
      style: { background: 'transparent', color: '#c9d1d9' },
      x: { type: 'time', label: 'time' },
      y: { grid: true, label: 'runs' },
      marks: [
        Plot.areaY(nerdQuery.data.agent_throughput, { x: (d) => new Date(d.bucket_start * 1000), y: 'total_runs', fill: '#58a6ff', fillOpacity: 0.25 }),
        Plot.lineY(nerdQuery.data.agent_throughput, { x: (d) => new Date(d.bucket_start * 1000), y: 'total_runs', stroke: '#58a6ff', strokeWidth: 2 }),
        Plot.lineY(nerdQuery.data.agent_throughput, { x: (d) => new Date(d.bucket_start * 1000), y: 'failed_runs', stroke: '#f85149', strokeWidth: 2 }),
      ],
    });
  }, [nerdQuery.data]);

  const latencyChart = useMemo(() => {
    if (!nerdQuery.data?.provider_latency.length) {
      return null;
    }
    return Plot.plot({
      height: 260,
      marginLeft: 56,
      style: { background: 'transparent', color: '#c9d1d9' },
      x: { type: 'time', label: 'time' },
      y: { grid: true, label: 'seconds' },
      color: { legend: true },
      marks: [
        Plot.lineY(nerdQuery.data.provider_latency, {
          x: (d) => new Date(d.bucket_start * 1000),
          y: 'avg_duration_seconds',
          stroke: (d) => `${d.provider_name} avg`,
          strokeWidth: 2,
        }),
        Plot.lineY(nerdQuery.data.provider_latency, {
          x: (d) => new Date(d.bucket_start * 1000),
          y: 'p95_duration_seconds',
          stroke: (d) => `${d.provider_name} p95`,
          strokeWidth: 2,
        }),
      ],
    });
  }, [nerdQuery.data]);

  const linkTypeChart = useMemo(() => {
    if (!nerdQuery.data) {
      return null;
    }
    const linkTypeRows = Object.entries(nerdQuery.data.graph_topology.link_type_counts).map(([type, count]) => ({
      type,
      count,
    }));
    if (!linkTypeRows.length) {
      return null;
    }
    return Plot.plot({
      height: 240,
      marginLeft: 56,
      style: { background: 'transparent', color: '#c9d1d9' },
      x: { label: 'link type' },
      y: { grid: true, label: 'count' },
      marks: [Plot.barY(linkTypeRows, { x: 'type', y: 'count', fill: '#7ee787' })],
    });
  }, [nerdQuery.data]);

  const statusChart = useMemo(() => {
    if (!nerdQuery.data) {
      return null;
    }
    const statusRows = Object.entries(nerdQuery.data.memory_lifecycle.by_status).map(([status, count]) => ({
      status,
      count,
    }));
    if (!statusRows.length) {
      return null;
    }
    return Plot.plot({
      height: 240,
      marginLeft: 56,
      style: { background: 'transparent', color: '#c9d1d9' },
      x: { label: 'status' },
      y: { grid: true, label: 'memories' },
      marks: [Plot.barY(statusRows, { x: 'status', y: 'count', fill: '#d2a8ff' })],
    });
  }, [nerdQuery.data]);

  if (nerdQuery.isPending) {
    return <section className="panel p-4 text-xs text-muted">Loading nerd telemetry…</section>;
  }
  if (nerdQuery.isError || !nerdQuery.data) {
    const message = nerdQuery.error instanceof ApiError && nerdQuery.error.status === 404
      ? 'Unable to load nerd metrics because the running daemon does not know about `/api/metrics/nerd` yet. Restart the daemon, then refresh this page.'
      : 'Unable to load nerd metrics.';
    return <section className="panel p-4 text-xs text-danger">{message}</section>;
  }

  const graphRows = [
    { label: 'Total memories', value: formatStatValue(nerdQuery.data.graph_topology.total_memories, null) },
    { label: 'Total links', value: formatStatValue(nerdQuery.data.graph_topology.total_links, null) },
    { label: 'Average degree', value: nerdQuery.data.graph_topology.average_degree.toFixed(2) },
    { label: 'Orphan rate', value: formatStatValue(nerdQuery.data.graph_topology.orphan_rate, 'pct') },
    { label: 'Graph-supported rate', value: formatStatValue(nerdQuery.data.graph_topology.graph_supported_rate, 'pct') },
  ];

  const lifecycleRows = [
    { label: 'Total content bytes', value: formatStatValue(nerdQuery.data.memory_lifecycle.total_content_bytes, 'B') },
    { label: 'Median memory size', value: formatStatValue(nerdQuery.data.memory_lifecycle.median_content_bytes, 'B') },
    { label: 'Cold memory count', value: formatStatValue(nerdQuery.data.memory_lifecycle.cold_memory_count, null) },
    { label: 'Cold memory rate', value: formatStatValue(nerdQuery.data.memory_lifecycle.cold_memory_rate, 'pct') },
    { label: 'Never surfaced', value: formatStatValue(nerdQuery.data.memory_lifecycle.never_surfaced_count, null) },
    { label: 'Degraded memories', value: formatStatValue(nerdQuery.data.memory_lifecycle.degraded_count, null) },
  ];

  const searchRows = [
    { label: 'Semantic enabled', value: nerdQuery.data.search_quality.semantic_enabled ? 'yes' : 'no' },
    { label: 'Degraded mode', value: nerdQuery.data.search_quality.degraded ? 'yes' : 'no' },
    { label: 'Fallback count', value: formatStatValue(nerdQuery.data.search_quality.fallback_count, null) },
    { label: 'Rebuild count', value: formatStatValue(nerdQuery.data.search_quality.rebuild_count, null) },
    { label: 'Graph-supported rate', value: formatStatValue(nerdQuery.data.search_quality.graph_supported_rate, 'pct') },
    { label: 'Last error', value: nerdQuery.data.search_quality.last_error ?? 'none' },
  ];

  const routeRows = nerdQuery.data.route_audit.map((route) => ({
    task: route.task_name,
    className: route.task_class,
    configured: route.configured_primary_route
      ? [route.configured_primary_route, ...route.configured_fallback_routes].join(' → ')
      : 'deterministic',
    resolved: route.resolved_provider_key
      ? `${route.resolved_provider_key}${route.resolved_model_name ? ` (${route.resolved_model_name})` : ''}`
      : 'none',
    recent: route.recent_provider_key
      ? `${route.recent_provider_key}${route.recent_status ? ` / ${route.recent_status}` : ''}`
      : 'none',
    calls: `${route.recent_success_count}✓ / ${route.recent_failure_count}✗`,
  }));

  return (
    <div className="space-y-4">
      <section className="grid gap-2 sm:grid-cols-2 xl:grid-cols-6">
        {nerdQuery.data.stats.map((stat) => (
          <article key={stat.key} className="metric-card">
            <p className="panel-title">{stat.label}</p>
            <p className="mt-1 text-lg font-semibold text-text">{formatStatValue(stat.value, stat.unit)}</p>
          </article>
        ))}
      </section>

      <section className="panel p-3">
        <p className="panel-title">Alert stream</p>
        <h2 className="mt-1 text-base font-semibold text-text">Threshold-backed operator signals</h2>
        <div className="mt-3 flex flex-wrap gap-2">
          {nerdQuery.data.alerts.length ? (
            nerdQuery.data.alerts.map((alert) => (
              <article key={alert.key} className={`rounded-md border px-3 py-2 text-xs ${severityClass(alert.severity)}`}>
                <p className="font-semibold uppercase tracking-wide">{alert.label}</p>
                <p className="mt-1">{alert.message}</p>
                <p className="mt-1 opacity-80">
                  value {formatStatValue(alert.value, alert.unit)}
                  {alert.threshold !== null ? ` / threshold ${formatStatValue(alert.threshold, alert.unit)}` : ''}
                </p>
              </article>
            ))
          ) : (
            <p className="text-xs text-muted">No active nerd alerts. Either things are healthy or the graph is plotting its next move quietly.</p>
          )}
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-[1fr_1fr]">
        <section className="panel p-3">
          <p className="panel-title">Agent throughput</p>
          <h2 className="mt-1 text-base font-semibold text-text">Runs and failures over time</h2>
          <div className="mt-3">{throughputChart ? <PlotFigure chart={throughputChart} /> : <p className="text-xs text-muted">No throughput data yet.</p>}</div>
        </section>

        <section className="panel p-3">
          <p className="panel-title">Provider latency</p>
          <h2 className="mt-1 text-base font-semibold text-text">Average and p95 latency over time</h2>
          <div className="mt-3">{latencyChart ? <PlotFigure chart={latencyChart} /> : <p className="text-xs text-muted">No provider telemetry yet.</p>}</div>
        </section>
      </section>

      <section className="grid gap-4 xl:grid-cols-[1fr_1fr]">
        <section className="panel p-3">
          <p className="panel-title">Graph intelligence</p>
          <h2 className="mt-1 text-base font-semibold text-text">Link distribution</h2>
          <div className="mt-3">{linkTypeChart ? <PlotFigure chart={linkTypeChart} /> : <p className="text-xs text-muted">No link topology data yet.</p>}</div>
        </section>

        <section className="panel p-3">
          <p className="panel-title">Memory lifecycle</p>
          <h2 className="mt-1 text-base font-semibold text-text">Status mix</h2>
          <div className="mt-3">{statusChart ? <PlotFigure chart={statusChart} /> : <p className="text-xs text-muted">No lifecycle distribution yet.</p>}</div>
        </section>
      </section>

      <section className="grid gap-4 xl:grid-cols-[0.7fr_1.3fr]">
        <section className="panel p-3">
          <p className="panel-title">Queue snapshot</p>
          <div className="mt-2 space-y-1 text-xs text-text">
            <div>Runnable: {nerdQuery.data.queue_snapshot.runnable_count}</div>
            <div>Scheduled: {nerdQuery.data.queue_snapshot.scheduled_count}</div>
            <div>Oldest queued age: {formatStatValue(nerdQuery.data.queue_snapshot.oldest_age_seconds, 's')}</div>
          </div>
        </section>

        <section className="table-shell">
          <div className="border-b border-border px-3 py-2">
            <p className="panel-title">Stats for nerds</p>
            <h2 className="mt-1 text-base font-semibold text-text">Current derived telemetry</h2>
          </div>
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {nerdQuery.data.stats.map((stat) => (
                <tr key={stat.key}>
                  <td>{stat.label}</td>
                  <td>{formatStatValue(stat.value, stat.unit)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </section>

      <section className="grid gap-4 xl:grid-cols-3">
        <KeyValueTable title="Graph topology" rows={graphRows} />
        <KeyValueTable title="Memory lifecycle" rows={lifecycleRows} />
        <KeyValueTable title="Search quality" rows={searchRows} />
      </section>

      <section className="table-shell">
        <div className="border-b border-border px-3 py-2">
          <p className="panel-title">Route audit</p>
          <h2 className="mt-1 text-base font-semibold text-text">Configured vs resolved background-agent routing</h2>
        </div>
        <table>
          <thead>
            <tr>
              <th>Task</th>
              <th>Class</th>
              <th>Configured</th>
              <th>Resolved now</th>
              <th>Recent usage</th>
              <th>24h</th>
            </tr>
          </thead>
          <tbody>
            {routeRows.map((route) => (
              <tr key={route.task}>
                <td>{route.task}</td>
                <td>{route.className}</td>
                <td>{route.configured}</td>
                <td>{route.resolved}</td>
                <td>{route.recent}</td>
                <td>{route.calls}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
