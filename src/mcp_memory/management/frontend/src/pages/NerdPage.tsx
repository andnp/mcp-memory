import { useEffect, useMemo, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import * as Plot from '@observablehq/plot';

import { ApiError, fetchNerdMetrics, type CountBucket } from '../lib/api';

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

function SectionHeading({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div>
      <p className="panel-title">{eyebrow}</p>
      <h2 className="mt-1 text-base font-semibold text-text">{title}</h2>
      <p className="mt-1 text-xs text-muted">{description}</p>
    </div>
  );
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

function formatShare(value: number, total: number): string {
  if (total <= 0) {
    return '0.0%';
  }
  return `${((value / total) * 100).toFixed(1)}%`;
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

function CountBreakdownTable({
  title,
  subtitle,
  rows,
  emptyMessage,
}: {
  title: string;
  subtitle: string;
  rows: CountBucket[];
  emptyMessage: string;
}) {
  const total = rows.reduce((sum, row) => sum + row.count, 0);
  const maxCount = Math.max(...rows.map((row) => row.count), 0);

  return (
    <section className="table-shell">
      <div className="border-b border-border px-3 py-2">
        <p className="panel-title">{title}</p>
        <h3 className="mt-1 text-sm font-semibold text-text">{subtitle}</h3>
      </div>
      <table>
        <thead>
          <tr>
            <th>Bucket</th>
            <th>Count</th>
            <th>Share</th>
          </tr>
        </thead>
        <tbody>
          {rows.length ? (
            rows.map((row) => {
              const width = maxCount > 0 ? Math.max((row.count / maxCount) * 100, row.count > 0 ? 8 : 0) : 0;
              return (
                <tr key={row.key}>
                  <td>
                    <div className="flex items-center justify-between gap-3">
                      <span>{row.label}</span>
                      {row.label !== row.key ? <span className="text-[11px] text-muted">{row.key}</span> : null}
                    </div>
                    <div className="mt-1 h-1.5 rounded-full bg-border/70">
                      <div
                        className="h-full rounded-full bg-accent"
                        style={{ width: `${width}%` }}
                      />
                    </div>
                  </td>
                  <td>{formatStatValue(row.count, null)}</td>
                  <td>{formatShare(row.count, total)}</td>
                </tr>
              );
            })
          ) : (
            <tr>
              <td colSpan={3} className="text-xs text-muted">
                {emptyMessage}
              </td>
            </tr>
          )}
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

  const memoryActivityChart = useMemo(() => {
    const rows = nerdQuery.data?.timelines.memory_activity ?? [];
    if (!rows.length) {
      return null;
    }
    return Plot.plot({
      height: 260,
      marginLeft: 48,
      style: { background: 'transparent', color: '#c9d1d9' },
      x: { type: 'time', label: 'time' },
      y: { grid: true, label: 'memories' },
      color: { legend: true },
      marks: [
        Plot.lineY(rows, {
          x: (d) => new Date(d.bucket_start * 1000),
          y: 'created_count',
          stroke: () => 'created',
          strokeWidth: 2,
        }),
        Plot.lineY(rows, {
          x: (d) => new Date(d.bucket_start * 1000),
          y: 'updated_count',
          stroke: () => 'updated',
          strokeWidth: 2,
        }),
      ],
    });
  }, [nerdQuery.data]);

  const contentBytesChart = useMemo(() => {
    const rows = nerdQuery.data?.timelines.memory_activity ?? [];
    if (!rows.length) {
      return null;
    }
    return Plot.plot({
      height: 260,
      marginLeft: 64,
      style: { background: 'transparent', color: '#c9d1d9' },
      x: { type: 'time', label: 'time' },
      y: { grid: true, label: 'bytes' },
      marks: [
        Plot.areaY(rows, {
          x: (d) => new Date(d.bucket_start * 1000),
          y: 'total_content_bytes',
          fill: '#79c0ff',
          fillOpacity: 0.2,
        }),
        Plot.lineY(rows, {
          x: (d) => new Date(d.bucket_start * 1000),
          y: 'total_content_bytes',
          stroke: '#79c0ff',
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

  const queueRows = [
    { label: 'Runnable', value: formatStatValue(nerdQuery.data.queue_snapshot.runnable_count, null) },
    { label: 'Scheduled', value: formatStatValue(nerdQuery.data.queue_snapshot.scheduled_count, null) },
    { label: 'Oldest runnable age', value: formatStatValue(nerdQuery.data.queue_snapshot.oldest_age_seconds, 's') },
  ];

  return (
    <div className="space-y-6">
      <section className="space-y-3">
        <SectionHeading
          eyebrow="Pulse"
          title="Operational baseline"
          description="High-signal runtime pulse: headline stats, active alerts, queue health, and provider throughput telemetry."
        />

        <section className="grid gap-2 sm:grid-cols-2 xl:grid-cols-6">
        {nerdQuery.data.stats.map((stat) => (
          <article key={stat.key} className="metric-card">
            <p className="panel-title">{stat.label}</p>
            <p className="mt-1 text-lg font-semibold text-text">{formatStatValue(stat.value, stat.unit)}</p>
          </article>
        ))}
        </section>

        <section className="grid gap-4 xl:grid-cols-[1.4fr_0.6fr]">
          <section className="panel p-3">
            <p className="panel-title">Alert stream</p>
            <h3 className="mt-1 text-base font-semibold text-text">Threshold-backed operator signals</h3>
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

          <KeyValueTable title="Queue snapshot" rows={queueRows} />
        </section>
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

      <section className="space-y-3">
        <SectionHeading
          eyebrow="Composition"
          title="What the memory base is made of"
          description="Workspace, tag, type, and status breakdowns from the scoped nerd metrics payload."
        />
        <div className="grid gap-4 xl:grid-cols-2">
          <CountBreakdownTable
            title="By workspace"
            subtitle="Visible memories by workspace association"
            rows={nerdQuery.data.composition.by_workspace}
            emptyMessage="No workspace composition data yet."
          />
          <CountBreakdownTable
            title="By tag"
            subtitle="Top tags in the visible memory set"
            rows={nerdQuery.data.composition.by_tag}
            emptyMessage="No tag composition data yet."
          />
          <CountBreakdownTable
            title="By type"
            subtitle="Memory record types"
            rows={nerdQuery.data.composition.by_type}
            emptyMessage="No type composition data yet."
          />
          <CountBreakdownTable
            title="By status"
            subtitle="Lifecycle state of visible memories"
            rows={nerdQuery.data.composition.by_status}
            emptyMessage="No status composition data yet."
          />
        </div>
      </section>

      <section className="space-y-3">
        <SectionHeading
          eyebrow="Distributions"
          title="How old and how large the memory base is"
          description="Age and size buckets are shown as dense tables with relative bars to keep the page compact but still scannable."
        />
        <div className="grid gap-4 xl:grid-cols-3">
          <CountBreakdownTable
            title="Created age"
            subtitle="Creation-time age buckets"
            rows={nerdQuery.data.distributions.created_age_buckets}
            emptyMessage="No created-age distribution data yet."
          />
          <CountBreakdownTable
            title="Updated age"
            subtitle="Last-update age buckets"
            rows={nerdQuery.data.distributions.updated_age_buckets}
            emptyMessage="No updated-age distribution data yet."
          />
          <CountBreakdownTable
            title="Content size"
            subtitle="Present-day content size buckets"
            rows={nerdQuery.data.distributions.content_size_buckets}
            emptyMessage="No content-size distribution data yet."
          />
        </div>
      </section>

      <section className="space-y-3">
        <SectionHeading
          eyebrow="Timeline"
          title="Creation and update activity over the selected window"
          description="Time-series panels show whether the memory base is growing, churning, or just quietly collecting fascinating dust."
        />
        <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
          <section className="panel p-3">
            <p className="panel-title">Memory activity</p>
            <h3 className="mt-1 text-base font-semibold text-text">Created vs updated counts</h3>
            <div className="mt-3">{memoryActivityChart ? <PlotFigure chart={memoryActivityChart} /> : <p className="text-xs text-muted">No timeline activity yet.</p>}</div>
          </section>

          <section className="panel p-3">
            <p className="panel-title">Content volume</p>
            <h3 className="mt-1 text-base font-semibold text-text">Cumulative visible content bytes</h3>
            <div className="mt-3">{contentBytesChart ? <PlotFigure chart={contentBytesChart} /> : <p className="text-xs text-muted">No content-volume timeline yet.</p>}</div>
          </section>
        </div>
      </section>

      <section className="space-y-3">
        <SectionHeading
          eyebrow="Graph & Quality"
          title="Structural health and retrieval quality"
          description="Graph topology, lifecycle signals, and search-quality diagnostics remain available, but grouped away from the new composition fundamentals."
        />
        <section className="grid gap-4 xl:grid-cols-[1fr_1fr]">
          <section className="panel p-3">
            <p className="panel-title">Graph intelligence</p>
            <h3 className="mt-1 text-base font-semibold text-text">Link distribution</h3>
            <div className="mt-3">{linkTypeChart ? <PlotFigure chart={linkTypeChart} /> : <p className="text-xs text-muted">No link topology data yet.</p>}</div>
          </section>

          <section className="table-shell">
            <div className="border-b border-border px-3 py-2">
              <p className="panel-title">Stats for nerds</p>
              <h3 className="mt-1 text-base font-semibold text-text">Current derived telemetry</h3>
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
      </section>

      <section className="grid gap-4 xl:grid-cols-3">
        <KeyValueTable title="Graph topology" rows={graphRows} />
        <KeyValueTable title="Memory lifecycle" rows={lifecycleRows} />
        <KeyValueTable title="Search quality" rows={searchRows} />
      </section>

      <section className="space-y-3">
        <SectionHeading
          eyebrow="Routing"
          title="Configured vs resolved task routing"
          description="Route audit stays intact so operator-facing analytics still show where background work is supposed to go and where it actually landed."
        />
        <section className="table-shell">
          <div className="border-b border-border px-3 py-2">
            <p className="panel-title">Route audit</p>
            <h3 className="mt-1 text-base font-semibold text-text">Configured vs resolved background-agent routing</h3>
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
      </section>
    </div>
  );
}
