import { useEffect, useMemo, useRef, useState, type ChangeEvent } from 'react';
import { useQuery } from '@tanstack/react-query';
import * as Plot from '@observablehq/plot';
import { Link } from 'react-router-dom';

import {
  ApiError,
  fetchNerdMetrics,
  type CountBucket,
  type MaintenanceDeltaBucket,
  type MaintenanceEvent,
  type NerdMetricsScope,
  type ShareSeries,
} from '../lib/api';

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

function formatTimestamp(value: number): string {
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value * 1000));
}

function formatTaskName(taskName: string): string {
  return taskName.split('-').join(' ');
}

function formatSignedDelta(value: number): string {
  if (value > 0) {
    return `+${Math.round(value)}`;
  }
  if (value < 0) {
    return `${Math.round(value)}`;
  }
  return '0';
}

function signedDeltaClass(value: number): string {
  if (value > 0) {
    return 'text-emerald-200';
  }
  if (value < 0) {
    return 'text-danger';
  }
  return 'text-muted';
}

function formatRatio(value: number): string {
  return value.toFixed(2);
}

function latestCount(series: { buckets: Array<{ count: number }> }): number {
  return series.buckets.at(-1)?.count ?? 0;
}

function seriesCountDelta(series: { buckets: Array<{ count: number }> }): number {
  if (!series.buckets.length) {
    return 0;
  }
  return (series.buckets.at(-1)?.count ?? 0) - series.buckets[0].count;
}

function latestShare(series: ShareSeries): number {
  return series.buckets.at(-1)?.share ?? 0;
}

function maintenanceBucketDelta(bucket: MaintenanceDeltaBucket): number {
  return bucket.created_count
    + bucket.merged_count
    + bucket.updated_count
    + bucket.archived_count
    + bucket.degraded_count
    + bucket.restored_count;
}

function maintenanceStatusClass(status: string): string {
  if (status === 'failed') {
    return 'border-danger/40 bg-danger/10 text-danger';
  }
  if (status === 'retry') {
    return 'border-yellow-500/40 bg-yellow-500/10 text-yellow-200';
  }
  if (status === 'cancelled') {
    return 'border-border bg-border/40 text-muted';
  }
  return 'border-emerald-500/40 bg-emerald-500/10 text-emerald-200';
}

function maintenanceStatusColor(status: string): string {
  if (status === 'failed') {
    return '#f85149';
  }
  if (status === 'retry') {
    return '#d29922';
  }
  if (status === 'cancelled') {
    return '#8b949e';
  }
  return '#3fb950';
}

function formatMaintenanceImpact(event: MaintenanceEvent): string {
  if (event.impact_summary) {
    return event.impact_summary;
  }
  if (event.result_summary) {
    return event.result_summary;
  }
  return '—';
}

function formatMaintenanceTitle(event: MaintenanceEvent): string {
  const parts = [formatTimestamp(event.completed_at), event.task_name, event.status];
  if (event.strategy_used) {
    parts.push(`strategy=${event.strategy_used}`);
  }
  if (event.impact_summary) {
    parts.push(event.impact_summary);
  } else if (event.result_summary) {
    parts.push(event.result_summary);
  }
  return parts.join(' • ');
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

type NerdWindow = '24h' | '7d' | '30d';

const NERD_WINDOW_OPTIONS: Record<NerdWindow, { label: string; window_hours: number; bucket_minutes: number }> = {
  '24h': { label: '24h', window_hours: 24, bucket_minutes: 60 },
  '7d': { label: '7d', window_hours: 24 * 7, bucket_minutes: 240 },
  '30d': { label: '30d', window_hours: 24 * 30, bucket_minutes: 24 * 60 },
};

const NERD_WINDOW_ORDER: NerdWindow[] = ['24h', '7d', '30d'];
const QUALITY_STAT_KEYS = [
  'trace_like_memory_count',
  'generic_summary_count',
  'untagged_observation_count',
  'untagged_observation_rate',
  'oversized_memory_count',
] as const;

export function NerdPage() {
  const [selectedScope, setSelectedScope] = useState<NerdMetricsScope>('global');
  const [selectedWorkspaceId, setSelectedWorkspaceId] = useState('');
  const [selectedWindow, setSelectedWindow] = useState<NerdWindow>('24h');
  const selectedWindowConfig = NERD_WINDOW_OPTIONS[selectedWindow];

  const workspaceOptionsQuery = useQuery({
    queryKey: ['nerd-metrics', 'workspace-options', selectedWindow],
    queryFn: () => fetchNerdMetrics({ ...selectedWindowConfig, scope: 'global' }),
    staleTime: 5 * 60 * 1000,
    refetchOnWindowFocus: false,
    retry: false,
  });

  const workspaceOptions = useMemo(
    () => workspaceOptionsQuery.data?.composition.by_workspace ?? [],
    [workspaceOptionsQuery.data],
  );

  useEffect(() => {
    if (selectedScope !== 'workspace') {
      return;
    }
    if (workspaceOptions.some((option) => option.key === selectedWorkspaceId)) {
      return;
    }
    const fallbackWorkspaceId = workspaceOptions[0]?.key ?? '';
    if (fallbackWorkspaceId) {
      setSelectedWorkspaceId(fallbackWorkspaceId);
      return;
    }
    setSelectedScope('global');
  }, [selectedScope, selectedWorkspaceId, workspaceOptions]);

  const nerdQuery = useQuery({
    queryKey: ['nerd-metrics', selectedWindow, selectedScope, selectedWorkspaceId],
    queryFn: () => fetchNerdMetrics({
      ...selectedWindowConfig,
      scope: selectedScope,
      workspace_id: selectedScope === 'workspace' ? selectedWorkspaceId : undefined,
    }),
    enabled: selectedScope === 'global' || Boolean(selectedWorkspaceId),
    refetchInterval: 5000,
    retry: false,
  });

  const [selectedQualitySignal, setSelectedQualitySignal] = useState<string>('trace_like_memory_count');
  const qualityDrilldownSignals = nerdQuery.data?.quality_drilldown.signals ?? [];

  useEffect(() => {
    if (!qualityDrilldownSignals.length) {
      return;
    }
    if (qualityDrilldownSignals.some((signal) => signal.key === selectedQualitySignal)) {
      return;
    }
    setSelectedQualitySignal(qualityDrilldownSignals[0].key);
  }, [qualityDrilldownSignals, selectedQualitySignal]);

  function handleWorkspaceChange(event: ChangeEvent<HTMLSelectElement>) {
    const workspaceId = event.target.value;
    if (!workspaceId) {
      setSelectedScope('global');
      setSelectedWorkspaceId('');
      return;
    }
    setSelectedWorkspaceId(workspaceId);
    setSelectedScope('workspace');
  }

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

  const maintenanceEvents = nerdQuery.data?.maintenance.events ?? [];

  const maintenanceOverlayEvents = useMemo(
    () => maintenanceEvents.filter((event) => event.status !== 'completed' || event.impact_summary || event.strategy_used).slice(0, 16),
    [maintenanceEvents],
  );

  const recentMaintenanceEvents = useMemo(() => maintenanceEvents.slice(0, 12), [maintenanceEvents]);

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
        Plot.ruleX(maintenanceOverlayEvents, {
          x: (d) => new Date(d.completed_at * 1000),
          stroke: (d) => maintenanceStatusColor(d.status),
          strokeOpacity: 0.3,
          strokeWidth: 1.5,
          title: (d) => formatMaintenanceTitle(d),
        }),
      ],
    });
  }, [maintenanceOverlayEvents, nerdQuery.data]);

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
        Plot.ruleX(maintenanceOverlayEvents, {
          x: (d) => new Date(d.completed_at * 1000),
          stroke: (d) => maintenanceStatusColor(d.status),
          strokeOpacity: 0.28,
          strokeWidth: 1.5,
          title: (d) => formatMaintenanceTitle(d),
        }),
      ],
    });
  }, [maintenanceOverlayEvents, nerdQuery.data]);

  const lifecycleTrendChart = useMemo(() => {
    const backlogRows = nerdQuery.data?.lifecycle_trends.never_surfaced_backlog ?? [];
    const coldTailRows = nerdQuery.data?.lifecycle_trends.cold_tail ?? [];
    if (!backlogRows.length && !coldTailRows.length) {
      return null;
    }
    const rows = [
      ...backlogRows.map((bucket) => ({ ...bucket, label: 'never surfaced backlog' })),
      ...coldTailRows.map((bucket) => ({ ...bucket, label: 'cold tail' })),
    ];
    return Plot.plot({
      height: 240,
      marginLeft: 56,
      style: { background: 'transparent', color: '#c9d1d9' },
      x: { type: 'time', label: 'time' },
      y: { grid: true, label: 'memories' },
      color: { legend: true },
      marks: [
        Plot.lineY(rows, {
          x: (d) => new Date(d.bucket_start * 1000),
          y: 'count',
          stroke: 'label',
          strokeWidth: 2,
        }),
      ],
    });
  }, [nerdQuery.data]);

  const topTagTrendChart = useMemo(() => {
    const rows = nerdQuery.data?.growth_dynamics.top_tag_trends.flatMap((series) =>
      series.buckets.map((bucket) => ({
        bucket_start: bucket.bucket_start,
        count: bucket.count,
        label: series.label,
      })),
    ) ?? [];
    if (!rows.length) {
      return null;
    }
    return Plot.plot({
      height: 240,
      marginLeft: 56,
      style: { background: 'transparent', color: '#c9d1d9' },
      x: { type: 'time', label: 'time' },
      y: { grid: true, label: 'memories' },
      color: { legend: true },
      marks: [
        Plot.lineY(rows, {
          x: (d) => new Date(d.bucket_start * 1000),
          y: 'count',
          stroke: 'label',
          strokeWidth: 2,
        }),
      ],
    });
  }, [nerdQuery.data]);

  const familyDeltaChart = useMemo(() => {
    const rows = nerdQuery.data?.maintenance_summary.family_delta_series.flatMap((series) =>
      series.buckets.map((bucket) => ({
        bucket_start: bucket.bucket_start,
        delta_total: maintenanceBucketDelta(bucket),
        label: series.label,
      })),
    ).filter((row) => row.delta_total > 0) ?? [];
    if (!rows.length) {
      return null;
    }
    return Plot.plot({
      height: 240,
      marginLeft: 56,
      style: { background: 'transparent', color: '#c9d1d9' },
      x: { type: 'time', label: 'time' },
      y: { grid: true, label: 'delta' },
      color: { legend: true },
      marks: [
        Plot.lineY(rows, {
          x: (d) => new Date(d.bucket_start * 1000),
          y: 'delta_total',
          stroke: 'label',
          strokeWidth: 2,
        }),
        Plot.dot(rows, {
          x: (d) => new Date(d.bucket_start * 1000),
          y: 'delta_total',
          fill: 'label',
          r: 3,
        }),
      ],
    });
  }, [nerdQuery.data]);

  const qualityTrendChart = useMemo(() => {
    const rows = nerdQuery.data?.lifecycle_trends.quality_signals.flatMap((series) =>
      series.buckets.map((bucket) => ({
        bucket_start: bucket.bucket_start,
        count: bucket.count,
        label: series.label,
      })),
    ) ?? [];
    if (!rows.length) {
      return null;
    }
    return Plot.plot({
      height: 250,
      marginLeft: 56,
      style: { background: 'transparent', color: '#c9d1d9' },
      x: { type: 'time', label: 'time' },
      y: { grid: true, label: 'memories' },
      color: { legend: true },
      marks: [
        Plot.lineY(rows, {
          x: (d) => new Date(d.bucket_start * 1000),
          y: 'count',
          stroke: 'label',
          strokeWidth: 2,
        }),
      ],
    });
  }, [nerdQuery.data]);

  const remediationActivityChart = useMemo(() => {
    const rows = nerdQuery.data?.quality_remediation.activity.flatMap((series) =>
      series.buckets.map((bucket) => ({
        bucket_start: bucket.bucket_start,
        count: bucket.count,
        label: series.label,
      })),
    ) ?? [];
    if (!rows.length) {
      return null;
    }
    return Plot.plot({
      height: 250,
      marginLeft: 56,
      style: { background: 'transparent', color: '#c9d1d9' },
      x: { type: 'time', label: 'time' },
      y: { grid: true, label: 'updated memories' },
      color: { legend: true },
      marks: [
        Plot.lineY(rows, {
          x: (d) => new Date(d.bucket_start * 1000),
          y: 'count',
          stroke: 'label',
          strokeWidth: 2,
        }),
        Plot.dot(rows, {
          x: (d) => new Date(d.bucket_start * 1000),
          y: 'count',
          fill: 'label',
          r: 3,
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
  const providerPolicyStatRows = nerdQuery.data.provider_policy.stats;
  const providerPolicyTaskRows = nerdQuery.data.provider_policy.by_task;
  const providerPolicyProviderRows = nerdQuery.data.provider_policy.by_provider;

  const queueRows = [
    { label: 'Runnable', value: formatStatValue(nerdQuery.data.queue_snapshot.runnable_count, null) },
    { label: 'Scheduled', value: formatStatValue(nerdQuery.data.queue_snapshot.scheduled_count, null) },
    { label: 'Oldest runnable age', value: formatStatValue(nerdQuery.data.queue_snapshot.oldest_age_seconds, 's') },
  ];

  const lifecycleEventRows = nerdQuery.data.lifecycle_trends.status_events.map((series) => ({
    key: series.key,
    label: series.label,
    total: series.buckets.reduce((sum, bucket) => sum + bucket.count, 0),
    latest: latestCount(series),
  }));

  const topTagRows = nerdQuery.data.growth_dynamics.top_tag_trends.map((series) => ({
    key: series.key,
    label: series.label,
    currentCount: latestCount(series),
    delta: seriesCountDelta(series),
  }));

  const workspaceShareRows = nerdQuery.data.growth_dynamics.workspace_contribution_share.map((series) => ({
    key: series.key,
    label: series.label,
    currentCount: latestCount(series),
    share: latestShare(series),
  }));

  const qualitySignalRows = nerdQuery.data.lifecycle_trends.quality_signals.map((series) => ({
    key: series.key,
    label: series.label,
    currentCount: latestCount(series),
    delta: seriesCountDelta(series),
  }));

  const qualityStatRows = nerdQuery.data.stats.filter((stat) => QUALITY_STAT_KEYS.includes(stat.key as (typeof QUALITY_STAT_KEYS)[number]));
  const activeQualitySignal = qualityDrilldownSignals.find((signal) => signal.key === selectedQualitySignal) ?? qualityDrilldownSignals[0] ?? null;
  const remediationStatRows = nerdQuery.data.quality_remediation.stats;
  const remediationActivityRows = nerdQuery.data.quality_remediation.activity.map((series) => ({
    key: series.key,
    label: series.label,
    total: series.buckets.reduce((sum, bucket) => sum + bucket.count, 0),
    latest: latestCount(series),
  }));

  const familySummaryRows = [...nerdQuery.data.maintenance_summary.by_family]
    .sort((left, right) => right.delta_total - left.delta_total || right.completed_runs - left.completed_runs);

  const agentYieldRows = [...nerdQuery.data.maintenance_summary.by_agent]
    .sort((left, right) => right.delta_per_completed_run - left.delta_per_completed_run || right.completed_runs - left.completed_runs);

  return (
    <div className="space-y-6">
      <section className="space-y-3">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <SectionHeading
            eyebrow="Pulse"
            title="Operational baseline"
            description="High-signal runtime pulse: headline stats, active alerts, queue health, and provider throughput telemetry."
          />

          <div className="flex flex-wrap items-center gap-2 self-start">
            <section className="panel flex items-center gap-1 p-1">
              <span className="px-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted">Window</span>
              {NERD_WINDOW_ORDER.map((windowKey) => {
                const isSelected = windowKey === selectedWindow;
                return (
                  <button
                    key={windowKey}
                    type="button"
                    onClick={() => setSelectedWindow(windowKey)}
                    className={`rounded-full border px-3 py-1 text-[11px] font-medium transition ${isSelected
                      ? 'border-accent bg-accent text-ink'
                      : 'border-border bg-transparent text-muted hover:border-accent hover:text-text'
                    }`}
                  >
                    {NERD_WINDOW_OPTIONS[windowKey].label}
                  </button>
                );
              })}
            </section>

            <section className="panel flex items-center gap-2 p-1">
              <span className="px-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted">Scope</span>
              <button
                type="button"
                onClick={() => setSelectedScope('global')}
                className={`rounded-full border px-3 py-1 text-[11px] font-medium transition ${selectedScope === 'global'
                  ? 'border-accent bg-accent text-ink'
                  : 'border-border bg-transparent text-muted hover:border-accent hover:text-text'
                }`}
              >
                Global
              </button>
              <label className="flex items-center">
                <span className="sr-only">Workspace scope</span>
                <select
                  value={selectedScope === 'workspace' ? selectedWorkspaceId : ''}
                  onChange={handleWorkspaceChange}
                  disabled={workspaceOptionsQuery.isPending || workspaceOptions.length === 0}
                  className="rounded-full border border-border bg-transparent px-3 py-1 text-[11px] font-medium text-text outline-none transition hover:border-accent disabled:cursor-not-allowed disabled:text-muted"
                >
                  <option value="">
                    {workspaceOptionsQuery.isPending ? 'Loading workspaces…' : 'Workspace…'}
                  </option>
                  {workspaceOptions.map((option) => (
                    <option key={option.key} value={option.key}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
            </section>
          </div>
        </div>

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
          description="Workspace, content-tag, provenance/process-tag, type, and status breakdowns from the scoped nerd metrics payload."
        />
        <div className="grid gap-4 xl:grid-cols-2 2xl:grid-cols-3">
          <CountBreakdownTable
            title="By workspace"
            subtitle="Visible memories by workspace association"
            rows={nerdQuery.data.composition.by_workspace}
            emptyMessage="No workspace composition data yet."
          />
          <CountBreakdownTable
            title="Content tags"
            subtitle="Top domain/content tags in the visible memory set"
            rows={nerdQuery.data.composition.by_content_tag}
            emptyMessage="No content-tag composition data yet."
          />
          <CountBreakdownTable
            title="Provenance / process tags"
            subtitle="System/process markers split from content tags"
            rows={nerdQuery.data.composition.by_provenance_tag}
            emptyMessage="No provenance/process-tag composition data yet."
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
            <p className="mt-1 text-[11px] text-muted">Subtle vertical rules mark the most recent maintenance runs in the selected window.</p>
            <div className="mt-3">{memoryActivityChart ? <PlotFigure chart={memoryActivityChart} /> : <p className="text-xs text-muted">No timeline activity yet.</p>}</div>
          </section>

          <section className="panel p-3">
            <p className="panel-title">Content volume</p>
            <h3 className="mt-1 text-base font-semibold text-text">Cumulative visible content bytes</h3>
            <p className="mt-1 text-[11px] text-muted">Overlay markers share the same event stream, so spikes line up with concrete maintenance evidence.</p>
            <div className="mt-3">{contentBytesChart ? <PlotFigure chart={contentBytesChart} /> : <p className="text-xs text-muted">No content-volume timeline yet.</p>}</div>
          </section>
        </div>

        <section className="table-shell">
          <div className="border-b border-border px-3 py-2">
            <p className="panel-title">Maintenance evidence</p>
            <h3 className="mt-1 text-base font-semibold text-text">Recent task runs behind the timeline markers</h3>
          </div>
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Task</th>
                <th>Status</th>
                <th>Strategy</th>
                <th>Impact</th>
                <th>Summary</th>
              </tr>
            </thead>
            <tbody>
              {recentMaintenanceEvents.length ? (
                recentMaintenanceEvents.map((event) => (
                  <tr key={`${event.task_id}-${event.completed_at}`}>
                    <td className="whitespace-nowrap text-xs text-muted">{formatTimestamp(event.completed_at)}</td>
                    <td>
                      <div className="font-medium text-text">{formatTaskName(event.task_name)}</div>
                      <div className="text-[11px] text-muted" title={event.task_id}>{event.task_id}</div>
                    </td>
                    <td>
                      <span className={`inline-flex rounded-full border px-2 py-0.5 text-[11px] font-medium ${maintenanceStatusClass(event.status)}`}>
                        {event.status}
                      </span>
                    </td>
                    <td className="text-xs text-muted">{event.strategy_used ?? '—'}</td>
                    <td className="text-xs text-muted">{formatMaintenanceImpact(event)}</td>
                    <td className="text-xs text-muted">{event.result_summary ?? '—'}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={6} className="text-xs text-muted">
                    No maintenance task evidence in the selected window yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </section>
      </section>

      <section className="space-y-3">
        <SectionHeading
          eyebrow="Lifecycle & Growth"
          title="Backlog pressure and growth mix"
          description="Current-stock trend lines keep never-surfaced pressure, cold-tail drift, and dominant tag/workspace growth visible without turning the page into a chart petting zoo."
        />
        <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
          <section className="panel p-3">
            <p className="panel-title">Lifecycle backlog</p>
            <h3 className="mt-1 text-base font-semibold text-text">Never surfaced vs cold tail</h3>
            <p className="mt-1 text-[11px] text-muted">Both lines are cumulative scoped stock approximations over the selected window.</p>
            <div className="mt-3">{lifecycleTrendChart ? <PlotFigure chart={lifecycleTrendChart} /> : <p className="text-xs text-muted">No lifecycle-trend data yet.</p>}</div>
          </section>

          <section className="panel p-3">
            <p className="panel-title">Growth dynamics</p>
            <h3 className="mt-1 text-base font-semibold text-text">Top-tag stock over time</h3>
            <p className="mt-1 text-[11px] text-muted">Top tags are capped by the backend; `other` absorbs the tail when the tag zoo gets ambitious.</p>
            <div className="mt-3">{topTagTrendChart ? <PlotFigure chart={topTagTrendChart} /> : <p className="text-xs text-muted">No tag-trend data yet.</p>}</div>
          </section>
        </div>

        <div className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
          <section className="table-shell">
            <div className="border-b border-border px-3 py-2">
              <p className="panel-title">Lifecycle events</p>
              <h3 className="mt-1 text-base font-semibold text-text">Run-reported status deltas</h3>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Event</th>
                  <th>Total</th>
                  <th>Latest bucket</th>
                </tr>
              </thead>
              <tbody>
                {lifecycleEventRows.length ? (
                  lifecycleEventRows.map((row) => (
                    <tr key={row.key}>
                      <td>{row.label}</td>
                      <td>{formatStatValue(row.total, null)}</td>
                      <td>{formatStatValue(row.latest, null)}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={3} className="text-xs text-muted">
                      No explicit lifecycle event counters landed in the selected window.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </section>

          <section className="grid gap-4 lg:grid-cols-[1fr_1fr]">
            <section className="table-shell">
              <div className="border-b border-border px-3 py-2">
                <p className="panel-title">Top tags</p>
                <h3 className="mt-1 text-base font-semibold text-text">Current stock and window delta</h3>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>Tag</th>
                    <th>Current</th>
                    <th>Δ window</th>
                  </tr>
                </thead>
                <tbody>
                  {topTagRows.length ? (
                    topTagRows.map((row) => (
                      <tr key={row.key}>
                        <td>{row.label}</td>
                        <td>{formatStatValue(row.currentCount, null)}</td>
                        <td className={signedDeltaClass(row.delta)}>{formatSignedDelta(row.delta)}</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={3} className="text-xs text-muted">
                        No scoped top-tag growth data yet.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </section>

            <section className="table-shell">
              <div className="border-b border-border px-3 py-2">
                <p className="panel-title">Workspace share</p>
                <h3 className="mt-1 text-base font-semibold text-text">Current contribution mix</h3>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>Workspace</th>
                    <th>Count</th>
                    <th>Share</th>
                  </tr>
                </thead>
                <tbody>
                  {workspaceShareRows.length ? (
                    workspaceShareRows.map((row) => (
                      <tr key={row.key}>
                        <td>{row.label}</td>
                        <td>{formatStatValue(row.currentCount, null)}</td>
                        <td>{formatStatValue(row.share, 'pct')}</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={3} className="text-xs text-muted">
                        No workspace-share growth data yet.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </section>
          </section>
        </div>
      </section>

      <section className="space-y-3">
        <SectionHeading
          eyebrow="Memory Quality"
          title="Signal hygiene and drift over time"
          description="These panels track the exact anti-patterns we just started guarding against: trace-like records, generic summaries, missing tags, and oversized memories." 
        />

        <section className="grid gap-2 sm:grid-cols-2 xl:grid-cols-5">
          {qualityStatRows.map((stat) => (
            <article key={stat.key} className="metric-card">
              <p className="panel-title">{stat.label}</p>
              <p className="mt-1 text-lg font-semibold text-text">{formatStatValue(stat.value, stat.unit)}</p>
            </article>
          ))}
        </section>

        <div className="grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
          <section className="panel p-3">
            <p className="panel-title">Quality drift</p>
            <h3 className="mt-1 text-base font-semibold text-text">Current-stock trend lines for problematic memories</h3>
            <p className="mt-1 text-[11px] text-muted">These are cumulative scoped-stock lines, so rising curves mean bad patterns are sticking around instead of being cleaned up.</p>
            <div className="mt-3">{qualityTrendChart ? <PlotFigure chart={qualityTrendChart} /> : <p className="text-xs text-muted">No quality-signal trend data yet.</p>}</div>
          </section>

          <section className="table-shell">
            <div className="border-b border-border px-3 py-2">
              <p className="panel-title">Current signals</p>
              <h3 className="mt-1 text-base font-semibold text-text">Stock and window delta</h3>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Signal</th>
                  <th>Current</th>
                  <th>Δ window</th>
                </tr>
              </thead>
              <tbody>
                {qualitySignalRows.length ? (
                  qualitySignalRows.map((row) => (
                    <tr key={row.key}>
                      <td>{row.label}</td>
                      <td>{formatStatValue(row.currentCount, null)}</td>
                      <td className={signedDeltaClass(row.delta)}>{formatSignedDelta(row.delta)}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={3} className="text-xs text-muted">
                      No quality drift signals in the selected window.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </section>
        </div>

        <div className="grid gap-4 xl:grid-cols-[0.95fr_1.05fr]">
          <section className="table-shell">
            <div className="border-b border-border px-3 py-2">
              <p className="panel-title">Signal drill-down</p>
              <h3 className="mt-1 text-base font-semibold text-text">Select a quality signal to inspect the offending memories</h3>
            </div>
            <div className="flex flex-wrap gap-2 border-b border-border px-3 py-3">
              {qualityDrilldownSignals.map((signal) => {
                const isActive = signal.key === activeQualitySignal?.key;
                return (
                  <button
                    key={signal.key}
                    type="button"
                    onClick={() => setSelectedQualitySignal(signal.key)}
                    className={`rounded-full border px-3 py-1 text-[11px] font-medium transition ${isActive
                      ? 'border-accent bg-accent text-ink'
                      : 'border-border bg-transparent text-muted hover:border-accent hover:text-text'
                    }`}
                  >
                    {signal.label} · {signal.count}
                  </button>
                );
              })}
            </div>
            <table>
              <thead>
                <tr>
                  <th>Memory</th>
                  <th>Type</th>
                  <th>Status</th>
                  <th>Updated</th>
                </tr>
              </thead>
              <tbody>
                {activeQualitySignal?.records.length ? (
                  activeQualitySignal.records.map((record) => (
                    <tr key={`${activeQualitySignal.key}-${record.memory_id}`}>
                      <td>
                        <Link className="block truncate font-medium text-accent" to={`/memory/${record.memory_id}`} title={record.title}>{record.title}</Link>
                        {record.summary ? <div className="mt-1 max-w-[30rem] truncate text-[11px] text-muted" title={record.summary}>{record.summary}</div> : null}
                        {record.tags.length ? <div className="mt-1 max-w-[30rem] truncate text-[10px] text-muted" title={record.tags.join(', ')}>{record.tags.join(', ')}</div> : null}
                      </td>
                      <td>{record.memory_type}</td>
                      <td>{record.status}</td>
                      <td className="text-xs text-muted">{record.updated_at}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={4} className="text-xs text-muted">
                      No offending memories are currently visible for this signal.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </section>

          <section className="table-shell">
            <div className="border-b border-border px-3 py-2">
              <p className="panel-title">Operator note</p>
              <h3 className="mt-1 text-base font-semibold text-text">How to read this section</h3>
            </div>
            <div className="space-y-2 px-3 py-3 text-xs text-muted">
              <p>
                The drift chart shows <span className="text-text">current bad-stock</span>: if the line rises, low-quality memories are accumulating.
              </p>
              <p>
                The drill-down table is the actionable slice: open the memory, split it, retag it, or tighten the summary.
              </p>
              <p>
                The remediation charts below are <span className="text-text">proxy cleanup activity</span> derived from updated memories that now carry better hygiene markers.
              </p>
            </div>
          </section>
        </div>

        <div className="grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
          <section className="panel p-3">
            <p className="panel-title">Remediation proxies</p>
            <h3 className="mt-1 text-base font-semibold text-text">Recent cleanup-style updates</h3>
            <p className="mt-1 text-[11px] text-muted">These counts are derived from updated memories with better hygiene markers, so they are best read as remediation proxies rather than audited mutation logs.</p>
            <div className="mt-3">{remediationActivityChart ? <PlotFigure chart={remediationActivityChart} /> : <p className="text-xs text-muted">No remediation activity proxies in the selected window.</p>}</div>
          </section>

          <section className="space-y-4">
            <section className="grid gap-2 sm:grid-cols-3 xl:grid-cols-1">
              {remediationStatRows.map((stat) => (
                <article key={stat.key} className="metric-card">
                  <p className="panel-title">{stat.label}</p>
                  <p className="mt-1 text-lg font-semibold text-text">{formatStatValue(stat.value, stat.unit)}</p>
                </article>
              ))}
            </section>

            <section className="table-shell">
              <div className="border-b border-border px-3 py-2">
                <p className="panel-title">Recent remediation buckets</p>
                <h3 className="mt-1 text-base font-semibold text-text">Latest bucket and total window counts</h3>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>Proxy</th>
                    <th>Total</th>
                    <th>Latest bucket</th>
                  </tr>
                </thead>
                <tbody>
                  {remediationActivityRows.length ? (
                    remediationActivityRows.map((row) => (
                      <tr key={row.key}>
                        <td>{row.label}</td>
                        <td>{formatStatValue(row.total, null)}</td>
                        <td>{formatStatValue(row.latest, null)}</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={3} className="text-xs text-muted">
                        No remediation proxy activity in the selected window.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </section>
          </section>
        </div>
      </section>

      <section className="space-y-3">
        <SectionHeading
          eyebrow="Maintenance Yield"
          title="What background work actually produced"
          description="Family rollups stay operator-readable while agent-level yield ratios show whether maintenance runs are paying rent."
        />
        <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
          <section className="panel p-3">
            <p className="panel-title">Family deltas</p>
            <h3 className="mt-1 text-base font-semibold text-text">Per-family maintenance output over time</h3>
            <p className="mt-1 text-[11px] text-muted">Each point is the sum of explicit run-reported delta counters in that bucket.</p>
            <div className="mt-3">{familyDeltaChart ? <PlotFigure chart={familyDeltaChart} /> : <p className="text-xs text-muted">No family delta evidence yet.</p>}</div>
          </section>

          <section className="table-shell">
            <div className="border-b border-border px-3 py-2">
              <p className="panel-title">By family</p>
              <h3 className="mt-1 text-base font-semibold text-text">Maintenance summary</h3>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Family</th>
                  <th>Runs</th>
                  <th>Delta</th>
                  <th>Actions</th>
                  <th>Lines</th>
                </tr>
              </thead>
              <tbody>
                {familySummaryRows.length ? (
                  familySummaryRows.map((row) => (
                    <tr key={row.key}>
                      <td>
                        <div className="font-medium text-text">{row.label}</div>
                        <div className="text-[11px] text-muted">{row.task_names.map(formatTaskName).join(', ') || '—'}</div>
                      </td>
                      <td className="text-xs text-muted">{`${row.completed_runs}/${row.total_runs} c · ${row.failed_runs} f · ${row.retry_runs} r`}</td>
                      <td>{formatStatValue(row.delta_total, null)}</td>
                      <td>{formatStatValue(row.meaningful_actions, null)}</td>
                      <td>{formatStatValue(row.lines_compressed, null)}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={5} className="text-xs text-muted">
                      No maintenance summary rows in the selected window.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </section>
        </div>

        <section className="table-shell">
          <div className="border-b border-border px-3 py-2">
            <p className="panel-title">By agent</p>
            <h3 className="mt-1 text-base font-semibold text-text">Condensed agent yield</h3>
          </div>
          <table>
            <thead>
              <tr>
                <th>Agent</th>
                <th>Family</th>
                <th>Completed</th>
                <th>Δ/run</th>
                <th>Actions/run</th>
                <th>Lines/run</th>
              </tr>
            </thead>
            <tbody>
              {agentYieldRows.length ? (
                agentYieldRows.map((row) => (
                  <tr key={row.key}>
                    <td>{formatTaskName(row.label)}</td>
                    <td>{row.family_label}</td>
                    <td>{formatStatValue(row.completed_runs, null)}</td>
                    <td>{formatRatio(row.delta_per_completed_run)}</td>
                    <td>{formatRatio(row.actions_per_completed_run)}</td>
                    <td>{formatRatio(row.lines_per_completed_run)}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={6} className="text-xs text-muted">
                    No agent-yield rows in the selected window.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </section>
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

        <section className="grid gap-2 sm:grid-cols-3">
          {providerPolicyStatRows.map((stat) => (
            <article key={stat.key} className="metric-card">
              <p className="panel-title">{stat.label}</p>
              <p className="mt-1 text-lg font-semibold text-text">{formatStatValue(stat.value, stat.unit)}</p>
            </article>
          ))}
        </section>

        <section className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
          <section className="table-shell">
            <div className="border-b border-border px-3 py-2">
              <p className="panel-title">Provider policy by task</p>
              <h3 className="mt-1 text-base font-semibold text-text">Windowed route exhaustion, fallback denials, and admission skips</h3>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Task</th>
                  <th>Route exhausted</th>
                  <th>Fallback denied</th>
                  <th>Admission skips</th>
                  <th>Top skipped provider</th>
                </tr>
              </thead>
              <tbody>
                {providerPolicyTaskRows.length ? (
                  providerPolicyTaskRows.map((row) => (
                    <tr key={row.task_name}>
                      <td>
                        <div className="font-medium text-text">{row.task_name}</div>
                        <div className="text-[11px] text-muted">
                          {row.active_admission_provider_count > 0 ? `${row.active_admission_provider_count} active admission gate(s)` : 'No active admissions'}
                        </div>
                      </td>
                      <td>{formatStatValue(row.route_exhaustion_count, null)}</td>
                      <td>{formatStatValue(row.legacy_fallback_denied_count, null)}</td>
                      <td>{formatStatValue(row.admission_skip_count, null)}</td>
                      <td className="text-xs text-muted">
                        {row.top_skip_provider_key
                          ? `${row.top_skip_provider_key}${row.top_skip_model_name ? ` (${row.top_skip_model_name})` : ''}${row.top_skip_reason_code ? ` · ${row.top_skip_reason_code}` : ''}`
                          : '—'}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={5} className="text-xs text-muted">
                      No provider-policy task rollups in the selected window.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </section>

          <section className="table-shell">
            <div className="border-b border-border px-3 py-2">
              <p className="panel-title">Provider policy by provider</p>
              <h3 className="mt-1 text-base font-semibold text-text">Who is getting skipped and why</h3>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Provider</th>
                  <th>Skips</th>
                  <th>Top task</th>
                  <th>Active admission</th>
                </tr>
              </thead>
              <tbody>
                {providerPolicyProviderRows.length ? (
                  providerPolicyProviderRows.map((row) => (
                    <tr key={`${row.provider_key}:${row.model_name}`}>
                      <td>
                        <div className="font-medium text-text">{row.provider_key}</div>
                        <div className="text-[11px] text-muted">{row.model_name}</div>
                      </td>
                      <td>
                        <div>{formatStatValue(row.admission_skip_count, null)}</div>
                        <div className="text-[11px] text-muted">{row.distinct_task_count} task(s)</div>
                      </td>
                      <td className="text-xs text-muted">
                        {row.top_task_name ?? '—'}
                        {row.top_reason_code ? <div className="mt-1 text-[11px]">{row.top_reason_code}</div> : null}
                      </td>
                      <td className="text-xs text-muted">
                        {row.active_admission_reason
                          ? `${row.active_admission_reason}${row.active_retry_delay_seconds !== null ? ` · retry ${formatStatValue(row.active_retry_delay_seconds, 's')}` : ''}`
                          : '—'}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={4} className="text-xs text-muted">
                      No provider-level admission rollups in the selected window.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </section>
        </section>

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
