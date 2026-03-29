import { useEffect, useMemo, useRef, useState, type ChangeEvent } from 'react';
import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import * as Plot from '@observablehq/plot';

import { ApiError, fetchNerdMetrics, type NerdMetricsScope } from '../lib/api';

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

function formatStatValue(value: number): string {
  return Math.round(value).toLocaleString();
}

function formatTimestamp(value: number | null): string {
  if (value === null) {
    return '—';
  }
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value * 1000));
}

type NerdWindow = '24h' | '7d' | '30d';

const NERD_WINDOW_OPTIONS: Record<NerdWindow, { label: string; window_hours: number; bucket_minutes: number }> = {
  '24h': { label: '24h', window_hours: 24, bucket_minutes: 60 },
  '7d': { label: '7d', window_hours: 24 * 7, bucket_minutes: 240 },
  '30d': { label: '30d', window_hours: 24 * 30, bucket_minutes: 24 * 60 },
};

const NERD_WINDOW_ORDER: NerdWindow[] = ['24h', '7d', '30d'];

export function RetrievalPage() {
  const [selectedScope, setSelectedScope] = useState<NerdMetricsScope>('global');
  const [selectedWorkspaceId, setSelectedWorkspaceId] = useState('');
  const [selectedWindow, setSelectedWindow] = useState<NerdWindow>('7d');
  const selectedWindowConfig = NERD_WINDOW_OPTIONS[selectedWindow];

  const workspaceOptionsQuery = useQuery({
    queryKey: ['retrieval-metrics', 'workspace-options', selectedWindow],
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

  const retrievalQuery = useQuery({
    queryKey: ['retrieval-metrics', selectedWindow, selectedScope, selectedWorkspaceId],
    queryFn: () => fetchNerdMetrics({
      ...selectedWindowConfig,
      scope: selectedScope,
      workspace_id: selectedScope === 'workspace' ? selectedWorkspaceId : undefined,
    }),
    enabled: selectedScope === 'global' || Boolean(selectedWorkspaceId),
    refetchInterval: 5000,
    retry: false,
  });

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

  const searchTagTimelineChart = useMemo(() => {
    const rows = retrievalQuery.data?.retrieval.tag_timelines.flatMap((series) =>
      series.search_buckets.map((bucket) => ({
        bucket_start: bucket.bucket_start,
        count: bucket.count,
        label: series.label,
      })),
    ) ?? [];
    if (!rows.length) {
      return null;
    }
    return Plot.plot({
      height: 260,
      marginLeft: 56,
      style: { background: 'transparent', color: '#c9d1d9' },
      x: { type: 'time', label: 'time' },
      y: { grid: true, label: 'search hits' },
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
  }, [retrievalQuery.data]);

  const readTagTimelineChart = useMemo(() => {
    const rows = retrievalQuery.data?.retrieval.tag_timelines.flatMap((series) =>
      series.read_buckets.map((bucket) => ({
        bucket_start: bucket.bucket_start,
        count: bucket.count,
        label: series.label,
      })),
    ) ?? [];
    if (!rows.length) {
      return null;
    }
    return Plot.plot({
      height: 260,
      marginLeft: 56,
      style: { background: 'transparent', color: '#c9d1d9' },
      x: { type: 'time', label: 'time' },
      y: { grid: true, label: 'reads' },
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
  }, [retrievalQuery.data]);

  if (retrievalQuery.isPending) {
    return <section className="panel p-4 text-xs text-muted">Loading retrieval telemetry…</section>;
  }
  if (retrievalQuery.isError || !retrievalQuery.data) {
    const message = retrievalQuery.error instanceof ApiError && retrievalQuery.error.status === 404
      ? 'Unable to load retrieval metrics because the running daemon does not know about the updated nerd metrics payload yet. Restart the daemon, then refresh this page.'
      : 'Unable to load retrieval metrics.';
    return <section className="panel p-4 text-xs text-danger">{message}</section>;
  }

  const retrieval = retrievalQuery.data.retrieval;
  const summaryCards = [
    { key: 'search-invocations', label: 'Search invocations', value: retrieval.summary.search_invocations },
    { key: 'search-hits', label: 'Search hits', value: retrieval.summary.search_hits },
    { key: 'zero-result', label: 'Zero-result searches', value: retrieval.summary.zero_result_searches },
    { key: 'read-events', label: 'Read events', value: retrieval.summary.read_events },
    { key: 'unique-search', label: 'Unique searched memories', value: retrieval.summary.unique_search_memories },
    { key: 'unique-read', label: 'Unique read memories', value: retrieval.summary.unique_read_memories },
  ];

  return (
    <div className="space-y-6">
      <section className="space-y-3">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <p className="panel-title">Retrieval</p>
            <h2 className="mt-1 text-base font-semibold text-text">What the memory tools are actually reading and surfacing</h2>
            <p className="mt-1 text-xs text-muted">Windowed MCP retrieval telemetry: top read memories, top search-hit memories, tag rollups, and tag timelines.</p>
          </div>

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
          {summaryCards.map((stat) => (
            <article key={stat.key} className="metric-card">
              <p className="panel-title">{stat.label}</p>
              <p className="mt-1 text-lg font-semibold text-text">{formatStatValue(stat.value)}</p>
            </article>
          ))}
        </section>
      </section>

      <section className="grid gap-4 xl:grid-cols-2">
        <section className="panel p-3">
          <p className="panel-title">Search → read funnel</p>
          <h3 className="mt-1 text-base font-semibold text-text">How often surfaced search hits become actual reads</h3>
          <div className="mt-3 grid gap-2 sm:grid-cols-3">
            <article className="metric-card">
              <p className="panel-title">Search hits</p>
              <p className="mt-1 text-lg font-semibold text-text">{formatStatValue(retrieval.funnel.search_hits)}</p>
            </article>
            <article className="metric-card">
              <p className="panel-title">Converted hits</p>
              <p className="mt-1 text-lg font-semibold text-text">{formatStatValue(retrieval.funnel.converted_search_hits)}</p>
            </article>
            <article className="metric-card">
              <p className="panel-title">Conversion rate</p>
              <p className="mt-1 text-lg font-semibold text-text">{(retrieval.funnel.conversion_rate * 100).toFixed(1)}%</p>
            </article>
          </div>
          <p className="mt-3 text-[11px] text-muted">A converted hit means a surfaced memory was later read in the same window. Helpful signal, not courtroom-grade causality.</p>
        </section>

        <section className="table-shell">
          <div className="border-b border-border px-3 py-2">
            <p className="panel-title">Caller mix</p>
            <h3 className="mt-1 text-base font-semibold text-text">Which tool path is doing the retrieval</h3>
          </div>
          <table>
            <thead>
              <tr>
                <th>Caller</th>
                <th>Searches</th>
                <th>Hits</th>
                <th>Reads</th>
                <th>Zero-result</th>
              </tr>
            </thead>
            <tbody>
              {retrieval.by_caller_kind.length ? retrieval.by_caller_kind.map((row) => (
                <tr key={row.key}>
                  <td>{row.label}</td>
                  <td>{formatStatValue(row.search_invocations)}</td>
                  <td>{formatStatValue(row.search_hits)}</td>
                  <td>{formatStatValue(row.read_events)}</td>
                  <td>{formatStatValue(row.zero_result_searches)}</td>
                </tr>
              )) : (
                <tr>
                  <td colSpan={5} className="text-xs text-muted">No caller-kind retrieval activity yet.</td>
                </tr>
              )}
            </tbody>
          </table>
        </section>

        <section className="table-shell">
          <div className="border-b border-border px-3 py-2">
            <p className="panel-title">Query families</p>
            <h3 className="mt-1 text-base font-semibold text-text">Top repeated search prompts in the selected window</h3>
          </div>
          <table>
            <thead>
              <tr>
                <th>Query family</th>
                <th>Searches</th>
                <th>Hits</th>
                <th>Converted</th>
                <th>Rate</th>
                <th>Zero-result</th>
                <th>Unique memories</th>
              </tr>
            </thead>
            <tbody>
              {retrieval.top_query_families.length ? retrieval.top_query_families.map((row) => (
                <tr key={row.key}>
                  <td>
                    <div className="max-w-[24rem] truncate" title={row.label}>{row.label}</div>
                  </td>
                  <td>{formatStatValue(row.search_invocations)}</td>
                  <td>{formatStatValue(row.search_hits)}</td>
                  <td>{formatStatValue(row.converted_search_hits)}</td>
                  <td>{(row.conversion_rate * 100).toFixed(1)}%</td>
                  <td>{formatStatValue(row.zero_result_searches)}</td>
                  <td>{formatStatValue(row.unique_search_memories)}</td>
                </tr>
              )) : (
                <tr>
                  <td colSpan={7} className="text-xs text-muted">No repeated search-query patterns yet.</td>
                </tr>
              )}
            </tbody>
          </table>
        </section>
      </section>

      <section className="grid gap-4 xl:grid-cols-2">
        <section className="table-shell">
          <div className="border-b border-border px-3 py-2">
            <p className="panel-title">Zero-result diagnostics</p>
            <h3 className="mt-1 text-base font-semibold text-text">Which query families keep coming up empty</h3>
          </div>
          <table>
            <thead>
              <tr>
                <th>Query family</th>
                <th>Zero-result</th>
                <th>Searches</th>
                <th>Hits</th>
              </tr>
            </thead>
            <tbody>
              {retrieval.top_zero_result_query_families.length ? retrieval.top_zero_result_query_families.map((row) => (
                <tr key={row.key}>
                  <td><div className="max-w-[24rem] truncate" title={row.label}>{row.label}</div></td>
                  <td>{formatStatValue(row.zero_result_searches)}</td>
                  <td>{formatStatValue(row.search_invocations)}</td>
                  <td>{formatStatValue(row.search_hits)}</td>
                </tr>
              )) : (
                <tr>
                  <td colSpan={4} className="text-xs text-muted">No zero-result query families in the selected window.</td>
                </tr>
              )}
            </tbody>
          </table>
        </section>

        <section className="table-shell">
          <div className="border-b border-border px-3 py-2">
            <p className="panel-title">Low-conversion memories</p>
            <h3 className="mt-1 text-base font-semibold text-text">Memories that surface often but still do not get opened much</h3>
          </div>
          <table>
            <thead>
              <tr>
                <th>Memory</th>
                <th>Search hits</th>
                <th>Converted</th>
                <th>Reads</th>
                <th>Rate</th>
              </tr>
            </thead>
            <tbody>
              {retrieval.low_conversion_memories.length ? retrieval.low_conversion_memories.map((row) => (
                <tr key={row.memory_id}>
                  <td>
                    <Link className="text-accent" to={`/memory/${row.memory_id}`}>{row.title}</Link>
                    <div className="mt-1 text-[10px] text-muted">{row.memory_type} · {row.status}</div>
                  </td>
                  <td>{formatStatValue(row.search_count)}</td>
                  <td>{formatStatValue(row.converted_search_count)}</td>
                  <td>{formatStatValue(row.read_count)}</td>
                  <td>{(row.conversion_rate * 100).toFixed(1)}%</td>
                </tr>
              )) : (
                <tr>
                  <td colSpan={5} className="text-xs text-muted">No surfaced memories yet to rank for conversion.</td>
                </tr>
              )}
            </tbody>
          </table>
        </section>
      </section>

      <section className="grid gap-4 xl:grid-cols-2">
        <section className="table-shell">
          <div className="border-b border-border px-3 py-2">
            <p className="panel-title">Top reads</p>
            <h3 className="mt-1 text-base font-semibold text-text">Top 100 memories opened through the tool</h3>
          </div>
          <table>
            <thead>
              <tr>
                <th>Reads</th>
                <th>Search hits</th>
                <th>Memory</th>
                <th>Last read</th>
              </tr>
            </thead>
            <tbody>
              {retrieval.top_read_memories.length ? retrieval.top_read_memories.map((row) => (
                <tr key={row.memory_id}>
                  <td>{formatStatValue(row.read_count)}</td>
                  <td>{formatStatValue(row.search_count)}</td>
                  <td>
                    <Link className="text-accent" to={`/memory/${row.memory_id}`}>{row.title}</Link>
                    <div className="mt-1 text-[10px] text-muted">{row.memory_type} · {row.status}</div>
                    <div className="mt-1 truncate text-[10px] text-muted" title={row.tags.join(', ')}>{row.tags.join(', ') || '—'}</div>
                  </td>
                  <td className="text-xs text-muted">{formatTimestamp(row.last_read_at)}</td>
                </tr>
              )) : (
                <tr>
                  <td colSpan={4} className="text-xs text-muted">No read telemetry in the selected window yet.</td>
                </tr>
              )}
            </tbody>
          </table>
        </section>

        <section className="table-shell">
          <div className="border-b border-border px-3 py-2">
            <p className="panel-title">Top search hits</p>
            <h3 className="mt-1 text-base font-semibold text-text">Top 100 memories that surfaced in search results</h3>
          </div>
          <table>
            <thead>
              <tr>
                <th>Search hits</th>
                <th>Reads</th>
                <th>Memory</th>
                <th>Last surfaced</th>
              </tr>
            </thead>
            <tbody>
              {retrieval.top_search_memories.length ? retrieval.top_search_memories.map((row) => (
                <tr key={row.memory_id}>
                  <td>{formatStatValue(row.search_count)}</td>
                  <td>{formatStatValue(row.read_count)}</td>
                  <td>
                    <Link className="text-accent" to={`/memory/${row.memory_id}`}>{row.title}</Link>
                    <div className="mt-1 text-[10px] text-muted">{row.memory_type} · {row.status}</div>
                    <div className="mt-1 truncate text-[10px] text-muted" title={row.tags.join(', ')}>{row.tags.join(', ') || '—'}</div>
                  </td>
                  <td className="text-xs text-muted">{formatTimestamp(row.last_search_at)}</td>
                </tr>
              )) : (
                <tr>
                  <td colSpan={4} className="text-xs text-muted">No search-hit telemetry in the selected window yet.</td>
                </tr>
              )}
            </tbody>
          </table>
        </section>
      </section>

      <section className="table-shell">
        <div className="border-b border-border px-3 py-2">
          <p className="panel-title">Tag hotspots</p>
          <h3 className="mt-1 text-base font-semibold text-text">Top 25 tags by combined retrieval traffic</h3>
        </div>
        <table>
          <thead>
            <tr>
              <th>Tag</th>
              <th>Reads</th>
              <th>Search hits</th>
              <th>Total</th>
            </tr>
          </thead>
          <tbody>
            {retrieval.top_tags.length ? retrieval.top_tags.map((row) => (
              <tr key={row.key}>
                <td>{row.label}</td>
                <td>{formatStatValue(row.read_count)}</td>
                <td>{formatStatValue(row.search_count)}</td>
                <td>{formatStatValue(row.total_count)}</td>
              </tr>
            )) : (
              <tr>
                <td colSpan={4} className="text-xs text-muted">No tag-level retrieval activity yet.</td>
              </tr>
            )}
          </tbody>
        </table>
      </section>

      <section className="grid gap-4 xl:grid-cols-2">
        <section className="panel p-3">
          <p className="panel-title">Tag search timelines</p>
          <h3 className="mt-1 text-base font-semibold text-text">Top 10 tags by search-result appearances</h3>
          <p className="mt-1 text-[11px] text-muted">Lines use event-time buckets from tool telemetry, not memory creation timestamps.</p>
          <div className="mt-3">{searchTagTimelineChart ? <PlotFigure chart={searchTagTimelineChart} /> : <p className="text-xs text-muted">No search-tag timeline data yet.</p>}</div>
        </section>

        <section className="panel p-3">
          <p className="panel-title">Tag read timelines</p>
          <h3 className="mt-1 text-base font-semibold text-text">Top 10 tags by read traffic</h3>
          <p className="mt-1 text-[11px] text-muted">Same window, same scope, but counting explicit record reads instead of search surfaces.</p>
          <div className="mt-3">{readTagTimelineChart ? <PlotFigure chart={readTagTimelineChart} /> : <p className="text-xs text-muted">No read-tag timeline data yet.</p>}</div>
        </section>
      </section>
    </div>
  );
}