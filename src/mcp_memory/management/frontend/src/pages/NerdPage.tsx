import { useEffect, useMemo, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import * as Plot from '@observablehq/plot';

import { fetchNerdMetrics } from '../lib/api';

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
  return `${Math.round(value)}${unit ? ` ${unit}` : ''}`;
}

export function NerdPage() {
  const nerdQuery = useQuery({
    queryKey: ['nerd-metrics'],
    queryFn: () => fetchNerdMetrics({ window_hours: 24, bucket_minutes: 60 }),
    refetchInterval: 5000,
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

  if (nerdQuery.isPending) {
    return <section className="panel p-4 text-xs text-muted">Loading nerd telemetry…</section>;
  }
  if (nerdQuery.isError || !nerdQuery.data) {
    return <section className="panel p-4 text-xs text-danger">Unable to load nerd metrics.</section>;
  }

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
    </div>
  );
}