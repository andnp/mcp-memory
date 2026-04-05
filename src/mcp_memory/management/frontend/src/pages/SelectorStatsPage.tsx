import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';

import {
  ApiError,
  fetchNerdMetrics,
  fetchSelectorStats,
  type SelectorFeatureRollupRow,
  type NerdMetricsScope,
  type SelectorMetricSnapshot,
  type SelectorPopulationSnapshot,
} from '../lib/api';
import { WorkspaceScopeSelector } from '../components/WorkspaceScopeSelector';

type SelectorWindow = '24h' | '7d' | '30d';

const SELECTOR_WINDOW_OPTIONS: Record<SelectorWindow, { label: string; window_hours: number }> = {
  '24h': { label: '24h', window_hours: 24 },
  '7d': { label: '7d', window_hours: 24 * 7 },
  '30d': { label: '30d', window_hours: 24 * 30 },
};

const SELECTOR_WINDOW_ORDER: SelectorWindow[] = ['24h', '7d', '30d'];

const SNAPSHOT_METRIC_ORDER = [
  { key: 'content_chars', label: 'Content chars', kind: 'number' as const },
  { key: 'updated_age_seconds', label: 'Updated age', kind: 'age' as const },
  { key: 'last_access_age_seconds', label: 'Last access age', kind: 'age' as const },
  { key: 'last_surfaced_age_seconds', label: 'Last surfaced age', kind: 'age' as const },
  { key: 'read_count', label: 'Read count', kind: 'number' as const },
  { key: 'support_count', label: 'Support count', kind: 'number' as const },
];

const SNAPSHOT_SHARE_ORDER = [
  { key: 'never_surfaced_share', label: 'Never surfaced' },
  { key: 'never_accessed_share', label: 'Never accessed' },
  { key: 'cooldown_share', label: 'Cooldown' },
  { key: 'low_support_share', label: 'Low support' },
];

function formatTimestamp(value: number): string {
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value * 1000));
}

function formatRatio(value: number | null): string {
  return value === null ? '—' : value.toFixed(2);
}

function formatScores(scores: Record<string, number>): string {
  const entries = Object.entries(scores);
  if (!entries.length) {
    return '—';
  }
  return entries
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .map(([key, value]) => `${key}=${value.toFixed(3)}`)
    .join(', ');
}

function formatCompactAge(seconds: number | null): string {
  if (seconds === null) {
    return '—';
  }
  if (seconds >= 24 * 60 * 60) {
    return `${(seconds / (24 * 60 * 60)).toFixed(1)}d`;
  }
  if (seconds >= 60 * 60) {
    return `${(seconds / (60 * 60)).toFixed(1)}h`;
  }
  if (seconds >= 60) {
    return `${(seconds / 60).toFixed(1)}m`;
  }
  return `${seconds.toFixed(1)}s`;
}

function formatMetricValue(value: number | null, kind: 'number' | 'age'): string {
  if (value === null) {
    return '—';
  }
  if (kind === 'age') {
    return formatCompactAge(value);
  }
  return value >= 100 ? value.toFixed(0) : value.toFixed(1);
}

function formatMetricSummary(metric: SelectorMetricSnapshot | undefined, kind: 'number' | 'age'): string {
  if (!metric || metric.count <= 0) {
    return '—';
  }
  return `μ ${formatMetricValue(metric.mean, kind)} · p50 ${formatMetricValue(metric.p50, kind)} · p90 ${formatMetricValue(metric.p90, kind)}`;
}

function formatRollupMetricSummary(
  metrics: Record<string, number>,
  kindByKey: Record<string, 'number' | 'age'>,
): Array<{ key: string; label: string; value: string }> {
  return SNAPSHOT_METRIC_ORDER
    .filter(({ key }) => metrics[key] !== undefined)
    .map((metric) => ({
      key: metric.key,
      label: metric.label,
      value: formatMetricValue(metrics[metric.key] ?? null, kindByKey[metric.key] ?? metric.kind),
    }));
}

function formatRollupShareSummary(
  row: SelectorFeatureRollupRow,
): Array<{ key: string; label: string; candidate: string; selected: string }> {
  return SNAPSHOT_SHARE_ORDER
    .filter(
      ({ key }) => row.candidate_share_means[key] !== undefined || row.selected_share_means[key] !== undefined,
    )
    .map((share) => ({
      key: share.key,
      label: share.label,
      candidate: formatShare(row.candidate_share_means[share.key]),
      selected: formatShare(row.selected_share_means[share.key]),
    }));
}

function formatShare(value: number | undefined): string {
  return value === undefined ? '—' : `${(value * 100).toFixed(1)}%`;
}

function availableSnapshotMetrics(
  candidatePopulation: SelectorPopulationSnapshot,
  selectedPopulation: SelectorPopulationSnapshot,
) {
  return SNAPSHOT_METRIC_ORDER.filter(
    ({ key }) => candidatePopulation.metrics[key] !== undefined || selectedPopulation.metrics[key] !== undefined,
  );
}

function availableSnapshotShares(
  candidatePopulation: SelectorPopulationSnapshot,
  selectedPopulation: SelectorPopulationSnapshot,
) {
  return SNAPSHOT_SHARE_ORDER.filter(
    ({ key }) => candidatePopulation.shares[key] !== undefined || selectedPopulation.shares[key] !== undefined,
  );
}

function classificationTone(value: string): string {
  if (value === 'fresh_selector') {
    return 'text-emerald-200';
  }
  if (value === 'seeded_claimed') {
    return 'text-yellow-200';
  }
  return 'text-muted';
}

function prettyClassification(value: string): string {
  if (value === 'fresh_selector') {
    return 'fresh selector';
  }
  if (value === 'seeded_claimed') {
    return 'seeded/claimed';
  }
  return 'unknown';
}

function statusTone(status: string): string {
  if (status === 'failed') {
    return 'text-danger';
  }
  if (status === 'retry') {
    return 'text-yellow-200';
  }
  if (status === 'completed') {
    return 'text-success';
  }
  return 'text-text';
}

export function SelectorStatsPage() {
  const [selectedWindow, setSelectedWindow] = useState<SelectorWindow>('24h');
  const [selectedScope, setSelectedScope] = useState<NerdMetricsScope>('global');
  const [selectedWorkspaceId, setSelectedWorkspaceId] = useState('');
  const selectedWindowConfig = SELECTOR_WINDOW_OPTIONS[selectedWindow];
  const metricKindByKey = Object.fromEntries(
    SNAPSHOT_METRIC_ORDER.map((metric) => [metric.key, metric.kind]),
  ) as Record<string, 'number' | 'age'>;

  const workspaceOptionsQuery = useQuery({
    queryKey: ['selector-stats', 'workspace-options', selectedWindow],
    queryFn: () => fetchNerdMetrics({ ...selectedWindowConfig, scope: 'global', bucket_minutes: 60 }),
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

  const selectorQuery = useQuery({
    queryKey: ['selector-stats', selectedWindow, selectedScope, selectedWorkspaceId],
    queryFn: () => fetchSelectorStats({
      window_hours: selectedWindowConfig.window_hours,
      limit: 200,
      scope: selectedScope,
      workspace_id: selectedScope === 'workspace' ? selectedWorkspaceId : undefined,
    }),
    enabled: selectedScope === 'global' || Boolean(selectedWorkspaceId),
    refetchInterval: 10000,
    retry: false,
  });

  function handleWorkspaceChange(workspaceId: string) {
    if (!workspaceId) {
      setSelectedScope('global');
      setSelectedWorkspaceId('');
      return;
    }
    setSelectedWorkspaceId(workspaceId);
    setSelectedScope('workspace');
  }

  if (selectorQuery.isPending) {
    return <section className="panel p-4 text-xs text-muted">Loading selector stats…</section>;
  }

  if (selectorQuery.isError || !selectorQuery.data) {
    const message = selectorQuery.error instanceof ApiError && selectorQuery.error.status === 404
      ? 'Unable to load selector stats because the running daemon does not know about `/api/selector-stats` yet. Restart the daemon, then refresh this page.'
      : 'Unable to load selector stats.';
    return <section className="panel p-4 text-xs text-danger">{message}</section>;
  }

  const { summary, classification_breakdown, outcome_rows, feature_rollup_rows, recent_runs } = selectorQuery.data;
  const hasActivity = summary.total_runs > 0;

  return (
    <div className="space-y-6">
      <section className="space-y-3">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <p className="panel-title">Selector telemetry</p>
            <h1 className="mt-1 text-base font-semibold text-text">Selector Stats</h1>
            <p className="mt-1 max-w-3xl text-xs text-muted">
              This page combines recent-run diagnostics with cross-run rollups from persisted selection-time snapshots.
              Score inputs shown below come from what the selector captured at run time, not from today’s mutable memory
              state.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 self-start">
            <section className="panel flex items-center gap-1 p-1">
              <span className="px-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted">Window</span>
              {SELECTOR_WINDOW_ORDER.map((windowKey) => {
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
                    {SELECTOR_WINDOW_OPTIONS[windowKey].label}
                  </button>
                );
              })}
            </section>

            <WorkspaceScopeSelector
              selectedScope={selectedScope}
              selectedWorkspaceId={selectedWorkspaceId}
              workspaceOptions={workspaceOptions}
              workspaceOptionsPending={workspaceOptionsQuery.isPending}
              workspaceOptionsError={workspaceOptionsQuery.isError}
              onSelectGlobal={() => setSelectedScope('global')}
              onSelectWorkspace={handleWorkspaceChange}
            />
          </div>
        </div>

        <section className="grid gap-2 sm:grid-cols-2 xl:grid-cols-6">
          <article className="metric-card">
            <p className="panel-title">Runs in window</p>
            <p className="mt-1 text-lg font-semibold text-text">{summary.total_runs}</p>
          </article>
          <article className="metric-card">
            <p className="panel-title">Selector-signal runs</p>
            <p className="mt-1 text-lg font-semibold text-text">{summary.selector_signal_runs}</p>
          </article>
          <article className="metric-card">
            <p className="panel-title">Fresh selector</p>
            <p className="mt-1 text-lg font-semibold text-text">{summary.fresh_selector_runs}</p>
          </article>
          <article className="metric-card">
            <p className="panel-title">Seeded/claimed</p>
            <p className="mt-1 text-lg font-semibold text-text">{summary.seeded_claimed_runs}</p>
          </article>
          <article className="metric-card">
            <p className="panel-title">Unknown split</p>
            <p className="mt-1 text-lg font-semibold text-text">{summary.unknown_runs}</p>
          </article>
          <article className="metric-card">
            <p className="panel-title">Avg candidates</p>
            <p className="mt-1 text-lg font-semibold text-text">{formatRatio(summary.average_candidate_count)}</p>
          </article>
        </section>
      </section>

      {!hasActivity ? (
        <section className="panel p-4 text-xs text-muted">
          <p className="panel-title">Quiet window</p>
          <div className="mt-2 space-y-2">
            <p>No selector runs landed in this window yet, so the cards and tables stay intentionally sparse instead of guessing.</p>
            <p>Useful next steps: widen the window, keep the daemon running, or trigger a maintenance task that emits selector telemetry.</p>
          </div>
        </section>
      ) : null}

      <section className="grid gap-4 xl:grid-cols-[0.7fr_1.3fr]">
        <section className="table-shell">
          <div className="border-b border-border px-3 py-2">
            <p className="panel-title">Fresh vs seeded split</p>
            <h2 className="mt-1 text-base font-semibold text-text">Current classification coverage</h2>
          </div>
          <table>
            <thead>
              <tr>
                <th>Bucket</th>
                <th>Runs</th>
              </tr>
            </thead>
            <tbody>
              {classification_breakdown.map((row) => (
                <tr key={row.key}>
                  <td>{row.label}</td>
                  <td>{row.runs}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section className="panel p-4 text-xs text-muted">
          <p className="panel-title">Operator note</p>
          <div className="mt-2 space-y-2">
            <p>
              Fresh-vs-seeded classification is deliberately conservative. Runs with <code>claimed_work_item_count &gt; 0</code>
              are labeled <span className="text-text">seeded/claimed</span>; runs with <code>claimed_work_item_count = 0</code>
              are labeled <span className="text-text">fresh selector</span>; missing claim telemetry stays <span className="text-text">unknown</span>.
            </p>
            <p>
              That means seeded continuation or claimed-item work can be separated from likely frontier selection behavior now,
              without pretending certainty where current telemetry is silent.
            </p>
            <p>
              Cross-run rollups use only persisted snapshot means and shares. The page intentionally avoids averaging stored
              p50/p90 values across runs because that would look precise while being semantically sloppy.
            </p>
          </div>
        </section>
      </section>

      <section className="table-shell">
        <div className="border-b border-border px-3 py-2">
          <p className="panel-title">Cross-run feature rollups</p>
          <h2 className="mt-1 text-base font-semibold text-text">By task and freshness split</h2>
          <p className="mt-1 max-w-4xl text-xs text-muted">
            Each row averages persisted candidate-vs-selected snapshot means and share flags across runs in this window.
            Blank cells mean the selector did not persist that dimension for any run in the group.
          </p>
        </div>
        <table>
          <thead>
            <tr>
              <th>Task</th>
              <th>Classification</th>
              <th>Snapshot coverage</th>
              <th>Candidate means</th>
              <th>Selected means</th>
              <th>Share flags</th>
              <th>Avg strategy signals</th>
            </tr>
          </thead>
          <tbody>
            {feature_rollup_rows.length ? (
              feature_rollup_rows.map((row) => {
                const candidateMetricRows = formatRollupMetricSummary(row.candidate_metric_means, metricKindByKey);
                const selectedMetricRows = formatRollupMetricSummary(row.selected_metric_means, metricKindByKey);
                const shareRows = formatRollupShareSummary(row);
                return (
                  <tr key={`${row.task_name}-${row.run_classification}`}>
                    <td>
                      <div className="font-medium text-text">{row.task_name}</div>
                    </td>
                    <td className={classificationTone(row.run_classification)}>{prettyClassification(row.run_classification)}</td>
                    <td>
                      <div className="text-text">{row.snapshot_runs}/{row.runs} runs</div>
                      <div className="mt-1 text-[11px] text-muted">snapshot-bearing / total</div>
                    </td>
                    <td>
                      {candidateMetricRows.length ? (
                        <div className="space-y-1 text-[11px] text-muted">
                          {candidateMetricRows.map((metric) => (
                            <div key={`${row.task_name}-${row.run_classification}-candidate-${metric.key}`}>
                              <span className="text-text">{metric.label}:</span> {metric.value}
                            </div>
                          ))}
                        </div>
                      ) : '—'}
                    </td>
                    <td>
                      {selectedMetricRows.length ? (
                        <div className="space-y-1 text-[11px] text-muted">
                          {selectedMetricRows.map((metric) => (
                            <div key={`${row.task_name}-${row.run_classification}-selected-${metric.key}`}>
                              <span className="text-text">{metric.label}:</span> {metric.value}
                            </div>
                          ))}
                        </div>
                      ) : '—'}
                    </td>
                    <td>
                      {shareRows.length ? (
                        <div className="space-y-1 text-[11px] text-muted">
                          {shareRows.map((share) => (
                            <div key={`${row.task_name}-${row.run_classification}-share-${share.key}`}>
                              <span className="text-text">{share.label}:</span> cand {share.candidate} · sel {share.selected}
                            </div>
                          ))}
                        </div>
                      ) : '—'}
                    </td>
                    <td className="max-w-[18rem] text-[11px] text-muted" title={formatScores(row.strategy_signal_means)}>
                      {formatScores(row.strategy_signal_means)}
                    </td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan={7} className="text-xs text-muted">
                  No persisted selector feature snapshots landed in the selected window yet. Try a wider window or wait for the next selector-bearing run.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </section>

      <section className="space-y-3">
        <div>
          <p className="panel-title">Recent run snapshots</p>
          <h2 className="mt-1 text-base font-semibold text-text">Recent candidate vs selected feature summaries</h2>
          <p className="mt-1 max-w-4xl text-xs text-muted">
            These are individual runs, shown separately from the cross-run rollups above so operators can inspect specific score inputs without losing the broader pattern.
          </p>
        </div>
        <div className="grid gap-4 xl:grid-cols-2">
          {recent_runs.length ? (
            recent_runs.slice(0, 6).map((run) => {
              const candidatePopulation = run.selector_feature_snapshot.candidate_population;
              const selectedPopulation = run.selector_feature_snapshot.selected_population;
              const metricRows = availableSnapshotMetrics(candidatePopulation, selectedPopulation);
              const shareRows = availableSnapshotShares(candidatePopulation, selectedPopulation);
              return (
                <article key={`snapshot-${run.task_id ?? run.task_name}-${run.completed_at}`} className="panel p-4">
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <div>
                      <p className="panel-title">{run.task_name}</p>
                      <h3 className="mt-1 text-sm font-semibold text-text">
                        {run.strategy_used ?? run.requested_strategy ?? 'unspecified strategy'}
                      </h3>
                      <p className="mt-1 text-[11px] text-muted">
                        {formatTimestamp(run.completed_at)} · {prettyClassification(run.run_classification)} · {run.status}
                      </p>
                    </div>
                    <div className="text-right text-[11px] text-muted">
                      <div>candidate count: {candidatePopulation.count}</div>
                      <div>selected count: {selectedPopulation.count}</div>
                    </div>
                  </div>

                  <div className="mt-3 space-y-3 text-xs text-muted">
                    <div>
                      <p className="panel-title">Strategy signals</p>
                      <p className="mt-1">{formatScores(run.selector_feature_snapshot.strategy_signals)}</p>
                    </div>

                    <div className="overflow-x-auto">
                      <table>
                        <thead>
                          <tr>
                            <th>Metric</th>
                            <th>Candidate population</th>
                            <th>Selected population</th>
                          </tr>
                        </thead>
                        <tbody>
                          {metricRows.length ? (
                            metricRows.map((metric) => (
                              <tr key={`${run.task_id ?? run.task_name}-${metric.key}`}>
                                <td>{metric.label}</td>
                                <td>{formatMetricSummary(candidatePopulation.metrics[metric.key], metric.kind)}</td>
                                <td>{formatMetricSummary(selectedPopulation.metrics[metric.key], metric.kind)}</td>
                              </tr>
                            ))
                          ) : (
                            <tr>
                              <td colSpan={3}>No persisted metric snapshot for this run.</td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>

                    <div className="overflow-x-auto">
                      <table>
                        <thead>
                          <tr>
                            <th>Share</th>
                            <th>Candidate population</th>
                            <th>Selected population</th>
                          </tr>
                        </thead>
                        <tbody>
                          {shareRows.length ? (
                            shareRows.map((share) => (
                              <tr key={`${run.task_id ?? run.task_name}-${share.key}`}>
                                <td>{share.label}</td>
                                <td>{formatShare(candidatePopulation.shares[share.key])}</td>
                                <td>{formatShare(selectedPopulation.shares[share.key])}</td>
                              </tr>
                            ))
                          ) : (
                            <tr>
                              <td colSpan={3}>No persisted share snapshot for this run.</td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </article>
              );
            })
          ) : (
            <section className="panel p-4 text-xs text-muted">
              No recent runs landed in the selected window. Widen the window or wait for the daemon to finish a selector-bearing task.
            </section>
          )}
        </div>
      </section>

      <section className="table-shell">
        <div className="border-b border-border px-3 py-2">
          <p className="panel-title">Outcome rollups</p>
          <h2 className="mt-1 text-base font-semibold text-text">By task, strategy, mode, reason family, and freshness split</h2>
        </div>
        <table>
          <thead>
            <tr>
              <th>Task</th>
              <th>Classification</th>
              <th>Strategy</th>
              <th>Mode</th>
              <th>Reason</th>
              <th>Runs</th>
              <th>Fallbacks</th>
              <th>Mutations</th>
              <th>Tool calls</th>
              <th>Avg candidates</th>
              <th>No-op rate</th>
            </tr>
          </thead>
          <tbody>
            {outcome_rows.length ? (
              outcome_rows.map((row) => (
                <tr key={`${row.task_name}-${row.run_classification}-${row.strategy_used}-${row.strategy_selection_mode}-${row.reason_family}`}>
                  <td>{row.task_name}</td>
                  <td className={classificationTone(row.run_classification)}>{prettyClassification(row.run_classification)}</td>
                  <td>{row.strategy_used}</td>
                  <td>{row.strategy_selection_mode}</td>
                  <td>{row.reason_family}</td>
                  <td>{row.runs}</td>
                  <td>{row.fallback_count}</td>
                  <td>{row.total_mutations}</td>
                  <td>{row.total_tool_calls}</td>
                  <td>{formatRatio(row.average_candidate_count)}</td>
                  <td>{(row.no_op_rate * 100).toFixed(1)}%</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={11} className="text-xs text-muted">
                  No selector-signal runs landed in the selected window yet. Summary cards stay available so a quiet daemon still tells you what the window contains: currently, not much.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </section>

      <section className="table-shell">
        <div className="border-b border-border px-3 py-2">
          <p className="panel-title">Recent diagnostics</p>
          <h2 className="mt-1 text-base font-semibold text-text">Most recent runs with score maps and operational context</h2>
        </div>
        <table>
          <thead>
            <tr>
              <th>Completed</th>
              <th>Task</th>
              <th>Status</th>
              <th>Classification</th>
              <th>Strategy</th>
              <th>Mode</th>
              <th>Candidates</th>
              <th>Claimed</th>
              <th>Mutations</th>
              <th>Tool calls</th>
              <th>Scores</th>
            </tr>
          </thead>
          <tbody>
            {recent_runs.length ? (
              recent_runs.map((run) => (
                <tr key={`${run.task_id ?? run.task_name}-${run.completed_at}`}>
                  <td className="whitespace-nowrap text-xs text-muted">{formatTimestamp(run.completed_at)}</td>
                  <td>
                    <div className="font-medium text-text">{run.task_name}</div>
                    <div className="mt-1 max-w-[26rem] truncate text-[11px] text-muted" title={run.result_summary ?? run.classification_reason}>
                      {run.result_summary ?? run.classification_reason}
                    </div>
                  </td>
                  <td className={statusTone(run.status)}>{run.status}</td>
                  <td className={classificationTone(run.run_classification)}>
                    <div>{prettyClassification(run.run_classification)}</div>
                    <div className="mt-1 text-[11px] text-muted">{run.classification_reason}</div>
                  </td>
                  <td>{run.strategy_used ?? run.requested_strategy ?? '—'}</td>
                  <td>
                    <div>{run.strategy_selection_mode ?? '—'}</div>
                    <div className="mt-1 max-w-[18rem] truncate text-[11px] text-muted" title={run.strategy_selection_reason ?? run.strategy_fallback_reason ?? ''}>
                      {run.strategy_selection_reason ?? run.strategy_fallback_reason ?? '—'}
                    </div>
                  </td>
                  <td>{run.candidate_count ?? '—'}</td>
                  <td>{run.claimed_work_item_count ?? '—'}</td>
                  <td>{run.mutations ?? '—'}</td>
                  <td>{run.tool_calls_executed ?? '—'}</td>
                  <td className="max-w-[22rem] text-[11px] text-muted" title={formatScores(run.strategy_selection_scores)}>
                    {formatScores(run.strategy_selection_scores)}
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={11} className="text-xs text-muted">
                  No recent runs in the selected window. If that seems wrong, restart the daemon or widen the time window before assuming telemetry broke.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </section>
    </div>
  );
}