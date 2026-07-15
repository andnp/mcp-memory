import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import {
  ApiError,
  fetchMutationHistory,
  fetchMutationHistoryDetail,
  fetchMutationHistoryDiff,
  fetchNerdMetrics,
  fetchRestoreEligibility,
  removeProtection,
  requestRestore,
  setProtection,
  type LinkHistoryRevision,
  type MutationHistoryEvent,
  type RecordHistoryRevision,
} from '../lib/api';

const MAX_VISIBLE_RECORDS = 12;
const MAX_VISIBLE_LINKS = 20;
const MAX_VISIBLE_FIELDS = 12;
const MAX_VALUE_CHARS = 1_600;

const PROTECTION_MODES = [
  'no_autonomous_mutation',
  'no_autonomous_destructive_change',
  'manual_review_required',
  'local_provider_only',
  'no_external_provider_disclosure',
  'pinned_active',
];

function formatTimestamp(value: string | null | undefined): string {
  if (!value) {
    return '—';
  }
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString();
}

function formatCountMap(values: Record<string, number> | undefined): string {
  if (!values) {
    return '—';
  }
  const entries = Object.entries(values);
  return entries.length ? entries.map(([key, value]) => `${key}: ${value}`).join(' · ') : '—';
}

function statusTone(value: string | null | undefined): string {
  if (value === 'failed' || value === 'rejected' || value === 'conflict') {
    return 'text-danger';
  }
  if (value === 'verified' || value === 'applied' || value === 'terminal') {
    return 'text-success';
  }
  if (value === 'stale' || value === 'verification_failed') {
    return 'text-yellow-200';
  }
  return 'text-text';
}

function boundedValue(value: unknown): string {
  if (value === null || value === undefined) {
    return '—';
  }
  if (typeof value === 'string') {
    return value.length > MAX_VALUE_CHARS ? `${value.slice(0, MAX_VALUE_CHARS)}…` : value;
  }
  const encoded = JSON.stringify(value, null, 2);
  if (!encoded) {
    return String(value);
  }
  return encoded.length > MAX_VALUE_CHARS ? `${encoded.slice(0, MAX_VALUE_CHARS)}…` : encoded;
}

function snapshotMap(value: unknown): Record<string, unknown> | null {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return null;
  }
  const record = value as Record<string, unknown>;
  return record.truncated === true ? null : record;
}

function changedFields(before: unknown, after: unknown): string[] {
  const beforeMap = snapshotMap(before);
  const afterMap = snapshotMap(after);
  if (!beforeMap || !afterMap) {
    return [];
  }
  return Array.from(new Set([...Object.keys(beforeMap), ...Object.keys(afterMap)]))
    .filter((key) => JSON.stringify(beforeMap[key]) !== JSON.stringify(afterMap[key]))
    .sort()
    .slice(0, MAX_VISIBLE_FIELDS);
}

function prettyMode(mode: string): string {
  return mode.split('_').join(' ');
}

function eventLabel(event: MutationHistoryEvent): string {
  return `${event.operation} · ${event.id.slice(0, 8)}`;
}

function RecordDiff({ record }: { record: RecordHistoryRevision }) {
  const before = record.before ?? record.before_snapshot;
  const after = record.after ?? record.after_snapshot;
  const fields = changedFields(before, after);
  const beforeMap = snapshotMap(before);
  const afterMap = snapshotMap(after);

  return (
    <article className="rounded-lg border border-border bg-ink/60 p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <p className="text-xs font-semibold text-text">{record.memory_id}</p>
          <p className="mt-1 text-[10px] uppercase tracking-[0.14em] text-muted">{record.role}</p>
        </div>
        <span className="text-[10px] text-muted">
          {record.before_exists ? 'before' : 'created'} → {record.after_exists ? 'after' : 'removed'}
        </span>
      </div>
      {fields.length ? (
        <div className="mt-3 space-y-2">
          {fields.map((field) => (
            <div key={field} className="grid gap-2 sm:grid-cols-2">
              <div className="min-w-0 rounded border border-border/70 bg-panel/60 p-2">
                <p className="text-[10px] uppercase tracking-[0.12em] text-muted">{field} · before</p>
                <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap break-words text-[11px] text-text">
                  {boundedValue(beforeMap?.[field])}
                </pre>
              </div>
              <div className="min-w-0 rounded border border-border/70 bg-panel/60 p-2">
                <p className="text-[10px] uppercase tracking-[0.12em] text-muted">{field} · after</p>
                <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap break-words text-[11px] text-text">
                  {boundedValue(afterMap?.[field])}
                </pre>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="mt-3 grid gap-2 sm:grid-cols-2">
          <div>
            <p className="text-[10px] uppercase tracking-[0.12em] text-muted">Before · bounded</p>
            <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-words text-[11px] text-text">{boundedValue(before)}</pre>
          </div>
          <div>
            <p className="text-[10px] uppercase tracking-[0.12em] text-muted">After · bounded</p>
            <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-words text-[11px] text-text">{boundedValue(after)}</pre>
          </div>
        </div>
      )}
    </article>
  );
}

function LinkDiff({ link }: { link: LinkHistoryRevision }) {
  return (
    <div className="grid gap-2 rounded-lg border border-border/70 bg-ink/50 p-2 text-[11px] sm:grid-cols-[1fr_auto_1fr] sm:items-center">
      <span className="min-w-0 break-all">{link.source_id}</span>
      <span className="text-muted">{link.link_type}</span>
      <span className="min-w-0 break-all">{link.target_id}</span>
      <span className="text-muted sm:col-span-3">
        {link.before_exists ? `before: ${boundedValue(link.before_context ?? link.context)}` : 'before: absent'}
        {' · '}
        {link.after_exists ? `after: ${boundedValue(link.after_context ?? link.context)}` : 'after: absent'}
      </span>
    </div>
  );
}

export function CurationAuditPage() {
  const queryClient = useQueryClient();
  const [operation, setOperation] = useState('');
  const [family, setFamily] = useState('');
  const [offset, setOffset] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [restoreReason, setRestoreReason] = useState('Restore after operator review');
  const [restoreArmed, setRestoreArmed] = useState(false);
  const [protectionMemoryId, setProtectionMemoryId] = useState('');
  const [protectionMode, setProtectionMode] = useState(PROTECTION_MODES[0]);
  const [protectionReason, setProtectionReason] = useState('Operator protection from curation audit');
  const [pendingProtection, setPendingProtection] = useState<'add' | 'remove' | null>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);

  const metricsQuery = useQuery({
    queryKey: ['curation-metrics'],
    queryFn: () => fetchNerdMetrics({ window_hours: 24, bucket_minutes: 60 }),
    refetchInterval: 30000,
    refetchOnWindowFocus: false,
  });
  const historyQuery = useQuery({
    queryKey: ['curation-history', operation, family, offset],
    queryFn: () => fetchMutationHistory({ limit: 50, offset, operation: operation || undefined, family: family || undefined }),
    refetchInterval: 15000,
    refetchOnWindowFocus: false,
  });
  const detailQuery = useQuery({
    queryKey: ['curation-history-detail', selectedId],
    queryFn: () => fetchMutationHistoryDetail(selectedId as string),
    enabled: selectedId !== null,
    refetchOnWindowFocus: false,
  });
  const diffQuery = useQuery({
    queryKey: ['curation-history-diff', selectedId],
    queryFn: () => fetchMutationHistoryDiff(selectedId as string),
    enabled: selectedId !== null,
    refetchOnWindowFocus: false,
  });
  const eligibilityQuery = useQuery({
    queryKey: ['curation-restore-eligibility', selectedId],
    queryFn: () => fetchRestoreEligibility(selectedId as string),
    enabled: selectedId !== null,
    refetchOnWindowFocus: false,
  });

  const selectedEvent = useMemo(
    () => historyQuery.data?.events.find((event) => event.id === selectedId) ?? historyQuery.data?.events[0] ?? null,
    [historyQuery.data?.events, selectedId],
  );
  const selectedDiff = diffQuery.data;
  const selectedRecords = selectedDiff?.records ?? detailQuery.data?.records ?? [];
  const selectedMemoryIds = useMemo(
    () => Array.from(new Set(selectedRecords.map((record) => record.memory_id))).slice(0, MAX_VISIBLE_RECORDS),
    [selectedRecords],
  );
  const curation = metricsQuery.data?.curation;
  const specialistRoutes = curation?.specialist_routes;
  const protections = eligibilityQuery.data?.protections ?? {};
  const selectedProtections = protectionMemoryId ? protections[protectionMemoryId] ?? [] : [];
  const canRestore = Boolean(
    eligibilityQuery.data
    && eligibilityQuery.data.risk === 'low'
    && (eligibilityQuery.data.eligible || eligibilityQuery.data.conflict_code === 'confirmation_required')
    && restoreReason.trim(),
  );

  useEffect(() => {
    const events = historyQuery.data?.events ?? [];
    if (!events.length) {
      setSelectedId(null);
      return;
    }
    if (!selectedId || !events.some((event) => event.id === selectedId)) {
      setSelectedId(events[0].id);
    }
  }, [historyQuery.data?.events, selectedId]);

  useEffect(() => {
    if (!selectedMemoryIds.length) {
      setProtectionMemoryId('');
      return;
    }
    if (!selectedMemoryIds.includes(protectionMemoryId)) {
      setProtectionMemoryId(selectedMemoryIds[0]);
    }
  }, [protectionMemoryId, selectedMemoryIds]);

  const restoreMutation = useMutation({
    mutationFn: () => requestRestore({
      event_id: selectedId as string,
      expected_record_tokens: eligibilityQuery.data?.current_record_tokens ?? {},
      expected_link_tokens: eligibilityQuery.data?.current_link_tokens ?? {},
      reason: restoreReason.trim(),
      idempotency_key: `dashboard-restore-${selectedId}`,
      confirmation: true,
    }),
    onSuccess: (result) => {
      setRestoreArmed(false);
      setActionMessage(`Restore request: ${result.status}.`);
      void queryClient.invalidateQueries({ queryKey: ['curation-history'] });
      void queryClient.invalidateQueries({ queryKey: ['curation-history-detail', selectedId] });
      void queryClient.invalidateQueries({ queryKey: ['curation-history-diff', selectedId] });
      void queryClient.invalidateQueries({ queryKey: ['curation-restore-eligibility', selectedId] });
      void queryClient.invalidateQueries({ queryKey: ['curation-metrics'] });
    },
    onError: () => setActionMessage('Restore request failed; refresh eligibility before retrying.'),
  });

  const addProtectionMutation = useMutation({
    mutationFn: () => setProtection({
      memory_id: protectionMemoryId,
      mode: protectionMode,
      reason: protectionReason.trim(),
      actor_id: 'management-dashboard',
    }),
    onSuccess: () => {
      setPendingProtection(null);
      setActionMessage('Protection applied.');
      void queryClient.invalidateQueries({ queryKey: ['curation-restore-eligibility', selectedId] });
    },
    onError: () => setActionMessage('Protection change failed.'),
  });

  const removeProtectionMutation = useMutation({
    mutationFn: () => removeProtection({ memory_id: protectionMemoryId, mode: protectionMode }),
    onSuccess: () => {
      setPendingProtection(null);
      setActionMessage('Protection removed.');
      void queryClient.invalidateQueries({ queryKey: ['curation-restore-eligibility', selectedId] });
    },
    onError: () => setActionMessage('Protection change failed.'),
  });

  if (historyQuery.isError) {
    const message = historyQuery.error instanceof ApiError && historyQuery.error.status === 404
      ? 'The running daemon does not expose mutation history yet. Restart it, then refresh this page.'
      : 'Unable to load curation history.';
    return <section className="panel p-4 text-xs text-danger">{message}</section>;
  }

  const events = historyQuery.data?.events ?? [];
  const visibleRecords = selectedRecords.slice(0, MAX_VISIBLE_RECORDS);
  const visibleLinks = (selectedDiff?.links ?? detailQuery.data?.links ?? []).slice(0, MAX_VISIBLE_LINKS);
  const run = detailQuery.data?.curation_run;
  const receipt = detailQuery.data?.receipt ?? selectedEvent?.receipt;

  function stageRestore() {
    if (!canRestore) {
      setActionMessage('This event is not currently eligible for a low-risk restore.');
      return;
    }
    setActionMessage(null);
    setRestoreArmed(true);
  }

  function stageProtectionChange(kind: 'add' | 'remove') {
    if (!protectionMemoryId || (kind === 'add' && !protectionReason.trim())) {
      setActionMessage('Choose a memory and provide a protection reason first.');
      return;
    }
    setActionMessage(null);
    setPendingProtection(kind);
  }

  function confirmProtectionChange() {
    if (pendingProtection === 'add') {
      addProtectionMutation.mutate();
    } else if (pendingProtection === 'remove') {
      removeProtectionMutation.mutate();
    }
  }

  return (
    <div className="space-y-6">
      <section className="space-y-3">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <p className="panel-title">Curation audit</p>
            <h1 className="mt-1 text-base font-semibold text-text">History, protection, and safe restore</h1>
            <p className="mt-1 max-w-3xl text-xs text-muted">
              Durable runs and mutation history are authoritative here. Content diffs, lists, and controls stay bounded;
              provider prose is not presented as audit truth.
            </p>
          </div>
          <span className="rounded-full border border-border bg-panel px-3 py-1 text-[10px] uppercase tracking-[0.14em] text-muted">24h metrics</span>
        </div>
        <section className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
          <article className="metric-card"><p className="panel-title">Run states</p><p className="mt-1 text-sm text-text">{formatCountMap(curation?.run_states)}</p></article>
          <article className="metric-card"><p className="panel-title">Applied history</p><p className="mt-1 text-lg font-semibold text-text">{curation?.history.applied_event_count ?? '—'}</p></article>
          <article className="metric-card"><p className="panel-title">Restorable events</p><p className="mt-1 text-lg font-semibold text-text">{curation?.history.restorable_event_count ?? '—'}</p></article>
          <article className="metric-card"><p className="panel-title">Verified yield</p><p className="mt-1 text-lg font-semibold text-text">{curation ? `${(curation.verified_yield * 100).toFixed(1)}%` : '—'}</p></article>
          <article className="metric-card"><p className="panel-title">Specialist routes</p><p className="mt-1 text-lg font-semibold text-text">{specialistRoutes?.total ?? '—'}</p></article>
          <article className="metric-card"><p className="panel-title">No-op runs</p><p className="mt-1 text-lg font-semibold text-text">{curation?.no_op_runs ?? '—'}</p></article>
        </section>
      </section>

      <section className="panel p-3">
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-[1fr_1fr_auto]">
          <select value={operation} onChange={(event) => { setOperation(event.target.value); setOffset(0); }} className="rounded-lg border border-border bg-ink px-3 py-2 text-xs text-text outline-none focus:border-accent">
            <option value="">all operations</option>
            {Array.from(new Set(events.map((event) => event.operation))).map((value) => <option key={value} value={value}>{value}</option>)}
          </select>
          <input value={family} onChange={(event) => { setFamily(event.target.value); setOffset(0); }} className="rounded-lg border border-border bg-ink px-3 py-2 text-xs text-text outline-none focus:border-accent" placeholder="filter family (e.g. curator)" />
          <button type="button" onClick={() => void historyQuery.refetch()} className="rounded-lg border border-border px-3 py-2 text-xs text-muted hover:border-accent hover:text-text">Refresh history</button>
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-[0.85fr_1.15fr]">
        <section className="table-shell">
          <div className="border-b border-border px-4 py-4">
            <p className="panel-title">Mutation history</p>
            <h2 className="mt-2 text-lg font-semibold text-text">Select an event to inspect</h2>
          </div>
          <table>
            <thead><tr><th>Operation</th><th>Status</th><th>Family</th><th>Created</th></tr></thead>
            <tbody>
              {historyQuery.isPending ? <tr><td colSpan={4} className="text-muted">Loading bounded history…</td></tr> : null}
              {!historyQuery.isPending && !events.length ? <tr><td colSpan={4} className="text-muted">No curation mutation events match these filters.</td></tr> : null}
              {events.map((event) => (
                <tr key={event.id} className={selectedEvent?.id === event.id ? 'bg-ink/80' : ''} onClick={() => setSelectedId(event.id)}>
                  <td className="max-w-[14rem] truncate" title={event.operation}>{event.operation}<div className="mt-1 text-[10px] text-muted">{event.id.slice(0, 12)}</div></td>
                  <td className={statusTone(event.status)}>{event.status}<div className="mt-1 text-[10px] text-muted">{event.receipt?.status ?? 'no receipt'}</div></td>
                  <td>{event.family ?? '—'}</td>
                  <td className="whitespace-nowrap text-[11px]">{formatTimestamp(event.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border px-3 py-2 text-xs text-muted">
            <span>Showing up to {historyQuery.data?.limit ?? 50} events · offset {historyQuery.data?.offset ?? offset}</span>
            <div className="flex gap-2">
              <button type="button" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 50))} className="rounded border border-border px-2 py-1 disabled:cursor-not-allowed disabled:opacity-40">Previous</button>
              <button type="button" disabled={!historyQuery.data?.has_more} onClick={() => setOffset(historyQuery.data?.next_offset ?? offset)} className="rounded border border-border px-2 py-1 disabled:cursor-not-allowed disabled:opacity-40">Next</button>
            </div>
          </div>
        </section>

        <div className="space-y-6">
          <section className="panel p-4">
            <p className="panel-title">Selected event</p>
            <h2 className="mt-2 break-words text-lg font-semibold text-text">{selectedEvent ? eventLabel(selectedEvent) : 'No event selected'}</h2>
            {selectedEvent ? (
              <div className="mt-3 grid gap-2 sm:grid-cols-2">
                <div className="rounded-lg border border-border bg-ink/60 p-2.5"><p className="panel-title">Event status</p><p className={`mt-1 ${statusTone(selectedEvent.status)}`}>{selectedEvent.status}</p></div>
                <div className="rounded-lg border border-border bg-ink/60 p-2.5"><p className="panel-title">Receipt</p><p className={`mt-1 ${statusTone(receipt?.status)}`}>{receipt?.status ?? 'not linked'}</p></div>
                <div className="rounded-lg border border-border bg-ink/60 p-2.5"><p className="panel-title">Actor / family</p><p className="mt-1 text-xs text-text">{selectedEvent.actor_kind} · {selectedEvent.family ?? '—'}</p></div>
                <div className="rounded-lg border border-border bg-ink/60 p-2.5"><p className="panel-title">Provider</p><p className="mt-1 break-words text-xs text-text">{selectedEvent.provider_id ?? 'not disclosed'}</p></div>
              </div>
            ) : <p className="mt-3 text-xs text-muted">Choose an event from the history table.</p>}
          </section>

          {selectedEvent ? (
            <>
              <section className="panel p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div><p className="panel-title">Curation run and receipt</p><h2 className="mt-1 text-base font-semibold text-text">Execution state</h2></div>
                  <span className="text-[10px] text-muted">{formatTimestamp(run?.created_at ?? receipt?.applied_at)}</span>
                </div>
                <div className="mt-3 grid gap-2 sm:grid-cols-2">
                  <div className="rounded-lg border border-border bg-ink/60 p-2.5"><p className="panel-title">Run state</p><p className={`mt-1 ${statusTone(run?.state)}`}>{run?.state ?? 'run unavailable'}</p><p className="mt-1 text-[10px] text-muted">{run?.outcome ?? '—'}</p></div>
                  <div className="rounded-lg border border-border bg-ink/60 p-2.5"><p className="panel-title">Action</p><p className="mt-1 break-words text-xs text-text">{receipt ? `${receipt.operation} · ${receipt.action_id.slice(0, 12)}` : '—'}</p><p className="mt-1 text-[10px] text-muted">{receipt?.affected_ids.length ?? 0} affected record(s)</p></div>
                </div>
                {run?.rejection_codes?.length ? <p className="mt-3 text-xs text-yellow-200">Rejections: {run.rejection_codes.join(', ')}</p> : null}
              </section>

              <section className="panel p-4">
                <div className="flex flex-wrap items-start justify-between gap-3"><div><p className="panel-title">Specialist routing</p><h2 className="mt-1 text-base font-semibold text-text">Ownership and route evidence</h2></div><span className="text-xs text-text">Selected family: {selectedEvent.family ?? '—'}</span></div>
                <div className="mt-3 space-y-2 text-xs text-muted">
                  <p>Total routes: <span className="text-text">{specialistRoutes?.total ?? 0}</span> · by family: <span className="text-text">{formatCountMap(specialistRoutes?.by_family)}</span></p>
                  <p>Reasons: <span className="text-text">{formatCountMap(specialistRoutes?.by_reason)}</span></p>
                  <p>Status: <span className="text-text">{formatCountMap(specialistRoutes?.by_status)}</span></p>
                </div>
              </section>

              <section className="panel p-4">
                <div className="flex flex-wrap items-start justify-between gap-3"><div><p className="panel-title">Before / after</p><h2 className="mt-1 text-base font-semibold text-text">Bounded mutation diff</h2></div><span className="text-[10px] text-muted">{selectedDiff?.truncated || detailQuery.data?.truncated ? 'server truncated' : 'bounded display'}</span></div>
                {diffQuery.isPending ? <p className="mt-3 text-xs text-muted">Loading diff…</p> : null}
                {diffQuery.isError ? <p className="mt-3 text-xs text-danger">Unable to load the authoritative diff.</p> : null}
                <div className="mt-3 space-y-3">{visibleRecords.map((record) => <RecordDiff key={`${record.memory_id}-${record.role}`} record={record} />)}</div>
                {selectedRecords.length > MAX_VISIBLE_RECORDS ? <p className="mt-3 text-xs text-muted">Only the first {MAX_VISIBLE_RECORDS} record revisions are rendered.</p> : null}
                {visibleLinks.length ? <div className="mt-4 space-y-2"><p className="panel-title">Relationship changes</p>{visibleLinks.map((link) => <LinkDiff key={`${link.source_id}-${link.target_id}-${link.link_type}`} link={link} />)}</div> : null}
                {(selectedDiff?.links.length ?? 0) > MAX_VISIBLE_LINKS ? <p className="mt-3 text-xs text-muted">Only the first {MAX_VISIBLE_LINKS} relationship revisions are rendered.</p> : null}
                {!visibleRecords.length && !visibleLinks.length ? <p className="mt-3 text-xs text-muted">No revision rows were returned for this event.</p> : null}
              </section>

              <section className="panel p-4">
                <div className="flex flex-wrap items-start justify-between gap-3"><div><p className="panel-title">Protection and restore</p><h2 className="mt-1 text-base font-semibold text-text">Operator controls</h2></div><span className={`text-xs ${statusTone(eligibilityQuery.data?.conflict_code)}`}>{eligibilityQuery.data?.conflict_code ?? (eligibilityQuery.data?.eligible ? 'eligible' : 'checking')}</span></div>
                {eligibilityQuery.isPending ? <p className="mt-3 text-xs text-muted">Checking current tokens and protection state…</p> : null}
                {eligibilityQuery.isError ? <p className="mt-3 text-xs text-danger">Unable to check restore eligibility.</p> : null}
                {eligibilityQuery.data ? <>
                  <div className="mt-3 grid gap-2 sm:grid-cols-2"><div className="rounded-lg border border-border bg-ink/60 p-2.5"><p className="panel-title">Restore</p><p className="mt-1 text-xs text-text">{eligibilityQuery.data.inverse_operation ?? 'unsupported'} · risk {eligibilityQuery.data.risk ?? 'unknown'}</p><p className="mt-1 text-[10px] text-muted">{eligibilityQuery.data.conflict_reason ?? 'Current revision tokens match.'}</p></div><div className="rounded-lg border border-border bg-ink/60 p-2.5"><p className="panel-title">Current protections</p><p className="mt-1 text-xs text-text">{Object.values(protections).reduce((total, values) => total + values.length, 0)} protection(s) across {Object.keys(protections).length} record(s)</p></div></div>
                  <div className="mt-4 grid gap-3 lg:grid-cols-[1fr_1fr]">
                    <div className="rounded-lg border border-border bg-ink/60 p-3"><p className="panel-title">Low-risk restore</p><textarea value={restoreReason} onChange={(event) => setRestoreReason(event.target.value)} className="mt-2 min-h-20 w-full rounded border border-border bg-ink px-2 py-2 text-xs text-text outline-none focus:border-accent" placeholder="Reason for restore" />
                      {!restoreArmed ? <button type="button" disabled={!canRestore || restoreMutation.isPending} onClick={stageRestore} className="mt-2 rounded-lg border border-accent px-3 py-2 text-xs font-semibold text-accent disabled:cursor-not-allowed disabled:opacity-40">Stage restore confirmation</button> : <div className="mt-2 space-y-2"><p className="text-xs text-yellow-200">Confirming creates a new restore event and never rewrites history.</p><div className="flex flex-wrap gap-2"><button type="button" onClick={() => restoreMutation.mutate()} disabled={restoreMutation.isPending} className="rounded-lg border border-danger bg-danger/10 px-3 py-2 text-xs font-semibold text-danger disabled:opacity-40">Confirm restore</button><button type="button" onClick={() => setRestoreArmed(false)} className="rounded-lg border border-border px-3 py-2 text-xs text-muted">Cancel</button></div></div>}
                    </div>
                    <div className="rounded-lg border border-border bg-ink/60 p-3"><p className="panel-title">Protection state</p><select value={protectionMemoryId} onChange={(event) => { setProtectionMemoryId(event.target.value); setPendingProtection(null); }} className="mt-2 w-full rounded border border-border bg-ink px-2 py-2 text-xs text-text outline-none focus:border-accent"><option value="">select affected memory</option>{selectedMemoryIds.map((memoryId) => <option key={memoryId} value={memoryId}>{memoryId}</option>)}</select><p className="mt-2 text-[10px] text-muted">{selectedProtections.length ? selectedProtections.map((protection) => `${prettyMode(protection.mode)} (${protection.reason})`).join(' · ') : 'No protection rows for this record.'}</p><select value={protectionMode} onChange={(event) => setProtectionMode(event.target.value)} className="mt-2 w-full rounded border border-border bg-ink px-2 py-2 text-xs text-text outline-none focus:border-accent">{PROTECTION_MODES.map((mode) => <option key={mode} value={mode}>{prettyMode(mode)}</option>)}</select><input value={protectionReason} onChange={(event) => setProtectionReason(event.target.value)} className="mt-2 w-full rounded border border-border bg-ink px-2 py-2 text-xs text-text outline-none focus:border-accent" placeholder="Protection reason" /><div className="mt-2 flex flex-wrap gap-2"><button type="button" onClick={() => stageProtectionChange('add')} className="rounded-lg border border-accent px-3 py-2 text-xs text-accent">Stage protection</button><button type="button" disabled={!selectedProtections.some((protection) => protection.mode === protectionMode)} onClick={() => stageProtectionChange('remove')} className="rounded-lg border border-border px-3 py-2 text-xs text-muted disabled:cursor-not-allowed disabled:opacity-40">Stage removal</button></div>{pendingProtection ? <div className="mt-2 rounded border border-yellow-300/50 bg-yellow-300/5 p-2"><p className="text-xs text-yellow-200">Confirm {pendingProtection === 'add' ? 'adding' : 'removing'} this protection for {protectionMemoryId}.</p><div className="mt-2 flex gap-2"><button type="button" onClick={confirmProtectionChange} disabled={addProtectionMutation.isPending || removeProtectionMutation.isPending} className="rounded-lg border border-accent px-3 py-2 text-xs font-semibold text-accent">Confirm change</button><button type="button" onClick={() => setPendingProtection(null)} className="rounded-lg border border-border px-3 py-2 text-xs text-muted">Cancel</button></div></div> : null}</div>
                  </div>
                </> : null}
                {actionMessage ? <p className="mt-3 text-xs text-muted" role="status">{actionMessage}</p> : null}
              </section>
            </>
          ) : null}
        </div>
      </section>
    </div>
  );
}
