import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';

import { fetchAIConversations, fetchOverview } from '../lib/api';

function formatTimestamp(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleString();
}

export function ActivityPage() {
  const overviewQuery = useQuery({ queryKey: ['overview'], queryFn: fetchOverview, refetchInterval: 5000 });
  const conversationsQuery = useQuery({ queryKey: ['ai-conversations'], queryFn: () => fetchAIConversations(10), refetchInterval: 5000 });

  const [taskFilter, setTaskFilter] = useState('');
  const [selectedRunKey, setSelectedRunKey] = useState<string | null>(null);

  const filteredRuns = useMemo(() => {
    const runs = overviewQuery.data?.recent_agent_runs ?? [];
    if (!taskFilter.trim()) {
      return runs;
    }
    const query = taskFilter.trim().toLowerCase();
    return runs.filter((run) => run.task_name.toLowerCase().includes(query));
  }, [overviewQuery.data?.recent_agent_runs, taskFilter]);

  useEffect(() => {
    if (!filteredRuns.length) {
      setSelectedRunKey(null);
      return;
    }
    if (selectedRunKey !== null && filteredRuns.some((run, index) => `${run.task_name}-${run.completed_at}-${index}` === selectedRunKey)) {
      return;
    }
    setSelectedRunKey(`${filteredRuns[0].task_name}-${filteredRuns[0].completed_at}-0`);
  }, [filteredRuns, selectedRunKey]);

  const selectedRun = useMemo(() => {
    if (!filteredRuns.length) {
      return null;
    }
    const matching = filteredRuns.find((run, index) => `${run.task_name}-${run.completed_at}-${index}` === selectedRunKey);
    return matching ?? filteredRuns[0];
  }, [filteredRuns, selectedRunKey]);

  const relatedConversations = useMemo(() => {
    const conversations = conversationsQuery.data?.conversations ?? [];
    if (!selectedRun) {
      return conversations;
    }
    return conversations.filter((conversation) => conversation.task_name === selectedRun.task_name);
  }, [conversationsQuery.data?.conversations, selectedRun]);

  return (
    <div className="space-y-6">
      <section className="panel p-4">
        <p className="panel-title">Activity</p>
        <h2 className="mt-1 text-lg font-semibold text-text">Recent agent runs and AI traces</h2>
        <div className="mt-3 max-w-md">
          <input
            value={taskFilter}
            onChange={(event) => setTaskFilter(event.target.value)}
            className="w-full rounded-lg border border-border bg-ink px-3 py-2 text-xs text-text outline-none focus:border-accent"
            placeholder="Filter by task name"
          />
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-[0.85fr_1.15fr]">
        <div className="table-shell">
          <div className="border-b border-border px-4 py-4">
            <p className="panel-title">Agent runs</p>
            <h2 className="mt-2 text-lg font-semibold text-text">Pick a run to inspect</h2>
          </div>
          <table>
            <thead>
              <tr>
                <th>Task</th>
                <th>Status</th>
                <th>Completed</th>
                <th>Duration</th>
              </tr>
            </thead>
            <tbody>
              {filteredRuns.map((run, index) => {
                const runKey = `${run.task_name}-${run.completed_at}-${index}`;
                const isSelected = selectedRun?.task_name === run.task_name && selectedRun.completed_at === run.completed_at;
                return (
                <tr key={runKey} className={isSelected ? 'bg-ink/80' : ''} onClick={() => setSelectedRunKey(runKey)}>
                  <td>{run.task_name}</td>
                  <td>{run.status}</td>
                  <td>{formatTimestamp(run.completed_at)}</td>
                  <td>{run.duration_seconds.toFixed(2)}s</td>
                </tr>
              );})}
            </tbody>
          </table>
        </div>

        <div className="space-y-6">
          <section className="panel p-4">
            <p className="panel-title">Selected run</p>
            <h2 className="mt-2 text-lg font-semibold text-text">{selectedRun?.task_name ?? 'No run selected'}</h2>
            {selectedRun ? (
              <div className="mt-3 space-y-3 text-sm text-text">
                <div className="grid gap-2 sm:grid-cols-2">
                  <div className="rounded-lg border border-border bg-ink/60 p-2.5">
                    <div className="text-xs uppercase tracking-[0.16em] text-muted">Status</div>
                    <div className="mt-1.5">{selectedRun.status}</div>
                  </div>
                  <div className="rounded-lg border border-border bg-ink/60 p-2.5">
                    <div className="text-xs uppercase tracking-[0.16em] text-muted">Completed</div>
                    <div className="mt-1.5">{formatTimestamp(selectedRun.completed_at)}</div>
                  </div>
                  <div className="rounded-lg border border-border bg-ink/60 p-2.5">
                    <div className="text-xs uppercase tracking-[0.16em] text-muted">Duration</div>
                    <div className="mt-1.5">{selectedRun.duration_seconds.toFixed(2)}s</div>
                  </div>
                  <div className="rounded-lg border border-border bg-ink/60 p-2.5">
                    <div className="text-xs uppercase tracking-[0.16em] text-muted">Selection strategy</div>
                    <div className="mt-1.5">{selectedRun.result_metadata?.strategy_used ?? '-'}</div>
                  </div>
                </div>

                <div className="rounded-lg border border-border bg-ink/60 p-3">
                  <div className="text-xs uppercase tracking-[0.16em] text-muted">Result</div>
                  <pre className="mt-2 whitespace-pre-wrap break-words text-xs text-text">{selectedRun.result_summary ?? selectedRun.error_text ?? '-'}</pre>
                </div>
              </div>
            ) : (
              <p className="mt-4 text-sm text-muted">No runs match the current filter.</p>
            )}
          </section>

          <section className="table-shell">
            <div className="border-b border-border px-4 py-4">
              <p className="panel-title">Related AI conversations</p>
              <h2 className="mt-2 text-lg font-semibold text-text">Prompt/response audit surface</h2>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Task</th>
                  <th>Provider</th>
                  <th>Status</th>
                  <th>Prompt / response</th>
                </tr>
              </thead>
              <tbody>
                {relatedConversations.map((conversation) => (
                  <tr key={conversation.id}>
                    <td>
                      <div>{conversation.task_name ?? '-'}</div>
                      <div className="mt-1 text-xs text-muted">request {conversation.request_id}</div>
                    </td>
                    <td>
                      <div>{conversation.provider_name}</div>
                      <div className="mt-1 text-xs text-muted">{conversation.model_name}</div>
                    </td>
                    <td>{conversation.status}</td>
                    <td>
                      <div className="text-xs text-muted">Prompt</div>
                      <div className="mt-1 max-h-24 overflow-hidden text-ellipsis">{conversation.prompt_text}</div>
                      <div className="mt-3 text-xs text-muted">Response</div>
                      <div className="mt-1 max-h-24 overflow-hidden text-ellipsis">{conversation.response_text}</div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </div>
      </section>
    </div>
  );
}