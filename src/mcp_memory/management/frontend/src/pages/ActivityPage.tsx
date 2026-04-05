import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';

import type { ManagementScope } from '../lib/api';
import { fetchAIConversations, fetchOverview } from '../lib/api';

function formatTimestamp(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleString();
}

export function ActivityPage() {
  const [conversationScope, setConversationScope] = useState<'global' | 'workspace'>('global');
  const [conversationWorkspaceId, setConversationWorkspaceId] = useState('');
  const conversationFilters = useMemo(
    (): {
      limit: number;
      scope?: ManagementScope;
      workspace_id?: string;
    } => ({
      limit: 25,
      scope: conversationScope === 'workspace' ? 'workspace' : undefined,
      workspace_id:
        conversationScope === 'workspace'
          ? (conversationWorkspaceId.trim() || undefined)
          : undefined,
    }),
    [conversationScope, conversationWorkspaceId],
  );
  const overviewQuery = useQuery({
    queryKey: ['overview'],
    queryFn: fetchOverview,
    refetchInterval: 30000,
    refetchOnWindowFocus: false,
  });
  const conversationsQuery = useQuery({
    queryKey: ['ai-conversations', conversationFilters],
    queryFn: () => fetchAIConversations(conversationFilters),
    refetchInterval: 15000,
    refetchOnWindowFocus: false,
  });

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

  let runsState = 'Loading recent agent runs…';
  if (overviewQuery.isError) {
    runsState = 'Unable to load recent agent runs.';
  } else if (overviewQuery.isSuccess) {
    runsState = taskFilter.trim() ? 'No runs match the current task filter.' : 'No recent agent runs yet.';
  }

  let selectedRunState = 'Loading recent runs…';
  if (overviewQuery.isError) {
    selectedRunState = 'Recent runs failed to load.';
  } else if (overviewQuery.isSuccess) {
    selectedRunState = taskFilter.trim() ? 'No runs match the current filter.' : 'No recent agent runs yet.';
  }

  let conversationsState = 'Loading task-name related conversations…';
  if (conversationsQuery.isError) {
    conversationsState = 'Unable to load task-name related conversations.';
  } else if (conversationsQuery.isSuccess) {
    conversationsState = 'No task-name related conversations found for the current filters.';
  }

  return (
    <div className="space-y-6">
      <section className="panel p-4">
        <p className="panel-title">Activity</p>
        <h2 className="mt-1 text-lg font-semibold text-text">Recent global runs and task-related AI traces</h2>
        <div className="mt-3 grid gap-2 lg:grid-cols-3">
          <input
            value={taskFilter}
            onChange={(event) => setTaskFilter(event.target.value)}
            className="w-full rounded-lg border border-border bg-ink px-3 py-2 text-xs text-text outline-none focus:border-accent"
            placeholder="Filter by task name"
          />
          <select
            value={conversationScope}
            onChange={(event) => setConversationScope(event.target.value as 'global' | 'workspace')}
            className="rounded-lg border border-border bg-ink px-3 py-2 text-xs text-text outline-none focus:border-accent"
          >
            <option value="global">global conversations</option>
            <option value="workspace">current workspace conversations</option>
          </select>
          <input
            value={conversationWorkspaceId}
            onChange={(event) => setConversationWorkspaceId(event.target.value)}
            className="w-full rounded-lg border border-border bg-ink px-3 py-2 text-xs text-text outline-none focus:border-accent"
            placeholder="optional workspace-id override"
          />
        </div>
        <p className="mt-2 text-xs text-muted">
          Workspace ID override only applies when workspace scope is selected.
        </p>
        <p className="mt-2 text-xs text-muted">
          Conversation rows below are related to the selected run by <code>task_name</code>. They are not guaranteed to be the exact request(s) from that specific run.
        </p>
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
              {filteredRuns.length ? filteredRuns.map((run, index) => {
                const runKey = `${run.task_name}-${run.completed_at}-${index}`;
                const isSelected = selectedRun?.task_name === run.task_name && selectedRun.completed_at === run.completed_at;
                return (
                <tr key={runKey} className={isSelected ? 'bg-ink/80' : ''} onClick={() => setSelectedRunKey(runKey)}>
                  <td className="max-w-[14rem] truncate" title={run.task_name}>{run.task_name}</td>
                  <td>{run.status}</td>
                  <td>{formatTimestamp(run.completed_at)}</td>
                  <td>{run.duration_seconds.toFixed(2)}s</td>
                </tr>
              );}) : (
                <tr>
                  <td colSpan={4} className="text-muted">{runsState}</td>
                </tr>
              )}
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
              <p className="mt-4 text-sm text-muted">{selectedRunState}</p>
            )}
          </section>

          <section className="table-shell">
            <div className="border-b border-border px-4 py-4">
              <p className="panel-title">Task-name related AI conversations</p>
              <h2 className="mt-2 text-lg font-semibold text-text">Prompt/response audit surface</h2>
              <p className="mt-2 text-sm text-muted">Matched by task name only, with optional workspace narrowing.</p>
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
                {relatedConversations.length ? relatedConversations.map((conversation) => (
                  <tr key={conversation.id}>
                    <td>
                      <div className="max-w-[12rem] truncate" title={conversation.task_name ?? '-'}>{conversation.task_name ?? '-'}</div>
                      <div className="mt-1 text-[10px] text-muted">request {conversation.request_id}</div>
                      <div className="mt-1 text-[10px] text-muted">workspace {conversation.workspace_id ?? 'global'}</div>
                    </td>
                    <td>
                      <div className="truncate" title={conversation.provider_name}>{conversation.provider_name}</div>
                      <div className="mt-1 truncate text-[10px] text-muted" title={conversation.model_name}>{conversation.model_name}</div>
                    </td>
                    <td>
                      <div>{conversation.status}</div>
                      {conversation.reason_code ? (
                        <div className="mt-1 text-[10px] text-muted">
                          {conversation.reason_category ?? 'reason'} · {conversation.reason_code}
                        </div>
                      ) : null}
                    </td>
                    <td>
                      <div className="text-[10px] text-muted">P</div>
                      <div className="mt-0.5 max-w-[24rem] truncate" title={conversation.prompt_text}>{conversation.prompt_text}</div>
                      <div className="mt-1.5 text-[10px] text-muted">R</div>
                      <div className="mt-0.5 max-w-[24rem] truncate" title={conversation.response_text}>{conversation.response_text}</div>
                      {conversation.error_text ? <div className="mt-1.5 max-w-[24rem] truncate text-[10px] text-danger" title={conversation.error_text}>{conversation.error_text}</div> : null}
                    </td>
                  </tr>
                )) : (
                  <tr>
                    <td colSpan={4} className="text-muted">{conversationsState}</td>
                  </tr>
                )}
              </tbody>
            </table>
          </section>
        </div>
      </section>
    </div>
  );
}