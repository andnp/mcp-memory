import { useQuery } from '@tanstack/react-query';

import { fetchAIConversations, fetchOverview } from '../lib/api';

function formatTimestamp(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleString();
}

export function ActivityPage() {
  const overviewQuery = useQuery({ queryKey: ['overview'], queryFn: fetchOverview, refetchInterval: 5000 });
  const conversationsQuery = useQuery({ queryKey: ['ai-conversations'], queryFn: () => fetchAIConversations(10), refetchInterval: 5000 });

  return (
    <div className="space-y-6">
      <section className="panel p-6">
        <p className="panel-title">Activity</p>
        <h2 className="mt-2 text-2xl font-semibold text-text">Recent agent runs and AI traces</h2>
      </section>

      <section className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
        <div className="table-shell">
          <div className="border-b border-border px-4 py-4">
            <p className="panel-title">Agent runs</p>
            <h2 className="mt-2 text-lg font-semibold text-text">Recent maintenance outcomes</h2>
          </div>
          <table>
            <thead>
              <tr>
                <th>Task</th>
                <th>Status</th>
                <th>Completed</th>
                <th>Duration</th>
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              {overviewQuery.data?.recent_agent_runs.map((run, index) => (
                <tr key={`${run.task_name}-${run.completed_at}-${index}`}>
                  <td>{run.task_name}</td>
                  <td>{run.status}</td>
                  <td>{formatTimestamp(run.completed_at)}</td>
                  <td>{run.duration_seconds.toFixed(2)}s</td>
                  <td>{run.result_summary ?? run.error_text ?? '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="table-shell">
          <div className="border-b border-border px-4 py-4">
            <p className="panel-title">AI conversations</p>
            <h2 className="mt-2 text-lg font-semibold text-text">Prompt/response audit surface</h2>
          </div>
          <table>
            <thead>
              <tr>
                <th>Task</th>
                <th>Provider</th>
                <th>Status</th>
                <th>Duration</th>
              </tr>
            </thead>
            <tbody>
              {conversationsQuery.data?.conversations.map((conversation) => (
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
                  <td>{conversation.duration_seconds.toFixed(2)}s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}