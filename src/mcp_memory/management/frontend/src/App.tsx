import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';

import { CommandBar } from './components/CommandBar';
import { PulseOverview } from './components/PulseOverview';
import { fetchOverview } from './lib/api';

function formatTimestamp(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleString();
}

export default function App() {
  const [commandResult, setCommandResult] = useState('No command executed yet.');
  const overviewQuery = useQuery({
    queryKey: ['overview'],
    queryFn: fetchOverview,
    refetchInterval: 5000,
  });

  return (
    <main className="mx-auto flex min-h-screen max-w-7xl flex-col gap-6 px-6 py-8 lg:px-10">
      <header className="flex flex-col gap-3">
        <p className="panel-title">Memory Command Center</p>
        <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <h1 className="text-4xl font-semibold tracking-tight text-text">A React foundation for plan 6</h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-muted">
              This scaffold ports the dashboard toward a real command-center frontend: typed API access, a persistent command bar,
              polling-based overview refresh, and clear landing zones for search, audit stream, and graph views.
            </p>
          </div>
          <div className="rounded-2xl border border-border bg-panel/80 px-4 py-3 text-sm text-muted shadow-panel">
            Status: {overviewQuery.isLoading ? 'Loading pulse…' : overviewQuery.isError ? 'Pulse degraded' : 'Pulse online'}
          </div>
        </div>
      </header>

      <CommandBar onResult={setCommandResult} />

      <section className="panel p-4">
        <p className="panel-title">Command Result</p>
        <pre className="mt-3 overflow-x-auto whitespace-pre-wrap break-words text-sm text-text">{commandResult}</pre>
      </section>

      {overviewQuery.isError ? (
        <section className="panel p-6 text-danger">
          Failed to load the daemon overview. This scaffold expects the management API to be reachable at <code>/api</code>.
        </section>
      ) : null}

      {overviewQuery.data ? (
        <>
          <PulseOverview overview={overviewQuery.data} />

          <section className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
            <div className="table-shell">
              <div className="border-b border-border px-4 py-4">
                <p className="panel-title">Background agents</p>
                <h2 className="mt-2 text-lg font-semibold text-text">Current pulse</h2>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>Agent</th>
                    <th>Running</th>
                    <th>Runs</th>
                    <th>Failures</th>
                    <th>Last status</th>
                    <th>Last result</th>
                  </tr>
                </thead>
                <tbody>
                  {overviewQuery.data.agent_runs.slice(0, 8).map((agent) => (
                    <tr key={agent.task_name}>
                      <td>{agent.task_name}</td>
                      <td>{agent.running_count}</td>
                      <td>{agent.total_runs}</td>
                      <td>{agent.failed_runs}</td>
                      <td>{agent.last_status ?? 'never'}</td>
                      <td>{agent.last_result_summary ?? '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="space-y-6">
              <section className="table-shell">
                <div className="border-b border-border px-4 py-4">
                  <p className="panel-title">Recent runs</p>
                  <h2 className="mt-2 text-lg font-semibold text-text">Agent activity</h2>
                </div>
                <table>
                  <thead>
                    <tr>
                      <th>Agent</th>
                      <th>Status</th>
                      <th>Completed</th>
                    </tr>
                  </thead>
                  <tbody>
                    {overviewQuery.data.recent_agent_runs.slice(0, 6).map((run, index) => (
                      <tr key={`${run.task_name}-${run.completed_at}-${index}`}>
                        <td>{run.task_name}</td>
                        <td>{run.status}</td>
                        <td>{formatTimestamp(run.completed_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>

              <section className="panel p-5">
                <p className="panel-title">Next slices</p>
                <ul className="mt-3 space-y-2 text-sm leading-6 text-muted">
                  <li>• Add real search results to the command bar.</li>
                  <li>• Replace summary tables with an agent audit stream.</li>
                  <li>• Introduce graph and conflict views once the serving story is ready.</li>
                </ul>
              </section>
            </div>
          </section>
        </>
      ) : null}
    </main>
  );
}