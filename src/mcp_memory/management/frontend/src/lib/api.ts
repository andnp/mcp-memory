export interface AgentRun {
  task_name: string;
  running_count: number;
  total_runs: number;
  failed_runs: number;
  last_status: string | null;
  last_result_summary: string | null;
}

export interface RecentAgentRun {
  task_name: string;
  status: string;
  duration_seconds: number;
  completed_at: number;
  result_summary: string | null;
  error_text: string | null;
}

export interface ProviderUsage {
  task_name: string | null;
  provider_key: string;
  model_name: string;
  calls_last_day: number;
  failures_last_day: number;
}

export interface OverviewResponse {
  memories: { total: number };
  memory_metrics: {
    total_memory_lines: number;
    total_lines_compressed: number;
    thought_buffer_entries: number;
  };
  tasks: { failed_count: number };
  embeddings: { backend: string | null; model_name: string | null; model_cached: boolean };
  storage: { sqlite_bytes: number };
  agent_runs: AgentRun[];
  recent_agent_runs: RecentAgentRun[];
  provider_usage: ProviderUsage[];
  journal: { pending_count: number };
}

export interface CommandBarResult {
  status?: string;
  [key: string]: unknown;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export function fetchOverview(): Promise<OverviewResponse> {
  return requestJson<OverviewResponse>('/api/overview');
}

export function recordThought(content: string): Promise<CommandBarResult> {
  return requestJson<CommandBarResult>('/api/record-thought', {
    method: 'POST',
    body: JSON.stringify({ content }),
  });
}

export function runAgent(taskName: string): Promise<CommandBarResult> {
  return requestJson<CommandBarResult>('/api/admin/agents/run', {
    method: 'POST',
    body: JSON.stringify({ task_name: taskName, force: false }),
  });
}

export function runAllAgents(): Promise<CommandBarResult> {
  return requestJson<CommandBarResult>('/api/admin/agents/run-all', {
    method: 'POST',
    body: JSON.stringify({ force: false }),
  });
}