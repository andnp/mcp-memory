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
  result_metadata?: {
    strategy_used?: string | null;
    requested_strategy?: string | null;
    candidate_count?: number | null;
    grouping_strategy_used?: string | null;
    requested_grouping_strategy?: string | null;
    group_count?: number | null;
  };
}

export interface ProviderUsage {
  task_name: string | null;
  provider_key: string;
  model_name: string;
  calls_last_day: number;
  failures_last_day: number;
}

export interface RecentMemory {
  id: string;
  title: string;
  summary: string | null;
  type: string;
  status: string;
  updated_at: string;
  tags: string[];
}

export interface FailedTask {
  id: string;
  task_name: string;
  status: string;
  retries_count: number;
  max_retries: number;
  last_error: string | null;
}

export interface RecentLog {
  id: number;
  created_at: number;
  level: string;
  logger_name: string;
  source: string;
  message: string;
}

export interface QueueDiagnostic {
  task_name: string;
  pending_state: string;
  priority: number;
  age_seconds: number;
  trigger: string | null;
  workspace_id: string | null;
  ready_in_seconds: number;
  overdue_seconds: number;
}

export interface TopReadMemory {
  id: string;
  title: string;
  type: string;
  status: string;
  read_count: number;
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
  recent_memories: RecentMemory[];
  failed_tasks: FailedTask[];
  recent_logs: RecentLog[];
  queue_diagnostics: QueueDiagnostic[];
  top_read_memories: TopReadMemory[];
  top_read_memories_active: TopReadMemory[];
}

export interface MemorySearchResult {
  memory_id: string;
  title: string;
  summary: string | null;
  memory_type: string;
  status: string;
  tags: string[];
  workspace_ids: string[];
  score: number;
}

export interface MemorySearchResponse {
  results: MemorySearchResult[];
}

export interface MemoryDetailResponse {
  record: {
    id: string;
    title: string;
    content: string;
    summary: string | null;
    type: string;
    status: string;
    created_at: string;
    updated_at: string;
    workspace_ids: string[];
    tags: string[];
    metadata: Record<string, unknown>;
  };
  relationships: {
    incoming: Array<{ source_id: string; target_id: string; link_type: string; context: string }>;
    outgoing: Array<{ source_id: string; target_id: string; link_type: string; context: string }>;
  };
  superseded: Array<{
    id: string;
    title: string;
    summary: string | null;
    type: string;
    status: string;
  }>;
}

export interface AIConversation {
  id: number;
  request_id: string;
  task_name: string | null;
  task_id: string | null;
  provider_name: string;
  model_name: string;
  status: string;
  error_text: string | null;
  started_at: number;
  completed_at: number;
  duration_seconds: number;
  prompt_text: string;
  response_text: string;
}

export interface AIConversationListResponse {
  conversations: AIConversation[];
}

export interface RuntimeLogSummaryResponse {
  total: number;
  by_level: Record<string, number>;
  by_source: Record<string, number>;
}

export interface NerdStat {
  key: string;
  label: string;
  value: number;
  unit: string | null;
}

export interface AgentThroughputBucket {
  bucket_start: number;
  total_runs: number;
  completed_runs: number;
  failed_runs: number;
  retry_runs: number;
  avg_duration_seconds: number;
}

export interface ProviderLatencyBucket {
  bucket_start: number;
  provider_key: string;
  provider_name: string;
  model_name: string;
  call_count: number;
  failure_count: number;
  avg_duration_seconds: number;
  p95_duration_seconds: number;
}

export interface GraphTopology {
  total_memories: number;
  total_links: number;
  average_degree: number;
  orphan_count: number;
  orphan_rate: number;
  graph_supported_count: number;
  graph_supported_rate: number;
  link_type_counts: Record<string, number>;
}

export interface MemoryLifecycle {
  by_status: Record<string, number>;
  by_type: Record<string, number>;
  total_content_bytes: number;
  median_content_bytes: number;
  cold_memory_count: number;
  cold_memory_rate: number;
  never_surfaced_count: number;
  stale_count: number;
  degraded_count: number;
}

export interface SearchQuality {
  semantic_enabled: boolean;
  degraded: boolean;
  fallback_count: number;
  rebuild_count: number;
  graph_supported_count: number;
  graph_supported_rate: number;
  never_surfaced_count: number;
  last_error: string | null;
}

export interface CountBucket {
  key: string;
  label: string;
  count: number;
}

export interface Composition {
  by_workspace: CountBucket[];
  by_tag: CountBucket[];
  by_type: CountBucket[];
  by_status: CountBucket[];
}

export interface Distributions {
  created_age_buckets: CountBucket[];
  updated_age_buckets: CountBucket[];
  content_size_buckets: CountBucket[];
}

export interface MemoryTimelineBucket {
  bucket_start: number;
  created_count: number;
  updated_count: number;
  total_content_bytes: number;
}

export interface MaintenanceEvent {
  task_id: string;
  task_name: string;
  status: string;
  completed_at: number;
  bucket_start: number;
  duration_seconds: number;
  result_summary: string | null;
  strategy_used: string | null;
  impact_summary: string | null;
  candidate_count: number | null;
  group_count: number | null;
  created_count: number | null;
  merged_count: number | null;
  updated_count: number | null;
  archived_count: number | null;
  lines_compressed: number | null;
  meaningful_actions: number | null;
}

export interface Timelines {
  memory_activity: MemoryTimelineBucket[];
}

export interface MaintenanceMetrics {
  events: MaintenanceEvent[];
}

export interface NerdAlert {
  key: string;
  severity: 'info' | 'warning' | 'error';
  label: string;
  message: string;
  value: number;
  threshold: number | null;
  unit: string | null;
}

export interface TaskRouteAudit {
  task_name: string;
  task_class: string;
  execution_kind: string;
  low_priority: boolean;
  configured_primary_route: string | null;
  configured_fallback_routes: string[];
  resolved_provider_key: string | null;
  resolved_model_name: string | null;
  resolved_provider_type: string | null;
  resolved_supports_agentic: boolean | null;
  recent_provider_key: string | null;
  recent_model_name: string | null;
  recent_status: string | null;
  recent_success_count: number;
  recent_failure_count: number;
  on_primary_route: boolean | null;
}

export interface NerdMetricsResponse {
  generated_at: number;
  window_hours: number;
  bucket_minutes: number;
  stats: NerdStat[];
  queue_snapshot: {
    runnable_count: number;
    scheduled_count: number;
    oldest_age_seconds: number;
  };
  graph_topology: GraphTopology;
  memory_lifecycle: MemoryLifecycle;
  composition: Composition;
  distributions: Distributions;
  timelines: Timelines;
  maintenance: MaintenanceMetrics;
  search_quality: SearchQuality;
  route_audit: TaskRouteAudit[];
  alerts: NerdAlert[];
  agent_throughput: AgentThroughputBucket[];
  provider_latency: ProviderLatencyBucket[];
}

export interface CommandBarResult {
  status?: string;
  [key: string]: unknown;
}

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
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
    throw new ApiError(response.status, `Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export function fetchOverview(): Promise<OverviewResponse> {
  return requestJson<OverviewResponse>('/api/overview');
}

export function searchMemories(params: {
  query: string;
  limit?: number;
  memory_type?: string;
  status?: string;
}): Promise<MemorySearchResponse> {
  return requestJson<MemorySearchResponse>('/api/memories/search', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export function fetchMemoryDetail(memoryId: string): Promise<MemoryDetailResponse> {
  return requestJson<MemoryDetailResponse>(`/api/memories/${memoryId}`);
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

export function fetchAIConversations(limit = 12): Promise<AIConversationListResponse> {
  return requestJson<AIConversationListResponse>('/api/ai-conversations', {
    method: 'POST',
    body: JSON.stringify({ limit }),
  });
}

export function fetchLogs(params: {
  limit?: number;
  level?: string;
  q?: string;
  source?: string;
  logger_name?: string;
} = {}): Promise<{ logs: RecentLog[] }> {
  return requestJson<{ logs: RecentLog[] }>('/api/logs', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export function fetchLogSummary(params: {
  level?: string;
  q?: string;
  source?: string;
  logger_name?: string;
} = {}): Promise<RuntimeLogSummaryResponse> {
  return requestJson<RuntimeLogSummaryResponse>('/api/logs/summary', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export function fetchNerdMetrics(params: { window_hours?: number; bucket_minutes?: number } = {}): Promise<NerdMetricsResponse> {
  return requestJson<NerdMetricsResponse>('/api/metrics/nerd', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}
