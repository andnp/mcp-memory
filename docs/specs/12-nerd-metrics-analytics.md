# Specification: Nerd Metrics Analytics

**Status:** Draft
**Scope:** Management API `/api/metrics/nerd`
**Rollout:** Phased, additive, backend-first

## 1. Purpose

The nerd metrics surface exists to make the memory system legible under real operating conditions.

It is not a vanity dashboard. It is an operator/debugging surface for answering questions such as:

- What kinds of memories exist right now?
- Which workspaces, tags, and statuses dominate the graph?
- Is the memory base getting older, staler, or larger?
- Are updates and creation activity clustered, stalled, or skewed?

This spec expands the existing `/api/metrics/nerd` payload with a first analytics wave focused on **composition**, **distributions**, and **timelines**.

It now also defines the next additive backend tranche for **lifecycle trends**, **growth dynamics**, and **maintenance summaries**.

It also defines an operator-facing **retrieval analytics** slice for tool-level memory search/read traffic so the dashboard can answer questions such as:

- Which memories surface most often in searches?
- Which memories are actually opened/read most often?
- Which tags dominate retrieval traffic?
- How is retrieval traffic for the top tags changing over time?

## 2. Contract rules

### 2.1 Additive-only evolution

- Existing keys in `NerdMetricsPayload` remain valid.
- New analytics sections are appended as new top-level keys.
- Clients must tolerate additional future keys.

### 2.2 Workspace scoping semantics

All sections in `/api/metrics/nerd` are scoped to the requesting runtime workspace.

- `workspace_id=None` means global scope.
- `workspace_id=<id>` means only memories visible in that workspace contribute to counts, buckets, and timelines.
- For `composition.by_workspace`, each visible memory contributes to each of its associated workspace IDs.
  - This means a shared memory can appear in more than one workspace bucket.
  - A workspace-scoped request must still exclude memories that are not visible in that scope.

### 2.3 Empty-store behavior

- Empty stores must return a valid payload.
- Composition sections return empty lists.
- Distribution sections return the standard bucket list with zero counts.
- Timeline sections return empty lists.
- No section should raise or require special client handling.

## 3. Endpoint

`/api/metrics/nerd`

Existing request parameters remain unchanged:

- `window_hours`
- `bucket_minutes`
- `now` (test/debug override)

## 4. Payload additions

### 4.1 `composition`

```text
composition: {
  by_workspace: CountBucket[]
  by_tag: CountBucket[]
  by_type: CountBucket[]
  by_status: CountBucket[]
}
```

`CountBucket`:

- `key: str`
- `label: str`
- `count: int`

Rules:

- `by_workspace`: count visible memories by associated workspace ID.
- `by_tag`: top 10 tags by count, descending; append `other` when additional tags remain.
- `by_type`: count visible memories by memory type.
- `by_status`: count visible memories by status.

### 4.2 `distributions`

```text
distributions: {
  created_age_buckets: CountBucket[]
  updated_age_buckets: CountBucket[]
  content_size_buckets: CountBucket[]
}
```

#### Created/updated age buckets

Exact bucket keys for slice 1:

- `lt_1d` — age < 1 day
- `1d_to_7d` — 1 day <= age < 7 days
- `7d_to_30d` — 7 days <= age < 30 days
- `30d_to_90d` — 30 days <= age < 90 days
- `gte_90d` — age >= 90 days

Age is computed from the request's effective `now` timestamp against `created_at` or `updated_at`.

#### Content-size buckets

Exact bucket keys for slice 1:

- `0b_to_255b` — 0 to 255 bytes
- `256b_to_1kb` — 256 bytes to < 1 KiB
- `1kb_to_4kb` — 1 KiB to < 4 KiB
- `4kb_to_16kb` — 4 KiB to < 16 KiB
- `gte_16kb` — 16 KiB and above

Content size is computed from `LENGTH(COALESCE(content, ''))`.

### 4.3 `timelines`

```text
timelines: {
  memory_activity: MemoryTimelineBucket[]
}
```

`MemoryTimelineBucket`:

- `bucket_start: float` — unix timestamp for the bucket boundary
- `created_count: int`
- `updated_count: int`
- `total_content_bytes: int`

Rules:

- `created_count`: number of visible memories whose `created_at` falls in the bucket.
- `updated_count`: number of visible memories whose `updated_at` falls in the bucket.
- `total_content_bytes`: cumulative content bytes for currently visible memories across the selected window, seeded with visible memories created before the window start.
  - This is intentionally approximate for slice 1.
  - It is based on present-day memory size plus creation time, not historical content diff logs.

### 4.4 `lifecycle_trends`

```text
lifecycle_trends: {
  status_events: CountSeries[]
  never_surfaced_backlog: TimeCountBucket[]
  cold_tail: TimeCountBucket[]
}
```

`CountSeries`:

- `key: str`
- `label: str`
- `buckets: TimeCountBucket[]`

`TimeCountBucket`:

- `bucket_start: float`
- `count: int`

Rules:

- `status_events` includes only lifecycle-like event families that the runtime can report honestly from task-run results.
  - Current shipped keys are limited to counters such as `stale`, `degraded`, `archived`, and `restored` when those counters were explicitly reported by a run.
  - If the selected window has no explicit counters for one of those keys, omit that series rather than synthesizing zeros for a fake metric.
- `never_surfaced_backlog` is a cumulative/current-stock approximation for currently visible memories whose `last_surfaced_at` is still null.
  - It is seeded by memories created before the window start.
  - It is **not** a historical reconstruction of when memories became unsurfaced or were later surfaced.
- `cold_tail` is the same style of cumulative/current-stock approximation for currently visible memories whose `last_accessed_at` is still null.
  - It is intentionally a present-day cold-stock trend, not a true historical coldness reconstruction.

### 4.5 `growth_dynamics`

```text
growth_dynamics: {
  top_tag_trends: CountSeries[]
  workspace_contribution_share: ShareSeries[]
}
```

`ShareSeries`:

- `key: str`
- `label: str`
- `buckets: TimeShareBucket[]`

`TimeShareBucket`:

- `bucket_start: float`
- `count: int`
- `share: float`

Rules:

- `top_tag_trends` uses cumulative/current-stock series derived from currently visible memories and their `created_at` timestamps.
  - Select the top 5 tags by present-day scoped count.
  - If more tags remain, append `other` and aggregate the remainder there.
- `workspace_contribution_share` uses the same cumulative/current-stock method, but contributions are counted per visible memory workspace association.
  - This preserves the existing workspace semantics: a visible shared memory may contribute to more than one workspace bucket.
  - Select the top 5 workspace IDs by present-day scoped count, plus optional `other`.
  - `share` is computed within each bucket as `count / total visible contributions in that bucket`.
- These sections intentionally avoid historical stock reconstruction beyond the creation-time cumulative approximation.

### 4.6 `maintenance_summary`

```text
maintenance_summary: {
  by_family: MaintenanceSummaryRow[]
  by_agent: MaintenanceAgentYieldRow[]
  family_delta_series: MaintenanceDeltaSeries[]
}
```

`MaintenanceSummaryRow`:

- `key: str`
- `label: str`
- `task_names: str[]`
- `total_runs: int`
- `completed_runs: int`
- `failed_runs: int`
- `retry_runs: int`
- `created_count: int`
- `merged_count: int`
- `updated_count: int`
- `archived_count: int`
- `degraded_count: int`
- `restored_count: int`
- `meaningful_actions: int`
- `lines_compressed: int`
- `delta_total: int`

`MaintenanceAgentYieldRow` extends the summary row with:

- `family_key: str`
- `family_label: str`
- `actions_per_completed_run: float`
- `lines_per_completed_run: float`
- `delta_per_completed_run: float`

`MaintenanceDeltaSeries`:

- `key: str`
- `label: str`
- `task_names: str[]`
- `buckets: MaintenanceDeltaBucket[]`

Rules:

- This section uses **run-reported counters only**.
  - Do not infer maintenance impact from ingest audit side effects, linked rows, or current memory stock.
  - Missing counters mean zero contribution, not backfilled guesses.
- `by_family` rolls maintenance tasks into explicit reporting families. The shipped backend currently uses:
  - `organization` — `project-manager`, `taxonomist`
  - `verification` — `fact-checker`, `graph-linker`, `conflict-detector`
  - `compaction` — `defragmenter`, `deduplicator`, `memory-curator`
  - `retention` — `sweeper`
- `by_agent` keeps one row per task name and exposes lightweight yield ratios.
- `family_delta_series` emits bucketed per-family counters using the same explicit run-reported delta keys.

### 4.7 `retrieval`

```text
retrieval: {
  summary: {
    search_invocations: int
    search_hits: int
    zero_result_searches: int
    read_events: int
    unique_search_memories: int
    unique_read_memories: int
  }
  top_read_memories: RetrievalMemoryRow[]
  top_search_memories: RetrievalMemoryRow[]
  top_tags: RetrievalTagRow[]
  tag_timelines: RetrievalTagTimeline[]
}
```

`RetrievalMemoryRow`:

- `memory_id: str`
- `title: str`
- `memory_type: str`
- `status: str`
- `tags: str[]`
- `read_count: int`
- `search_count: int`
- `total_count: int`
- `last_read_at: float | null`
- `last_search_at: float | null`

`RetrievalTagRow`:

- `key: str`
- `label: str`
- `read_count: int`
- `search_count: int`
- `total_count: int`

`RetrievalTagTimeline`:

- `key: str`
- `label: str`
- `read_buckets: TimeCountBucket[]`
- `search_buckets: TimeCountBucket[]`

Rules:

- Retrieval analytics are based on persisted tool-level events, not inferred from current memory stock alone.
- Searches are logged at the MCP service layer for both external and internal tool usage.
- A search with no hits still contributes to `summary.search_invocations` and `summary.zero_result_searches`.
- `top_read_memories` is capped at the top 100 visible memories by read-event count within the selected window.
- `top_search_memories` is capped at the top 100 visible memories by search-hit count within the selected window.
- `top_tags` is capped at the top 25 visible tags by combined retrieval activity.
- `tag_timelines` is capped at the top 10 visible tags by combined retrieval activity and uses event-time buckets, not memory creation-time approximations.
- Workspace scoping follows the retrieval event's originating workspace context and then filters to memories currently visible in the scoped request.

## 5. First-slice information architecture

The UI is expected to consume the first analytics wave as three foundational panels:

1. **Composition** — what the memory base is made of
2. **Distributions** — how old and large the memory base is
3. **Timelines** — whether creation/update activity is moving

This slice intentionally does **not** cover the deeper graph/search/provider/operator panels already described elsewhere. Those remain later slices.

## 6. Rollout phases

### Phase A — fundamentals

- Composition by workspace, tag, type, status
- Created-age, updated-age, and content-size distributions
- Created/updated activity timeline
- Window-relative cumulative content-bytes timeline

### Phase B — lifecycle and search depth

- Honest lifecycle event trends
- Search confidence and zero-result analytics
- Never-surfaced and long-tail retrieval views

### Phase B.1 — additive backend tranche now shipped

- `lifecycle_trends`
- `growth_dynamics`
- `maintenance_summary`

### Phase C — graph and runtime depth

- More graph-topology drill-downs
- Agent efficiency ladders
- Queue/runtime anomaly evidence panels
- Cross-workspace reuse/linkage analytics

### Phase C.1 — retrieval operator panel now shipped

- Tool-level read/search telemetry persisted from MCP search/read services
- Top 100 read memories and top 100 search-hit memories
- Top 25 retrieval tags with read/search split
- Historical timelines for the top 10 retrieval tags

## 7. Testing requirements

At minimum, backend verification must cover:

- empty-store shape
- workspace scoping
- top-tag grouping behavior
- type/status grouping
- age and size bucket correctness
- timeline presence and basic correctness
- empty shapes for all additive sections
- cumulative/current-stock math for lifecycle and growth trends
- top-5 plus `other` capping for growth dynamics
- maintenance-family aggregation using run-reported counters only
- missing maintenance counters staying zero / omitted instead of fabricated
- retrieval event persistence for search-hit vs explicit-read separation
- zero-result search accounting
- top-100 memory ranking for reads and search hits
- top-25 tag rollups and top-10 tag timelines
- API contract exposure under `/api/metrics/nerd`
