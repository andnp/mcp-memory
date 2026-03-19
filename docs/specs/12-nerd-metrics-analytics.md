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

- Promotion and supersession funnels
- Search confidence and zero-result analytics
- Never-surfaced and long-tail retrieval views

### Phase C — graph and runtime depth

- More graph-topology drill-downs
- Agent efficiency ladders
- Queue/runtime anomaly evidence panels
- Cross-workspace reuse/linkage analytics

## 7. Testing requirements

At minimum, backend verification must cover:

- empty-store shape
- workspace scoping
- top-tag grouping behavior
- type/status grouping
- age and size bucket correctness
- timeline presence and basic correctness
- API contract exposure under `/api/metrics/nerd`
