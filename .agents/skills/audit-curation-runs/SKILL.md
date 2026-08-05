---
name: audit-curation-runs
description: Audit autonomous memory curation and maintenance for runtime health, mutation yield, quality regressions, provider-call cost, token-telemetry gaps, and current memorybase impact. Use when asked whether a curator is working, what it changed, how much it costs, whether it is safe to leave running, or to investigate an unattended mcp-memory maintenance window.
---

# Audit Curation Runs

Use a live, evidence-first audit. Start with the daemon management API and compact
JSON projections; inspect source or databases only when the API cannot answer a
specific question. The usual audit should take a few focused queries, not a
repository-wide exploration.

## Fast workflow

1. Define the window, normally the last 24 hours, and record the audit time. Use
   the daemon's timestamps consistently; do not infer the window from process
   uptime alone.
2. Search memory for prior curation, telemetry, cost, and quality findings. Read
   only the clearly relevant records. At the end, record one durable thought with
   the observed conclusion, evidence, and product gap.
3. Check runtime and storage before touching any database:

   ```bash
   curl -sS http://127.0.0.1:4242/api/health \
     | jq '{runtime_active,
            storage_backend,
            db_path,
            memory_path,
            task_queue_enabled,
            status}'
   ```

   If the endpoint is unavailable, use `uv run python -m mcp_memory.cli admin
   health --json`. If the backend is `postgres`, treat the daemon API as the
   authoritative read path and do not inspect the local SQLite file. A local
   `memory.db` may be old, empty, or unrelated to the live global store.
4. Query the overview and nerd metrics in parallel. Keep the output compact:

   ```bash
   curl -sS http://127.0.0.1:4242/api/overview \
     | jq '{memories, memory_metrics, token_usage,
            provider_usage:[.provider_usage[]
              | select(.task_name=="memory-curator")],
            agent_runs:[.agent_runs[]
              | select(.task_name=="memory-curator")]}'

   curl -sS -X POST \
     'http://127.0.0.1:4242/api/metrics/nerd?scope=global&window_hours=24&bucket_minutes=60' \
     | jq '{curation,
            curator:([.maintenance_summary.by_agent[]
              | select(.key=="memory-curator")][0]),
            quality:.curation.retrieval_quality,
            provider_stats:[.stats[]
              | select(.key|test("runs|provider|token|mutation|tool_calls"))]}'
   ```

   The overview gives current memory counts, provider-call summaries, and the
   running/last curator state. Nerd metrics gives curation outcomes, receipts,
   rejection reasons, quality signals, and the curator-only maintenance summary.
   Use `-X POST` for overview as a compatibility fallback if the installed daemon
   rejects the default GET.
5. Inspect recent mutation receipts only after the aggregate numbers are known:

   ```bash
   curl -sS -X POST \
     'http://127.0.0.1:4242/api/mutation-history?limit=20&offset=0' \
     | jq '{has_more,
            events:[.events[] | {
              operation,status,created_at,actor_kind,
              receipt_status:(.receipt.status // null),
              affected_count:(.receipt.affected_ids|length)
            }]}'
   ```

   Recent verified merges, rewrites, normalizations, archives, splits, or links
   are concrete evidence of memorybase change. Do not infer yield from candidate
   counts, provider prose, or a reported tool-call count.
6. Use task-run JSON only when run classifications or retry causes are missing
   from nerd metrics. Never print the full result payload:

   ```bash
   uv run python -m mcp_memory.cli admin task recent-runs \
     --limit 100 --json 2>/dev/null \
     | jq -c '
       [.runs[]
        | select(.task_name=="memory-curator")
        | select((.completed_at // .started_at) >= (now-86400))] as $r |
       {count:($r|length),
        statuses:($r|group_by(.status)|map({status:.[0].status,count:length})),
        classifications:($r|group_by(.run_classification // "unknown")
          | map({classification:(.[0].run_classification // "unknown"),count:length})),
        mutations:($r|map(.result.curation_campaign_result.mutation_count // 0)|add),
        productive:($r|map(.result.curation_campaign_result.productive_mutation_count // 0)|add),
        first_completed:($r|map(.completed_at)|min),
        last_completed:($r|map(.completed_at)|max)}'
   ```

## What to measure

Separate these quantities in the report:

- **Attempts:** total runs, completed runs, retries, cancellations, and active
  runs.
- **Yield:** mutation-bearing runs, persisted mutation count, accepted actions,
  verified receipts, and the operation/category mix.
- **Safety:** invalid plans, provider failures, verification failures, rejected
  receipts, stale-precondition errors, and admission skips.
- **Quality:** retrieval regressions, content-quality improvements and
  regressions, useful-work rate, valid-plan rate, and zero-result changes.
- **Cost:** actual provider calls, failures, skips, model/provider breakdown,
  average duration, and any budget/admission-limit evidence.

Use the curator-only row from `maintenance_summary.by_agent` for curator yield.
Label the broader `.curation` quality panel as curation-wide unless the API
explicitly scopes it to `memory-curator`; do not mix those populations silently.

## Token accounting rules

Provider-call counts are not token counts. Before claiming a token total, check
the live response's input, output, total, cached, and reasoning token fields.
In the current mcp-memory runtime, `provider_usage` is the authoritative
aggregate when its token source is populated, and `ai_conversations` exposes
per-call token fields. Report the token source and distinguish recorded totals
from **unknown** or incomplete telemetry.

If a rough estimate is useful, sum prompt and response character counts from at
most the API's 200-conversation limit and clearly label the result as a rough
token-equivalent estimate. Do not present it as billing data, and do not derive
tokens from provider-call count alone. A daily budget-exceeded or admission-skip
signal is useful evidence of cost pressure, but it is not a substitute for
token totals.

## Interpretation guardrails

- A nonzero mutation count answers “did it change the store?” but not “did it
  improve retrieval?” Require quality telemetry and verified receipts for the
  latter.
- Use persisted task results, mutation history, receipts, and budget counters as
  authoritative evidence. Provider-generated summaries and provider-reported
  tool counts are advisory only.
- Report current total/active/archived/stale memory counts, but do not claim a net
  24-hour increase or decrease without a start-of-window baseline.
- Treat retrieval or content-quality regressions as a first-class warning even if
  mutation throughput is high. A curator that changes many records can still be
  harmful.
- Do not start with codegraph, broad `rg`, or raw database spelunking for a live
  operations question. Use them only to resolve an endpoint/field ambiguity or
  explain a confirmed implementation gap.
- Do not enqueue, cancel, pause, or otherwise mutate tasks during a read-only
  audit unless the user explicitly asks for operational intervention.

## Report format

Lead with a verdict such as “active and productive, but quality-risky” or
“mostly retrying and not producing useful changes.” Then report:

1. Window and runtime state, including whether a curator run is currently active.
2. Attempts and yield, with mutation and receipt counts.
3. Quality and safety signals, especially regressions and rejection causes.
4. Provider calls, failures, skips, budget signals, and token-accounting status.
5. The current memorybase counts and any missing baseline or telemetry.
6. One concrete next action, if the evidence warrants it.

Keep raw JSON, full prompts, and large task results out of the final response.
State telemetry gaps plainly as product feedback. Finish by recording a durable
thought when the audit reveals a root cause, decision, verified behavior, or
reusable operational lesson.
