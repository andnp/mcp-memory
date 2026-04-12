# Architecture Decision Record: Work-Item Execution Architecture

**Status:** Proposed direction
**Date:** 2026-03-22

## 1. Context

`mcp-memory` currently has one durable `tasks` queue, but each background agent still owns too much of its own execution policy.

Today that means the runtime mixes several concerns inside individual handlers:

- selecting candidate work
- deciding whether execution is deterministic or AI-powered
- choosing providers and reacting to admission/backoff
- defining task-local batching shape
- deciding how long a run should continue
- reporting mutations and retries in agent-specific ways

There is also a product-economics constraint that matters explicitly:

- premium Copilot / strong-provider usage is charged per **execution call**
- call duration is not the primary cost driver
- the scheduler should therefore optimize for **useful work completed per premium execution**, not simply for shorter runs

This has been good enough for the current maintenance rollout, but it creates two architectural tensions:

1. low-risk deterministic work and AI-powered work share a queue, yet they do not share a clear execution contract
2. agentic maintenance flows are converging toward a common internal MCP mutation surface, but scheduling and batch ownership are still fragmented by agent

Recent work on provider admission, provider-policy telemetry, and taxonomist stabilization also highlighted that task-local provider behavior is still too bespoke. The system needs a cleaner abstraction for "what work exists" and "which executor lane should process it".

## 2. Decision

Introduce a shared **work-item architecture** above the current agent/task handlers.

The target shape is:

- one durable work-item model
- two execution lanes:
  - `deterministic`
  - `agentic`
- batch acquisition by **work family**, not by task-name-specific ad hoc logic
- lease-based execution with explicit soft-stop and hard-stop behavior

The design goal is to unify scheduling and observability **without** immediately collapsing all maintenance behavior into one mega-agent prompt.

The economic goal is equally important:

> keep premium executions alive long enough to complete multiple compatible work units, and only end them when productive continuation meaningfully drops off.

### 2.1 Preferred Runtime Shape

A durable work-item record should eventually describe:

- `work_type`
- `family_key`
- `execution_lane`
- `priority`
- `workspace_id`
- `status`
- `available_at`
- `attempt_count`
- `lease_owner`
- `lease_expires_at`
- `payload_json`
- `idempotency_key`
- optional parent task / batch lineage

Operationally, the runtime will behave like it has two queues, but architecturally it should prefer one shared work model with lane-aware dispatch.

### 2.2 Batch Acquisition

Batching should happen by **family + execution contract**.

Examples of likely first-class families:

- `journal_ingest`
- `memory_curation`
- `memory_tagging`
- `memory_dedup`
- `graph_linking`
- `conflict_review`

The internal batch tools should evolve toward shapes like:

- `get_work_batch(family="memory_tagging", limit=20)`
- `heartbeat_work_batch(batch_id=...)`
- `complete_work_batch(batch_id=..., results=...)`
- `defer_work_batch(batch_id=..., reason=...)`

This is intentionally more general than today’s ingest-only or curator-only batch tools.

Batching policy should also treat a premium execution as a **campaign**, not just a single-shot task body. In practice that means:

- initial work packets should be rich enough to let the model start productive work immediately
- the executor should be able to claim more compatible work during the same provider session
- deterministic precomputation is justified only when it increases total useful work completed in that same premium session

## 3. Key Architectural Constraint

This ADR does **not** recommend jumping directly to one provider call that mixes ingest, curation, deduplication, and tagging in the same run.

That is an appealing long-term possibility, but it is not the right first implementation target.

Why:

- mixed job families have different safety rules
- context windows become harder to control
- retries become more expensive and less explainable
- auditability gets weaker when unrelated mutations share one execution transcript
- partial failure recovery becomes much more complex

The better direction is:

> unify the durable work model first, then selectively widen multi-family execution only where the safety model is shared.

And more specifically:

> do not break one premium working session into multiple paid calls merely because the software architecture prefers small isolated steps.

## 4. Special Case: Ingest

`ingest-system1` is not just another maintenance task.

It currently depends on claim/finalize semantics that are stronger than most other handlers:

- claim journal rows for one execution unit
- mutate memories
- delete only handled entries
- release untouched entries
- preserve crash-safe recovery

That means ingest should participate in the shared work-item architecture **later** or through a family-specific contract. The scheduler can still treat ingest as work, but it should not erase the stronger invariants that protect journal durability.

## 5. Lease and Time Budget Semantics

Soft-stop and hard-stop should be part of the **lease protocol**, not just prompt wording.

Recommended behavior:

- **lease start** when a worker claims a batch
- **soft deadline**: future `get_work_batch(...)` calls return `wrap_up=true` and no new work
- **hard deadline**: batch can no longer expand; worker must complete, defer, or release what it holds
- **lease recovery**: abandoned leases become visible for re-claim or explicit recovery

This gives the system a consistent control plane for long-running agentic maintenance without relying on fragile prompt discipline alone.

For premium providers, the lease protocol should be interpreted as a way to support **long-lived adaptive sessions**:

- a soft deadline should encourage graceful wrap-up without discarding already-open context
- compatibility-group continuation should remain possible while the session is still productive
- hard stops should be used to protect safety and recoverability, not as an accidental source of extra paid calls

## 6. Consequences

### Positive

- deterministic and AI-powered work gain explicit execution lanes
- provider budgeting/admission can move upward into the agentic execution lane
- batching logic becomes reusable across maintenance families
- observability improves because work units become first-class objects rather than implicit task-local loops
- future dashboard views can report batch leases, defer reasons, mutation yield, and executor utilization consistently
- the runtime can explicitly optimize for premium-call yield instead of letting task boundaries silently multiply paid executions

### Negative

- this introduces a new abstraction layer while the current `tasks` system still exists
- migration will temporarily require adapters between task handlers and work items
- some current task handlers are simple enough that the new abstraction will initially feel heavier than necessary
- ingest and other family-specific flows still need bespoke safety rules even after the common scheduler exists

## 7. Non-Goals for the First Slice

This ADR does **not** require the initial implementation to:

- replace the existing `tasks` table immediately
- merge all maintenance prompts into one universal agent prompt
- eliminate family-specific batch semantics
- convert ingest to the exact same execution protocol as curator or taxonomist
- redesign all current provider-routing logic before the work-item seam exists

## 8. Recommended Migration Order

### Phase 1 — Introduce the abstraction

- define durable work-item records and execution lanes
- keep current handlers, but allow them to enqueue or consume work-item batches through adapters
- add basic lease/heartbeat/defer primitives

### Phase 2 — Move low-risk agentic families first

Start with families that are easier to batch safely and have lower crash-safety complexity:

- `memory_tagging` (`taxonomist`)
- `graph_linking`
- possibly `conflict_review`

### Phase 3 — Move richer maintenance families

- `memory_curation`
- `memory_dedup`

By this point, the runtime should already have a clearer mutation ledger and per-batch observability.

### Phase 4 — Revisit ingest

Only after the shared work-item architecture is stable should ingest be reconsidered for deeper convergence.

## 9. Implementation Direction

The first implementation slice should be intentionally modest:

1. add a durable work-item schema and repository
2. define execution-lane enums and family keys
3. implement one internal batch tool for general work acquisition
4. migrate `taxonomist`-style tagging work first
5. preserve existing task handlers as compatibility orchestrators during migration

After that baseline is stable, the next architectural question is no longer just “can a family consume work items?” but also:

- how much work can one premium execution finish before another call is necessary?
- which boundaries still force unnecessary premium re-entry?
- which precomputation steps genuinely expand same-call yield versus merely improving latency?

This keeps the system moving toward the new architecture without forcing a risky all-at-once cutover.

## 10. Decision Summary

The preferred direction is:

- **one durable work model**
- **two execution lanes**
- **batch by work family**
- **lease-driven soft/hard stop semantics**
- **incremental migration, not a prompt-level big bang**

In short:

> unify scheduling and work ownership first; widen execution sharing only where the safety model truly matches.

And evaluate the result by **useful work per premium call**, not by runtime neatness alone.
