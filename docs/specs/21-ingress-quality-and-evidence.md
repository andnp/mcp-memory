# Proposed Design: Ingress Quality and Evidence

**Status:** Proposed design

This document defines the next hardening boundary for `ingest-system1`.
Ingress already has stronger journal claim and recovery semantics than most
maintenance families. This proposal preserves those semantics while carrying
over the curation system's identity, idempotency, evidence, and quality
discipline.

It is intentionally additive. It does not make ingress another curator
campaign, replace the journal with generic work items, or treat a successful
task run as proof that the memory base improved.

## 1. Problem

The current ingress path is operationally durable but its evidence is split
across several authorities:

- `system1_journal` owns pending, claimed, recoverable, and released source
  entries;
- the task row and `task_runs` own execution state and result summaries;
- internal-tool tracking owns observed tool calls and mutation counts;
- direct mutation evidence may own before/after entity deltas; and
- memory metadata owns ingest lineage.

Those authorities are not yet reconciled into one ingress outcome. The current
path therefore has four important failure modes:

1. agentic ingress resets tool tracking without the task's `execution_epoch`,
   so retry attribution can diverge from the task and provider-attempt ledger;
2. a replayed create or append can repeat the memory write because the
   ingest-specific mutation boundary has no action-level idempotency guard;
3. deterministic action-analysis failures fall back to raw observation writes
   without an explicit degraded execution classification; and
4. task-local dispositions and mutation evidence do not independently prove
   durability, retrieval utility, or useful work.

The desired result is a trustworthy answer to four separate questions:

| Question | Evidence authority |
| --- | --- |
| Did the worker execute the intended ingress attempt? | task execution epoch, attempt ledger, provider usage |
| Did a memory mutation happen exactly once? | ingress action receipt, direct mutation evidence, mutation history |
| Were all claimed source entries accounted for? | source coverage and journal finalization |
| Did the resulting memory improve the product? | durable quality evidence or explicit `unobserved` reason |

No one of these answers should be inferred from another.

## 2. Design principles

1. **Preserve journal authority.** Ingress claims remain task-scoped and
   crash-safe. A generic scheduler or curator ledger must not consume source
   entries on ingress's behalf.
2. **Use one identity per boundary.** Provider attempts, ingress actions, and
   quality evaluations have related but distinct identities.
3. **Make replay safe before making it fast.** A timeout after commit must be
   safe to retry with the same logical action.
4. **Receipts prove store change, not quality.** A verified write is not a
   productive memory outcome without an independent quality disposition.
5. **Prefer explicit degradation to silent success.** Fallbacks may preserve
   durability, but their cost and quality status must remain visible.
6. **Keep source coverage exact.** Every claimed entry ends in one durable
   disposition, including an explicit unobserved or released reason.
7. **Keep the first rollout reversible.** New evidence should be additive and
   should not require destructive journal cleanup or broad mutation admission.

## 3. Scope and non-goals

### In scope

- execution identity propagation through agentic ingress;
- deterministic batch and action identities;
- idempotent create and append mutation receipts;
- source-entry snapshots and exact coverage;
- reconciliation between task, provider, tool, mutation, and quality evidence;
- deterministic ingress durability-risk signals;
- ingress-specific operator metrics and rollout gates; and
- SQLite/Postgres parity and restart/concurrency tests.

### Out of scope

- replacing `System1Journal` with the shared work-item store;
- making ingress use curator plan/action schemas unchanged;
- adding unrestricted merge, archive, delete, or graph maintenance authority;
- introducing random selection into FIFO journal claims;
- deciding whether a memory should ultimately be retained or deleted; and
- making retrieval evaluation synchronous with every write.

The existing work-item architecture remains the long-term convergence point,
but its own design says ingress should retain a family-specific contract until
the shared lease and recovery semantics are mature.

## 4. Identity model

Ingress uses four identity levels.

### 4.1 Execution identity

The worker-owned execution identity is:

```text
(task_id, execution_epoch)
```

Every ingress tracker reset, internal tool call, direct mutation evidence row,
provider usage sample, and task execution attempt must carry this pair. The
handler must pass `task.execution_epoch` when initializing agentic tracking;
the default epoch must never be used for a worker-owned run.

An execution epoch changes on retry or re-claim. Late callbacks from an older
epoch may record diagnostic failure evidence, but may not mutate current task
state or finalize current journal claims.

### 4.2 Batch identity

Each call that claims journal entries creates a deterministic batch identity:

```text
batch_id = hash(canonical_claimed_entry_ids, source_fingerprint)
```

`batch_sequence` and the current execution identity are stored as telemetry,
but are not part of the stable `batch_id`. A retry or re-claim of the same
immutable journal entries therefore addresses the same logical batch.

The batch records:

- the ordered claimed entry IDs;
- a canonical source fingerprint;
- each entry's workspace and timestamp;
- the grouping strategy requested and used;
- the grouping fallback reason, if any;
- the provider route and execution mode; and
- the policy/schema version used to build the prompt and payload.

The source fingerprint is calculated from normalized entry IDs, timestamps,
workspace IDs, and content digests. It is replay evidence, not a replacement
for the source snapshot.

### 4.3 Action identity

Each logical create, append, or future ingress mutation gets an action identity:

```text
action_id = hash(operation, canonical_entry_ids, canonical_target_ids)
```

The canonical payload includes the final title, content, summary, tags,
workspace IDs, and relevant metadata. Entry and target lists are normalized,
deduplicated, and sorted before hashing.

`action_id` is the stable replay key. The batch ID, execution identity, and
canonical payload digest are stored on the receipt for attribution and collision
checking, but do not change the action identity. The payload description above
is receipt data, not another input to the action hash. A replay with the same
action identity and payload digest returns the original receipt unchanged. A
second request with the same action identity and a different canonical payload
fails closed as an identity collision; it must not receive a new action ID.

The source-coverage ledger also rejects a new action whose entry set overlaps a
terminal action with a different identity. This prevents a retry that regroups
the same journal entries from creating a second partial memory.

### 4.4 Quality identity

Quality evidence is keyed by the mutation evidence ID and action identity, not
by provider request ID. Provider attempts reconcile at the execution boundary;
quality evidence reconciles at the mutation boundary. These identities must
not be joined directly.

## 5. Proposed durable records

The implementation may reuse existing storage seams where they provide the
required guarantees, but the logical records are:

### 5.1 Ingress batch evidence

One record per claimed batch:

```text
    batch_id
task_id
execution_epoch
batch_sequence
source_fingerprint
source_entries[]
grouping_strategy
grouping_fallback_reason
provider_route
execution_mode
policy_version
claimed_at
finalized_at
```

`source_entries[]` contains the entry ID, workspace ID, timestamp, content
digest, and a bounded source snapshot sufficient for replay and audit. It must
not contain provider transcripts.

### 5.2 Ingress action receipt

One record per logical mutation or explicit no-op:

```text
batch_id
action_id
operation
entry_ids[]
target_ids[]
canonical_payload_digest
status
mutation_evidence_id
before_revision_tokens
after_revision_tokens
created_at
terminalized_at
error_code
```

Terminal statuses distinguish at least `applied_unverified`, `no_op`,
`stale`, `failed`, and `unobserved`. `replayed` is a response projection; the
stored terminal action remains unchanged.

The database enforces uniqueness for `action_id` and for each terminal source
entry assignment. The mutation
receipt and authoritative memory write must commit atomically. Direct evidence
written after the domain transaction remains linked to the receipt and may
cause the quality projection to become `unobserved`, but it may not cause a
second write.

### 5.3 Source coverage

Coverage is computed against the batch's expected entry set, not against the
provider's reported output. Each claimed entry appears exactly once in the
authoritative final projection with one of:

- `created`;
- `appended`;
- `matched_existing`;
- `ignored`;
- `no_mutation`;
- `unobserved` with a reason.

`released_unhandled` is a journal lifecycle projection, not terminal source
coverage. Those entries return to `pending` without coverage credit and may be
claimed again.

Provider-reported outcomes are retained as advisory evidence. They cannot
replace the handler's claimed-entry set or task-data/tool evidence.

## 6. Execution and finalization protocol

The execution boundary is deliberately separate from journal finalization.

```text
claim batch
    -> persist batch evidence and source snapshot
    -> inspect/group/search
    -> reserve action identity
    -> apply memory mutation exactly once
    -> persist receipt, lineage, history, and repair intent atomically
    -> verify postcondition in a fresh read
    -> persist quality evidence or unobserved reason
    -> move handled entries to recoverable
    -> release untouched entries
```

Rules:

1. Claiming is atomic and remains owned by `System1Journal`.
2. A failed provider call does not consume claims.
3. A deterministic fallback may create a raw observation for durability, but
   it must be labeled `execution_mode=deterministic_fallback` and
   `fallback_reason=provider_analysis_failed` (or a more specific reason).
4. Fallback writes receive store-change evidence but no productive quality
   credit until independently evaluated; their initial quality disposition is
   `unobserved` with the fallback reason.
5. A mutation that commits before its response is lost is recovered by replay
   of the same action identity, not by issuing a new create or append.
6. A task retry uses a new execution epoch but may replay the prior batch/action
   identity. The committed receipt wins and the memory write is not repeated.
7. Only entries proven handled by authoritative mutation or explicit terminal
   no-op evidence move to recoverable. Untouched entries return to `pending`.
8. Replay-safe mutation enforcement requires one backend transaction owning the
   domain write, receipt, lineage, repair intent, source coverage, and handled
   journal transition while the journal API remains the policy owner.
9. If a backend cannot provide that combined transaction during migration, it
   cannot claim the Phase 2 exactly-once contract. Keep that backend in shadow
   or detection mode, record the limitation as `unobserved`, and do not enable
   unattended replay-safe mutation until an atomic boundary exists. A startup
   reconciler may repair expired claims from committed receipts, but must be
   idempotent and must never infer handling from memory metadata alone.
10. Recoverable retention and purge policy remain explicit; documentation must
   match implementation rather than describing recoverable rows as deleted.

## 7. Quality and durability contract

Ingress should provide deterministic risk context before asking the provider to
choose a landing spot. The initial signals reuse the curation durability
vocabulary:

- `dated_work_log`;
- `task_completion_residue`;
- `transient_execution_detail`;
- `mixed_durability_content`;
- `raw_ingress_candidate` when a target is already ingest-created and weakly
  titled, summarized, tagged, or oversized; and
- `cross_workspace_scope` when a proposed landing spot would broaden source
  provenance unexpectedly.

Signals are advisory context, not automatic retention decisions. Durable dates,
releases, incidents, deadlines, decisions, and historical facts must suppress
the corresponding transient-risk flag when the content clearly preserves that
meaning.

Every applied action receives one quality disposition:

| Disposition | Meaning | Productive credit |
| --- | --- | --- |
| `productive` | Independent content, durability, or retrieval evidence passed. | Yes |
| `structural_only` | Source coverage, lineage, or structural contract passed without claiming retrieval improvement. | Only if policy permits. |
| `neutral` | Quality was observed with no material change. | No. |
| `regressed` | Durable meaning, provenance, scope, or retrieval utility was harmed. | No; stop or escalate. |
| `verified_only` | Store state is correct but quality was not demonstrated. | No. |
| `unobserved` | Evaluation was unavailable or incomplete, with a required reason. | No. |

The initial implementation may evaluate content/durability immediately and
retrieval utility asynchronously. It must preserve the distinction between
"not yet evaluated" and "neutral". Inline finalization records `unobserved`
when asynchronous retrieval evidence is not available; it does not wait for
that evidence or convert the mutation into a failure.

Receipt state and quality state are separate. A committed memory write remains
`applied_unverified` when fresh postcondition verification is unavailable or
fails; it is not rewritten to `failed` after commit. The postcondition and
quality evaluator records `unverified` or `unobserved` separately. `failed`
means the mutation did not commit.

## 8. Reconciliation and operator gates

Ingress reporting must keep these counters separate:

- claimed, handled, recoverable, released, and unobserved entries;
- action attempts, applied receipts, replayed actions, no-ops, stale actions,
  failed actions, and direct-evidence failures;
- provider attempts, failures, retries, budget skips, and token totals;
- quality dispositions and their reasons; and
- pending, recoverable, and aged journal backlog.

The operator audit joins, for every execution epoch:

1. task status and `task_runs`;
2. task execution attempts and provider usage;
3. internal tool ledger and runtime errors;
4. ingress batch/action evidence;
5. direct mutation evidence and mutation history;
6. source coverage and journal finalization; and
7. quality evidence and downstream retrieval observations.

The following conditions fail closed for unattended ingress mutation:

- missing or mismatched execution identity;
- duplicate or incomplete source coverage;
- action receipt without matching mutation evidence or an explicit unobserved
  reason;
- provider/tool/task count disagreement;
- an idempotency collision;
- a direct-evidence persistence failure;
- a destructive or cross-scope mutation without an approved policy path; or
- quality regression above the versioned canary threshold. This threshold is
  evaluated asynchronously over a defined moving window and acts as an
  admission circuit breaker: it stops new unattended claims, but does not roll
  back or block an already committed mutation.

Receipts, recoverable claims, and current memory stock are not substitutes for
quality evidence.

## 9. Rollout plan

### Phase 0 — Identity correction

- pass `task.execution_epoch` into agentic ingress tracking;
- include the epoch in internal tool and provider joins; and
- add retry/late-callback tests.

This is a correctness fix and should land before new reporting is trusted.

### Phase 1 — Evidence-only shadow

- persist batch source fingerprints and snapshots;
- derive deterministic action identities;
- record receipts and direct evidence without changing claim finalization; and
- compare task results with evidence in a read-only audit.

Additive schema migrations create nullable evidence tables and unique indexes
without changing legacy claim behavior. Shadow writes may be unavailable
without blocking legacy ingress, but the unobserved reason is counted.

### Phase 2 — Replay-safe mutations

- enforce unique action identity in the mutation boundary;
- make create and append replay return the original receipt;
- preserve before/after tokens, lineage, and version-aware embedding repair
  intent in the same transaction; and
- add SQLite/Postgres concurrency and crash-window coverage.

The replay lookup remains enabled after receipt creation, even if later
evidence writes are disabled. A safe rollback is mutation admission off or
shadow mode with replay lookup preserved; disabling the lookup and returning
to an unguarded legacy writer is not a safe rollback after Phase 2.

### Phase 3 — Quality signals and fallback visibility

- add deterministic durability flags to batch payloads and prompts;
- classify deterministic fallback explicitly;
- persist quality dispositions or unobserved reasons; and
- add dashboard/CLI projections that do not infer quality from mutation counts.

### Phase 4 — Bounded canary

- keep destructive and broad refactoring operations disabled;
- run manual canaries before enabling unattended ingress;
- require complete reconciliation and zero unexplained attribution gaps; and
- increase batch, provider, and schedule bounds one dimension at a time.

### Phase 5 — Revisit shared work items

Only after the ingress evidence and replay contracts are stable should the
family-specific batch protocol be adapted to the shared work-item lease model.
The journal claim/finalize contract remains the compatibility boundary during
that migration.

## 10. Verification strategy

### Small tests

- execution epoch propagation into tracker and direct evidence;
- canonical batch/action identity normalization;
- duplicate action payload collision handling;
- exact claimed-entry coverage and duplicate/out-of-scope rejection;
- fallback classification and quality-credit exclusion;
- durability-risk flag boundaries; and
- empty, no-op, ignored, and partially handled batches.

### Medium tests

- create/append replay after a committed-but-lost response;
- concurrent duplicate action execution in SQLite and Postgres;
- task retry with a new execution epoch and the old committed receipt;
- crash recovery with handled and untouched claims;
- reconciliation across provider usage, task results, tool ledger, and evidence;
- embedding repair intent and search visibility after replay; and
- source retention through recoverable purge boundaries.

The crash-window test must cover a process failure after the memory/receipt
commit and before journal finalization. Restart must reconcile the committed
receipt and move handled entries to `recoverable` without reapplying the action.

### Large tests

- end-to-end thought capture → ingress → search/read → audit → restart;
- writeback outbox flush followed by exactly-once ingress action application;
- bounded canary stop conditions; and
- backend parity for identity, receipt, coverage, and recovery behavior.

Passing tests prove the contracts above. Mutation counts, completed task runs,
or provider-reported summaries alone do not.

## 11. Ownership and implementation boundaries

| Concern | Primary owner |
| --- | --- |
| Journal claim/release/recoverable lifecycle | `System1Journal` and ingress handler |
| Execution epoch and provider-attempt identity | task worker and provider telemetry |
| Batch/source/action evidence | ingress evidence store |
| Atomic create/append mutation and receipt | backend-neutral ingress mutation store |
| Direct before/after mutation evidence | MCP transport/evidence store |
| Quality policy and disposition evaluation | curation-quality seam, with ingress-specific inputs |
| Retrieval observations | retrieval telemetry and quality evaluator |
| Rollout admission and stop conditions | operator/runbook layer |

The ingress mutation store may share transaction primitives with the curation
action store, but ingress remains the owner of source-entry consumption. A
shared primitive is acceptable; a shared family contract is not required.

## 12. Open decisions

1. Whether source snapshots should be retained in the ingress evidence table,
   a versioned source table, or an existing mutation-history payload.
2. Whether fallback-created observations should remain recoverable until their
   first quality evaluation or use the current fixed retention window.
3. Which retrieval-quality sample size and delay provide useful evidence without
   making every ingress write wait on embedding/search work.
4. Whether the first mutation store should support only create/append or also
   the currently exposed adjacent cleanup tools.

These decisions should not block Phase 0 or the evidence-only shadow.

## 13. Retention and migration safety

Source snapshots are evidence, not an unbounded transcript archive. Retain a
complete normalized source snapshot, preferably compressed, until the later of:

- the journal recoverable-retention deadline;
- the quality-evaluation deadline; and
- closure of any reconciliation or operator review for the action.

After that point, the snapshot may be pruned while retaining the source digest,
entry IDs, action receipt, and final coverage. A configured snapshot-size limit
must fail the action into an explicit `unobserved`/`source_snapshot_truncated`
state rather than silently claiming replayability.

Schema rollout is additive:

1. migrate nullable evidence and receipt tables plus unique indexes;
2. deploy shadow writers and read-only reconciliation;
3. verify backend parity and crash recovery;
4. enable replay enforcement; and
5. enable quality gates and unattended admission last.

The logical feature controls are:

- `ingress_evidence_mode = off | shadow | enforce`;
- `ingress_replay_policy = legacy | detect | enforce`; and
- `ingress_quality_admission = disabled | manual | canary`.

`detect` records collisions and mismatches without applying the new rejection
path. `enforce` rejects an identity collision before any domain write. If an
evidence store is unavailable, `shadow` continues legacy ingress with an
explicit unobserved reason; `enforce` blocks new mutation admission and keeps
claims recoverable. No rollback mode deletes evidence or bypasses replay
lookup for actions that may already have committed. `legacy` is valid only
before replay enforcement has been activated; startup must reject `legacy` (or
raise the floor to `detect`) once committed action receipts exist. After Phase
2, the minimum safe rollback is detection with replay lookup preserved.

## 14. Deterministic durability precedence

Durability flags are computed by deterministic lexical/metadata rules before
prompt assembly. They are not suppressed by provider prose.

| Flag | Set when | Suppression/relationship |
| --- | --- | --- |
| `dated_work_log` | a date marker and work-log marker are both present | suppress only when an approved durable-date marker is present |
| `task_completion_residue` | completion/status markers are present | remains visible even when durable content is also present |
| `transient_execution_detail` | temporary execution/debug markers are present | remains visible even when durable content is also present |
| `mixed_durability_content` | durable-content markers coexist with status or execution markers | records the mixed case; it does not replace the component flags |
| `raw_ingress_candidate` | ingest lineage exists and weak tags, generic summary, or oversized content is present | metadata and content thresholds are deterministic |
| `cross_workspace_scope` | source workspace set and proposed target scope differ materially | requires explicit provenance or a new focused memory |

This precedence makes benchmark payloads and prompts reproducible across
providers and retries.
