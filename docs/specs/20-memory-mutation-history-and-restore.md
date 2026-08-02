# Proposed Direction: Memory Mutation History, Protection, and Restore

**Status:** Proposed direction

## 1. Purpose

Autonomous curation requires more than proof that a mutation happened. Users and operators need to inspect what changed, understand why it changed, protect important records, and reverse a bad mutation without reconstructing old state from logs or model transcripts.

The curation harness therefore separates three concerns:

- `curation_action_receipts` prove execution, idempotency, and verification state
- immutable mutation history preserves reversible before/after state
- protection policy constrains what autonomous agents may inspect or change

Receipts remain compact. Reversible state belongs in a dedicated relational history model rather than hashes, prompt transcripts, or unbounded memory metadata.

This is a proposed direction, not the canonical description of current runtime behavior.

## 2. Relationship to Other Documents

- `00-product-principles.md` owns global-store and workspace semantics.
- `01-architecture-principles.md` requires relational-first persistence and conservative automation.
- `03-memory-management-spec.md` owns current memory fields, statuses, and lineage conventions.
- `04-temporal-fact-graph-spec.md` owns current typed relationship semantics.
- `06-database-schema-design.md` requires first-class relational data to remain outside opaque metadata.
- `08-system-resilience.md` owns durable recovery expectations.
- `10-testing-strategy.md` owns backend and test-layer expectations.
- `16-storage-backend-selection-and-shared-mode.md` owns backend authority.
- `17-shared-mode-readthrough-cache.md` owns derivative-cache behavior.
- `19-curation-quality-and-family-ownership.md` owns semantic policy, human intent, and maintenance-family boundaries.
- `../plans/10-curation-harness-and-typed-planner.md` owns typed planning, action transactions, receipts, and reconciliation.

## 3. Decision

Add a backend-parity, append-oriented mutation-history subsystem for memory and relationship changes.

Every autonomous content or structural mutation must:

1. validate current revision and protection state
2. capture the mutation-relevant before-state
3. apply the mutation and derivative search changes
4. capture the normalized after-state
5. persist the mutation event, revisions, and curation receipt in the same authoritative transaction

Restoring an earlier state creates a new mutation event. History is never rewritten to make a restore look as if the original action did not occur.

Manual/public mutations should adopt the same history path incrementally so audit behavior eventually reflects all meaningful memory changes, not only curator actions.

## 4. Invariants

1. **Authoritative storage only.** Mutation history is stored in the active authoritative backend. Shared-mode cache sidecars never own or queue restore operations.
2. **Atomic audit.** A mutation cannot commit without its history event and required before-state; history cannot claim a mutation that rolled back.
3. **Append-oriented history.** Applied mutation events and revision payloads are not edited in place except for narrowly defined terminal metadata fields.
4. **First terminal write wins.** A mutation event or restore request reaches one terminal outcome. Late conflicting terminal writes are rejected or treated as benign no-ops.
5. **Restore is a new mutation.** Reversal records its actor, reason, preconditions, and resulting revision.
6. **Current-state protection.** Restore uses revision preconditions and never overwrites later human or autonomous changes silently.
7. **Lineage survives.** Merge, split, archive, and restore preserve enough relationship and metadata state to reconstruct provenance.
8. **No history in generic metadata.** Core mutation history and protection rules use first-class relational tables.
9. **Global store semantics.** Workspace associations are restored as provenance metadata, not as tenancy boundaries.
10. **Delete stays exceptional.** Automatic hard deletion remains disabled until retention, tombstone, relationship, and restore requirements are proven.

## 5. Logical Data Model

Exact names may follow repository conventions, but the logical separation should remain.

### 5.1 `memory_mutation_events`

One row per applied or attempted terminal mutation action:

- `id`
- `operation`
- `actor_kind`: `user`, `maintenance`, `system`, or `restore`
- `actor_id` or family/task identity
- optional task, work-item, curation-run, plan, action, and provider references
- `reason_code` and concise rationale
- `policy_version` and schema version
- `status`: `applied`, `rejected`, `stale`, or `failed`
- optional `restores_event_id`
- timestamps

`(curation_run_id, action_id)` is unique when both values are present. Restore requests use their own idempotency key.

A restore is represented by `operation=restore`, `status=applied`, and `restores_event_id=<original event>`. The original event is not updated to a synthetic `restored` status.

### 5.2 `memory_record_revisions`

One or more rows per mutation event and affected memory:

- event ID and memory ID
- role such as `target`, `canonical`, `source`, `original`, or `split_child`
- before existence flag and normalized before snapshot
- after existence flag and normalized after snapshot
- before and after revision tokens

The normalized snapshot contains mutation-relevant state:

- ID, title, content, summary, type, and status
- sorted tags and workspace associations
- lineage and other mutation-relevant metadata

It excludes volatile retrieval telemetry such as `read_count`, `access_score`, `last_accessed_at`, and `last_surfaced_at`. Restoring semantic content must not erase later user-access telemetry.

### 5.3 `memory_link_revisions`

Rows preserve created, removed, or context-updated relationships:

- event ID
- source ID, target ID, normalized type, and context
- before and after existence flags

Compound operations capture every link whose existence or context changes, including lineage links.

### 5.4 `memory_protections`

Protection is first-class state keyed by memory ID and protection mode:

- mode
- reason
- actor and timestamps
- optional expiry

Initial modes:

- `no_autonomous_mutation`
- `no_autonomous_destructive_change`
- `manual_review_required`
- `local_provider_only`
- `no_external_provider_disclosure`
- `pinned_active`

Protection records do not alter workspace or ownership semantics. They express user/operator intent about automation.

## 6. Snapshot and Diff Policy

### 6.1 Canonical snapshots

Snapshots use versioned canonical JSON with stable key order, normalized tag/workspace ordering, and explicit null handling. Revision tokens use the same canonicalization contract as curation preconditions when possible.

### 6.2 Storage cost

History optimizes for correctness first, then bounded retention.

- store complete mutation-relevant snapshots for content-changing operations
- store compact field/relationship state for link-only or status-only operations
- avoid embedding provider transcripts, read telemetry, or unrelated metadata
- permit later compression at the storage layer without changing logical semantics

### 6.3 Operator diffs

The management layer computes or stores bounded diffs for display:

- changed scalar fields
- content and summary text diff
- tag/workspace additions and removals
- relationship additions and removals
- lineage and status changes

Large content remains paginated or truncated in list views; full authorized detail is available on demand.

## 7. Protection and Human Intent

### 7.1 Protection evaluation

Protection is checked before provider disclosure, planning authorization, and execution.

| Protection mode | Effect |
| --- | --- |
| `no_autonomous_mutation` | autonomous plans may inspect only if disclosure policy allows; every mutation is rejected |
| `no_autonomous_destructive_change` | normalization/link creation may be allowed; rewrite, link removal, merge, split, archive, and delete require review |
| `manual_review_required` | plans may be generated but cannot auto-apply |
| `local_provider_only` | context may be sent only to an eligible local provider |
| `no_external_provider_disclosure` | external planner context excludes the record entirely |
| `pinned_active` | autonomous archive/delete is rejected; other mutation modes still apply |

The strictest applicable protection wins.

### 7.2 Human edits

A user-authored content, status, tag, workspace, or relationship change:

- creates a mutation event with `actor_kind=user`
- resets candidate no-op cooldown for the changed revision
- starts a post-human-edit stabilization window for autonomous destructive changes
- invalidates stale plans through revision tokens
- does not automatically remove explicit protection

### 7.3 Missing or deleted targets

Protection rows for missing memories remain inspectable for a retention window so accidental deletion does not erase evidence of user intent. A later tombstone policy may make this permanent for protected records.

## 8. Restore Contract

### 8.1 Restore request

A restore request identifies:

- target mutation event
- requested scope: all affected state or a supported subset
- expected current revision tokens
- actor and reason
- idempotency key

The default is to restore the complete operation. Partial restore is allowed only for operation classes whose invariants remain valid.

### 8.2 Preconditions

Automatic restore requires:

- target event exists and is restorable
- current affected records/links match the target event's after-state tokens
- no later dependent mutation would be orphaned
- current protection allows the requesting actor
- required source records and relationship targets remain available

If current state diverged, return a conflict with an operator-readable diff. Do not force overwrite through the ordinary restore API.

### 8.3 Operation-specific restore

| Original operation | Restore behavior |
| --- | --- |
| normalize/rewrite | restore semantic before snapshot while preserving current access telemetry |
| create link | remove the exact edge if its current context still matches the event after-state |
| remove link | recreate the exact prior edge if endpoints still exist and policy allows it |
| archive | restore prior status and associated semantic state |
| merge | restore canonical and source snapshots plus affected links only when neither side has dependent later changes |
| split | restore original and reverse child/lineage changes only when children have no dependent later changes |
| delete | unavailable until tombstone and retention design is accepted |

Merge and split restore are high risk and initially require operator confirmation even when preconditions pass.

### 8.4 Compensating restore

When postcommit verification fails, the reconciler may automatically apply a compensating restore only if:

- the inverse is deterministic
- current tokens still match the failed action's after-state
- no dependent mutations exist
- protection policy permits system recovery
- the operation class is enabled for automatic compensation

Otherwise, preserve the applied event, mark the curation run `verification_failed`, block further actions, and create operator-review work.

### 8.5 Restore history

The restore event references the original event through `restores_event_id`. A second restore of the same current state is idempotent. Reapplying a previously restored mutation is a new explicit action, not history editing.

## 9. Lifecycle and Status Authority

Current statuses include `active`, `stale`, `degraded`, and `archived`. Restore must respect the family that authored status evidence:

- fact-checker degradation is not cleared by ordinary curation
- project-manager staleness is not cleared by a content-only rewrite
- archive restore returns the exact prior status, not always `active`
- `pinned_active` prevents autonomous archive/delete but does not fabricate status changes

The broader question of whether stale/degraded should become orthogonal dimensions is outside this document. Until then, mutation history must preserve exact prior status and actor provenance.

## 10. Transaction Integration

The curation action transaction writes:

1. domain memory/link changes
2. lexical search projection changes
3. version-aware embedding-repair intent when semantic fields changed
4. mutation event and record/link revisions
5. compact curation receipt

All authoritative relational writes that are required for recovery commit atomically. If embedding generation itself is asynchronous, only the durable version-aware repair unit or equivalent intent belongs in the action transaction.

After commit:

- invalidate or age out affected derivative read/search caches according to shared-cache policy
- run execution postcondition verification
- update receipt verification state

The history event remains authoritative evidence even if verification later fails and requires compensation.

## 11. Terminalization and Recovery

### 11.1 First terminal write wins

Mutation event terminalization and restore request terminalization use conditional writes. A late provider callback, task shutdown path, or reconciler cannot overwrite an existing terminal result with a conflicting state.

### 11.2 Crash recovery

At daemon recovery and before new curation work:

- reconcile `applied_unverified` receipts against committed history
- reconstruct missing run summaries from receipts/history when possible
- retry transient SQLite lock failures with a bounded policy
- treat late post-terminal observer writes as benign no-ops
- release or defer associated work items through the existing worker lifecycle

### 11.3 History integrity checks

Operator health should detect:

- applied receipts without mutation events
- mutation events without required revision rows
- restore events referencing missing source events
- revision-token mismatches between event after-state and committed records at event time
- protected memories missing their target record

## 12. Management API and UI

Minimum operator surfaces:

- list mutation events with filters for memory, actor, family, operation, and time
- inspect one event, policy reason, affected IDs, and before/after diff
- inspect restore eligibility and conflicts
- request restore with explicit confirmation for high-risk operations
- protect or unprotect a memory with reason
- show protection state on memory detail views
- link curation run, provider conversation, receipt, mutation event, and restore event

The audit stream displays stored facts and diffs. Provider reasoning may be linked for debugging but is not the source of truth.

## 13. Data Governance and Retention

Mutation history may contain sensitive content and must follow the same access controls and backend authority as memories.

Retention has two independent horizons:

- **restore horizon:** full inverse state remains available
- **audit horizon:** event metadata and hashes remain after full snapshots may be compacted or removed

Policy requirements:

- protected and recently mutated records receive a longer restore horizon
- history cleanup is an explicit deterministic operation, not ordinary memory sweeping
- cleanup never removes history needed by active restore, unresolved verification, or lineage dependencies
- Postgres backup/restore owns shared-mode durability; local cache files are irrelevant to history recovery
- operator exports should minimize provider transcript duplication

Exact default horizons remain an operator/product decision and belong in typed configuration.

## 14. Search and Cache Effects of Restore

Restore is a normal semantic mutation for derivative systems:

- update lexical search projections in the restore transaction
- enqueue version-aware embedding repair for restored content
- skip repair work for superseded revisions
- invalidate affected cached memory records/projections
- invalidate or bound staleness for exact cached query responses
- verify public search/read behavior from authoritative state

Maintenance and restore decisions never rely on degraded shared-cache data.

## 15. Testing Requirements

### Small tests

- canonical snapshot and token stability
- semantic fields included and access telemetry excluded
- protection precedence
- restore eligibility and conflict reasons
- first-terminal-write-wins behavior
- event/action and restore idempotency
- operation-specific inverse construction
- history retention eligibility

### Shared repository contract tests

Run the same behavior against SQLite and Postgres:

- atomic mutation plus event/revision persistence
- rollback leaves neither domain changes nor history claims
- unique curation action identity
- append/list/read history
- protection CRUD
- conditional terminalization
- restore precondition enforcement

### Medium tests

- rewrite then restore survives runtime restart
- link removal then restore recreates exact context
- archive restore returns the exact previous status
- later human edit blocks automatic restore
- merge/split restore detects dependent changes
- verification failure compensates only for enabled safe operations
- content restore queues one version-aware embedding repair unit
- shared-mode restore uses Postgres and never local cache/writeback
- management API returns bounded before/after diffs

### Large tests

- thought capture -> ingest -> curation -> search/read -> audit -> restart -> restore -> search/read
- daemon crash after mutation commit but before verification recovers from history and receipts

## 16. Rollout

### Phase 1: History primitives

- canonical snapshots and revision tokens
- mutation event/revision schema for both backends
- repository contract tests
- no production behavior change

### Phase 2: Low-risk history integration

- record normalize, link-create, and link-remove mutations
- expose read-only management history API
- add protection state and enforcement

### Phase 3: Restore low-risk operations

- rewrite/status/link restore with strict current-token checks
- operator API and basic diff UI
- recovery integrity checks

### Phase 4: Compound history

- merge and split snapshots/link revisions
- operator-confirmed compound restore
- dependency detection

### Phase 5: Retention and broader adoption

- restore/audit horizons and cleanup
- incremental adoption by manual/public mutation paths
- evaluate tombstones and hard-delete policy separately

## 17. Acceptance Criteria

- Every enabled autonomous semantic mutation commits reversible history atomically with the mutation.
- Receipts remain compact and link to history rather than duplicating full snapshots.
- User protection state is enforced before disclosure, planning, and execution.
- Restore never silently overwrites later changes.
- Restoring creates a new auditable mutation event.
- SQLite and Postgres pass one shared history/protection/restore contract.
- Search projections, embedding repair, and derivative caches follow restored state.
- Operators can inspect a bounded before/after diff without reading provider prose.
- First terminal mutation/run outcomes remain authoritative under late callbacks and recovery.
- Automatic hard delete remains disabled until a separate accepted tombstone/retention design exists.

## 18. Non-Goals

- Full database point-in-time recovery.
- General event sourcing of every operational table.
- Restoring volatile read/access telemetry.
- Treating provider transcripts as reversible history.
- Bypassing current-state preconditions to force an old snapshot over newer edits.
- Generalized offline mutation or cache writeback.
- Enabling automatic hard delete in the initial rollout.

## 19. Open Decisions

1. Default restore and audit horizons.
2. Whether canonical snapshots are compressed in application code or storage.
3. The initial automatic-compensation allowlist.
4. Whether protection changes themselves require a separate immutable audit event.
5. The dependency model used to decide whether compound merge/split restore is still safe.
6. The eventual tombstone model required before any hard-delete rollout.
