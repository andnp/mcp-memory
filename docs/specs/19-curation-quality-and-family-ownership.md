# Proposed Direction: Curation Quality and Maintenance-Family Ownership

**Status:** Proposed direction

## 1. Purpose

The deterministic curation harness defines how maintenance work is planned, validated, executed, verified, and recovered. This document defines a different boundary: what good curation means, which maintenance family owns each class of work, and how the system determines whether safe mutations actually improved the memory base.

The distinction matters:

- execution verification proves that storage matches an approved plan
- quality validation decides whether that plan preserves meaning and improves the product
- ownership rules prevent several maintenance agents from repeatedly undoing or duplicating one another's work

This is a proposed direction, not the canonical description of current runtime behavior. Active product, storage, and runtime specifications remain authoritative.

## 2. Relationship to Other Documents

- `00-product-principles.md` owns the global-store and workspace-metadata invariants.
- `03-memory-management-spec.md` owns the current memory model, search behavior, and maintenance roster.
- `04-temporal-fact-graph-spec.md` owns the current relationship model.
- `07-background-task-architecture.md` owns durable task behavior.
- `10-testing-strategy.md` owns test-layer expectations.
- `12-nerd-metrics-analytics.md` owns retrieval and maintenance analytics contracts.
- `14-work-item-execution-architecture.md` owns the shared work-item direction.
- `15-search-repair-queue-architecture.md` owns embedding-repair behavior.
- `16-storage-backend-selection-and-shared-mode.md` and `17-shared-mode-readthrough-cache.md` own backend authority and cache boundaries.
- `20-memory-mutation-history-and-restore.md` owns reversible history, protected memories, and restore behavior.
- `../plans/10-curation-harness-and-typed-planner.md` owns the planner/executor runtime design.

## 3. Decision

Adopt an explicit, versioned curation quality policy with four responsibilities:

1. define a strict objective hierarchy for memory changes
2. require operation-specific semantic evidence and preservation rules
3. assign primary ownership for overlapping maintenance work
4. close the loop between selection, retrieval outcomes, write-stream defects, and future maintenance

The policy is evaluated before a typed plan is authorized. Model confidence is advisory. A mutation that is mechanically valid but violates the quality policy is rejected or routed to a specialist.

## 4. Quality Objective Hierarchy

Objectives are ordered. A lower objective never justifies violating a higher one.

1. **Preserve durable meaning and provenance.** Do not lose claims, qualifications, source lineage, workspace associations, temporal context, or intentional relationships.
2. **Prevent destructive false positives.** An unnecessary no-op is cheaper than an incorrect merge, rewrite, archive, or delete.
3. **Respect human intent and protection state.** User-authored edits, protection rules, and review requirements constrain automation.
4. **Improve retrieval usefulness.** Prefer changes that make relevant memories easier to find, recognize, and read while reducing misleading results.
5. **Improve structural coherence.** Reduce duplication, fragmentation, malformed lineage, weak taxonomy, and incorrect links.
6. **Reduce avoidable storage and context cost.** Compact filler and redundancy only after meaning and retrieval value are preserved.
7. **Optimize provider economics.** Increase useful work per premium request without lowering safety or quality thresholds.

Mutation count, transcript length, elapsed time, and model-reported action count are not quality objectives.

## 5. Memory-Type Semantics

The same operation has different risk depending on memory type.

| Memory type | Primary value | Default curation stance |
| --- | --- | --- |
| `journal` | temporal source material and short-lived context | consolidate when durable meaning is extracted; preserve source lineage; avoid promoting routine task traces |
| `observation` | focused evidence, findings, and reusable notes | preserve specificity and provenance; merge only with strong subject identity; add missing taxonomy when unambiguous |
| `fact` | durable claim intended for reuse | treat rewrite, merge, archive, and contradiction resolution as high risk; preserve sources, qualifiers, and temporal scope |
| `reflection` | synthesis across evidence or events | preserve the synthesis boundary and links to supporting material; do not merge merely because themes overlap |
| `plan` | intended future work and decisions | preserve status, decision context, and supersession; route aging to project-management policy rather than treating age as low value |

Unknown or future memory types use the most conservative policy until explicitly classified.

## 6. Semantic Preservation Contract

### 6.1 Evidence references

Every nontrivial proposed action identifies its evidence through memory IDs, relationship assertions, and revision tokens. Evidence text may be included only as bounded excerpts tied to those IDs. Free-form rationale without source references is not sufficient.

### 6.2 Claim manifest

`RewriteMemoryAction`, `MergeMemoriesAction`, `SplitMemoryAction`, and `ArchiveMemoryAction` include a bounded claim manifest:

- `preserved_claims`: durable claims that remain represented after the action
- `transformed_claims`: claims whose wording or structure changes without changing meaning
- `omitted_material`: material intentionally removed, with a typed reason
- `source_mapping`: output record or field mapped to its source memory IDs
- `unresolved_tensions`: contradictory or ambiguous claims that were not collapsed

The manifest is planning and audit evidence. It does not make model assertions authoritative, but it gives deterministic policy and later review a concrete preservation surface.

### 6.3 Omission reasons

Allowed omission reasons are narrow:

- duplicate wording already preserved in the output
- routine execution trace with no durable information
- formatting or boilerplate noise
- content moved intact to another split child
- explicit superseded material retained through lineage

Uncertainty, contradiction, inconvenient detail, and workspace mismatch are not omission reasons.

### 6.4 Contradiction handling

The curator does not silently choose a winner between incompatible durable claims.

- preserve both claims
- create or retain `CONTRADICTS` evidence when justified
- route unresolved fact conflicts to the conflict-review family
- use `AMENDS` or `SUPERSEDES` only when temporal/source evidence supports that relationship
- require human review when the system lacks enough provenance to determine the relationship

Until richer temporal fact semantics exist, apparently obsolete facts should be linked, degraded, or reviewed rather than deleted merely because they are old.

## 7. Operation-Specific Quality Rules

### 7.1 Normalize title, summary, or tags

- Preserve content and claim scope.
- Make summaries concrete enough to distinguish the memory in search results.
- Do not replace specific terminology with generic taxonomy.
- Apply only approved canonical tags; global tag renaming belongs to the taxonomist.
- Verify that lexical search projections reflect the normalized fields.

### 7.2 Rewrite one memory

- Require one coherent subject before and after the rewrite.
- Preserve every durable claim or identify an allowed omission reason.
- Preserve provenance, lineage metadata, workspace associations, and intentional links.
- Reject rewrites whose primary effect is stylistic churn.
- Keep automatic content rewriting in shadow mode until semantic-preservation evaluation passes.

### 7.3 Create or remove a link

- Require exact endpoint and type evidence.
- Normalize stable uppercase link types.
- Do not infer subject identity from generic vocabulary alone.
- Route contradiction links through conflict policy.
- Preserve the removed edge in mutation history so it can be restored.

### 7.4 Merge memories

- Require strong subject identity, not merely topical similarity.
- Require compatible claim scope and preserved provenance.
- Retain disagreements and temporal qualifiers rather than averaging them away.
- Choose the canonical record using explicit factors such as durability, source support, retrieval use, and lineage, not arbitrary ordering.
- Archive the source and create `SUPERSEDES`; do not hard-delete it.
- Preserve all source workspace associations as provenance metadata, not authorization boundaries.

### 7.5 Split a memory

- Require either multiple durable subjects or an oversized record whose retrieval/readability improves through decomposition.
- Ensure every durable original claim maps to at least one child or remains on the active original.
- Avoid thin children that cannot stand alone in search results.
- Preserve parent, sibling, ordering, and source lineage.
- Do not split focused records solely to meet an arbitrary size target.

### 7.6 Archive and delete

- Archive only when active retrieval is more harmful than retaining the record as active.
- Preserve enough history and lineage for inspection and restore.
- Do not archive solely because a record is cold, old, cross-workspace, or rarely read.
- Keep automatic memory deletion disabled until the restore window, lineage checks, and operator policy are proven.

## 8. Maintenance-Family Ownership

The shared harness may execute actions for several families, but discovery and planning policy remain family-specific.

| Work class | Primary family | Curator role |
| --- | --- | --- |
| System 1 promotion and source-entry handling | ingestor | none; preserve ingest claim/finalize semantics |
| Routine summary generation | summarizer | route missing/stale summaries; rewrite only as part of a broader approved structural action |
| Global tag ontology and synonym collapse | taxonomist | use approved tags locally; route ontology changes |
| Ordinary semantic relationship discovery | graph linker | create lineage links required by curator actions; route unrelated graph opportunities |
| Contradictory durable claims | conflict detector/review | retain evidence and route; do not resolve silently |
| Exact/near duplicate fact consolidation | deduplicator | handle exceptional mixed structural cases only when dedup policy cannot |
| Journal/reflection consolidation | defragmenter | handle mixed-type or split/reshape cases outside defragmenter scope |
| Oversized or mixed-topic record decomposition | curator | primary planner |
| Local title/summary/tag cleanup attached to curation | curator | primary planner within approved ontology and preservation rules |
| Fact degradation based on external evidence | fact checker | preserve status and route; do not overwrite degradation without new evidence |
| Plan aging and staleness | project manager | preserve lifecycle state and route |
| Operational telemetry retention | sweeper | no memory-content role |
| Memory archive/delete | curator plus operator policy | archive only under high-risk policy; delete disabled initially |

`needs_different_specialist` is a routing result, not a no-op. It creates an idempotent work item for the primary family when that family has an execution contract.

## 9. Convergence and Anti-Oscillation

Revision tokens prevent stale concurrent updates, but sequential agents can still create churn. The following rules apply across maintenance families:

- persist the originating family, policy version, action class, and semantic before/after tokens for every mutation
- suppress unchanged records from all autonomous mutation families for a short post-mutation stabilization window
- reset suppression when a human edit, new source evidence, relationship change, or materially new retrieval signal arrives
- use deterministic canonicalization for summaries, tags, link types, and lineage metadata
- do not reverse another family's recent action without changed evidence and an explicit reversal reason
- deduplicate specialist work by family, target revision, and reason
- report repeated A-to-B-to-A transformations as oscillation incidents
- escalate repeated cross-family disagreement to operator review rather than spending additional premium calls

Family-specific cooldown is allowed, but the shared mutation ledger is the source for cross-family stabilization.

## 10. Candidate Selection and Corpus Coverage

### 10.1 Global-first selection

Selection operates over the global corpus. Workspace associations may contribute soft provenance and ranking evidence but may not hard-partition maintenance work.

Curator seed and support acquisition therefore pass no task-derived workspace filter. Workspace IDs remain available on candidate records for ranking, provenance, disclosure, and analytics.

### 10.2 Bounded queries over the whole corpus

Strategy roulette must not be limited to a recent in-memory window. Backends should expose bounded candidate queries for:

- semantic clusters
- cold storage
- never surfaced
- oversized and thin anomalies
- graph bridges
- orphan/low-support records
- conflict frontiers
- low search-to-read conversion
- quality-signal violations
- stable seeded random samples

Each query returns compact candidates and a reproducible selection explanation. The planner still receives only a bounded frontier.

### 10.3 Coverage ledger

Candidate state records:

- last considered time and revision
- last selected strategy
- disposition and cooldown
- last mutation family and time
- escalation count

Coverage reporting should answer what share of eligible active memories has been considered within a configurable horizon, segmented by type and strategy.

### 10.4 No self-instrumenting maintenance reads

Planner investigation uses authoritative, non-instrumenting maintenance reads.

- do not increment user-facing `read_count`
- do not change `access_score` or `last_accessed_at`
- do not change `last_surfaced_at`
- do persist separate internal retrieval events and planner budget counters
- do not use shared-mode degraded cache results for mutation decisions

This prevents maintenance activity from manufacturing the retrieval signals used to select and evaluate memories.

## 11. Retrieval and Write-Stream Feedback

### 11.1 Retrieval outcomes

Use persisted retrieval analytics as evidence, not as an automatic deletion rule.

- repeated surfacing with low read-through can indicate a misleading title/summary or poor ranking fit
- zero-result query families can identify missing terminology, links, or coverage
- high read-through can identify canonical records worth preserving
- cold or never-surfaced records require content/graph inspection before archive decisions

### 11.2 Before/after evaluation

For shadow and sampled production runs, capture relevant historical or golden queries and compare:

- rank of intended memories
- irrelevant top-result rate
- zero-result behavior
- search-to-read proxy where available
- payload size and focusedness
- graph and lineage integrity

Query replay is an evaluation tool, not a production request-path dependency.

### 11.3 Prevent defects at the source

The system should distinguish active-stock cleanup from incoming write quality. Repeated defects are attributed to the creating task, tool, provider, or mutation path when available.

Examples:

- generic summaries repeatedly produced by one writer
- task-completion traces stored as memories
- untagged observations from one ingest path
- oversized records repeatedly created and later split

The curator may repair existing stock, but repeated producer defects should create an idempotent remediation signal for the source path. Quality reporting should include both repaired defects and prevented/repeated defects.

## 12. Provider Data Eligibility

Mutation authorization and provider disclosure are separate policy decisions. A record can be eligible for local deterministic cleanup but ineligible for an external planner.

Before constructing a provider context packet, evaluate:

- provider trust class and operator allowlist
- record protection state
- local-provider-only requirements
- sensitive tags or metadata markers
- credential/secret detection and redaction
- minimum necessary metadata and relationship context
- maximum disclosed characters or tokens

The curation run records which memory IDs and fields were disclosed to which provider. It should not duplicate the full disclosed content when existing provider transcript storage already provides an appropriately protected audit record.

Ineligible frontiers route to a local planner, deterministic specialist, or operator review. They are not silently skipped forever.

## 13. Evaluation and Rollout Gates

Planner evaluation is separate from executor correctness.

Required measurements by operation class:

- schema-valid plan rate
- complete seed disposition rate
- safe-action precision
- missed safe-opportunity rate
- claim-preservation rate
- false merge, archive, and delete rate
- specialist-routing accuracy
- repeated unchanged no-op rate
- cross-family oscillation rate
- retrieval-regression rate
- verified useful work per planner execution and premium request

Initial rollout rules:

- normalization and strongly evidenced link creation may graduate first
- rewrite, link removal, split, merge, and archive require shadow evaluation by class
- destructive false positives block rollout even when aggregate precision appears high
- delete remains disabled until the mutation-history/restore specification's gates pass

Numeric thresholds belong in versioned policy/configuration after baseline measurement. They should not be encoded only in prompts or prose.

## 14. Observability and Operator Review

Operators should be able to inspect:

- why a candidate was selected
- which family owned or received the work
- planner evidence and claim manifest
- policy acceptance and rejection reasons
- before/after diff through mutation history
- retrieval evidence before and after
- cooldown, stabilization, and escalation state
- provider disclosure decision
- restore availability

High-risk review queues should support approve, reject, edit, protect, restore, and route-to-specialist outcomes.

## 15. Testing Requirements

### Small tests

- type-specific policy rules
- claim-manifest completeness
- omission-reason validation
- family routing matrix
- cross-family stabilization and oscillation detection
- provider eligibility/redaction decisions
- coverage and cooldown math
- internal retrieval signals excluded from user-value features

### Medium tests

- global candidate queries cover cold and never-surfaced records outside the recent window
- planner reads do not mutate access/surfacing telemetry
- specialist routing creates one idempotent work item
- recent mutation suppresses another family until evidence changes
- content mutation queues version-aware search repair
- shared Postgres maintenance fails/defer safely rather than reading derivative cache data
- repeated producer defects generate one remediation signal

### Evaluation fixtures

- same vocabulary across unrelated projects
- temporal or source-qualified fact disagreements
- focused cold records that must be retained
- misleading high-surface/low-read records
- user-protected memories
- sensitive records requiring a local provider
- cross-family proposals for the same target

## 16. Acceptance Criteria

- Every enabled action class has explicit semantic preservation and ownership rules.
- Whole-corpus strategies are backed by bounded backend queries, not only a recent candidate window.
- Planner investigation does not modify user-facing retrieval signals.
- Cross-family churn is suppressed and observable.
- Provider data eligibility is decided before content disclosure.
- Retrieval evaluation can detect regressions independently of mutation receipts.
- Repeated write-stream defects are attributed and routed toward source prevention.
- High-risk or ambiguous facts are preserved and reviewed rather than silently collapsed.

## 17. Non-Goals

- Replacing the deterministic curation harness or work-item architecture.
- Making user retrieval analytics a direct archive/delete score.
- Defining a complete temporal truth model in this document.
- Allowing one universal planner prompt to own every maintenance family.
- Requiring external providers for protected or sensitive memories.
- Treating all old, cold, or cross-workspace records as low quality.

## 18. Open Decisions

1. The first numeric rollout gates for each operation class.
2. The minimal claim-manifest size that remains useful without inflating planner context.
3. The first-class representation of sensitive/provider-eligibility policy.
4. The coverage horizon and strategy mix used for operator reporting.
5. Whether high-risk review uses existing work items or a dedicated review-state table.
