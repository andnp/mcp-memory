# Curator Productivity and Feedback Loop

**Status:** Draft design

**Date:** 2026-08-18

This document proposes the next curation improvements. It is not the
canonical description of current runtime behavior. The active runtime
contract remains [Direct Agentic Curation](10-curation-harness-and-typed-planner.md),
while the accepted identity and transaction contracts remain authoritative:

- [Versioned canonical identities for curation](../adrs/2026-07-14-curation-canonical-identities.md)
- [Transaction-scoped curation action execution](../adrs/2026-07-14-curation-action-transactions.md)
- [Curation Quality and Maintenance-Family Ownership](../specs/19-curation-quality-and-family-ownership.md)
- [Curator quality rollout](../curator-quality-rollout.md)

## 1. Executive summary

The curator is active, but current telemetry cannot reliably distinguish
useful memorybase improvement from expensive, weakly reconciled activity.
The immediate design priority is to close the causal loop:

```text
candidate -> bounded packet -> provider attempt -> action -> durable receipt
         -> mutation history -> advisory quality observation -> future selection
```

The design has five parts:

1. make every candidate and action traceable across the existing task, run,
   identity, receipt, history, and quality contracts;
2. reduce context and provider cost with summary-first, family-specific,
   token-bounded packets and on-demand hydration;
3. make candidate dispositions durable and damp repeated unresolved work;
4. measure retrieval and content effects asynchronously after a committed
   mutation; and
5. use those observations to adjust future scope and cost without ever
   rejecting the current curator run for quality.

The first implementation tranche should establish canonical evidence closure;
bounded packets and ambition control follow only after that signal is complete.

## 2. Evidence and diagnosis

The 2026-08-18 post-restart audit provides the baseline for this design:

| Signal | Observation | Interpretation |
| --- | ---: | --- |
| Curator runs | 31 total, 26 completed, 1 failed, 4 retries | The campaign is active, with some retry pressure. |
| Reported mutations | 59 | Activity is not proof of useful work. |
| Valid tool ledgers (legacy: valid plans) | 22; rate 0.88 | Tool-call ledger validity is measurable, but not quality. |
| Productive mutations | 0 | The current aggregate cannot prove durable useful work. |
| Verified receipts | 0 | Direct curator activity is not joined to receipt telemetry in this window. |
| Mutation-history events | 0 | The direct mutation path is not visibly reconciled to history. |
| Candidates | 1,067 total; 496 escalated; 370 pending; 34 actioned; 167 cooldown | The queue contains a large unresolved/escalated population. |
| Provider calls | 62; p95 latency about 364 seconds | Provider work is expensive and slow. |
| Recorded tokens | About 60.8M total; 60.2M input | Context size is a likely cost and latency bottleneck, subject to token-source verification. |

The latest post-restart run completed with `curation_outcome=applied_verified`
and two reported mutations, while direct receipts and history were still not
visible. The rolling dashboard's one `quality_rejected` outcome predates the
policy change and must not be used as evidence against the new invariant.

These observations establish an observability problem. They do not establish
that all 59 mutations were lost, nor that all provider input was curator
packet content. The implementation must identify the missing joins before
introducing quality-based controls.

### 2.1 Root problems

#### Execution and evidence are disconnected

The direct agentic path reports tool activity and mutations, but the live
aggregate has no corresponding direct receipt or mutation-history evidence.
Possible causes include a bypass around the canonical action store, lost
execution identity across an async boundary, or an aggregate query that does
not include the direct path. The design treats all three as reconciliation
failures until verified.

#### Work packets are too expensive to evaluate

The input-token and latency profile is inconsistent with a small bounded
frontier. The likely failure mode is repeated neighborhood or support-record
context in provider prompts. This is a hypothesis to verify with packet-level
usage telemetry, not a reason to assume that every token was avoidable.

#### Candidate dispositions do not close the queue

The large `escalated` and `pending` populations suggest that ambiguous or
unchanged candidates may be reconsidered without a durable, reasoned backoff.
Repeated evaluation consumes provider budget while producing no new evidence.

#### Quality is conflated with execution

A receipt proves that a store mutation committed. It does not prove that
retrieval improved, that content became clearer, or that a structural change
was useful. Conversely, missing quality evidence is an observability gap, not
permission to reject the curator run.

## 3. Goals and non-goals

### Goals

- Provide an unbroken, queryable identity chain from candidate selection to
  later quality observation.
- Make committed curator mutations visible through the canonical receipt,
  history, revision, and repair-intent contracts.
- Increase useful work per provider call and per recorded token.
- Prefer high-confidence, family-owned work and avoid repeated unresolved
  candidates.
- Measure quality from replayable retrieval/content evidence.
- Adjust future admission, packet size, and family mix using advisory signals.
- Preserve the invariant that a curator run is never quality-rejected.

### Non-goals

- Adding another independent curator campaign or specialist scheduler.
- Replacing the accepted transaction or canonical-identity ADRs.
- Treating mutation count, provider prose, tool-call count, or receipts as
  quality evidence.
- Introducing cross-workspace automatic merges.
- Building a multi-agent debate or consensus stage.
- Rebuilding the entire search index inside a curator session.
- Making destructive delete behavior automatic.

## 4. Design principles

1. **Execution, quality, and control are separate contracts.** The harness
   proves what happened; quality evaluation measures what changed; the
   controller chooses what to attempt later.
2. **The current run is not quality-gated.** A quality regression, neutral
   result, or missing observation becomes durable advisory evidence. It may
   narrow future work, but it cannot terminalize the current curator run as
   `quality_rejected`.
3. **Evidence beats narrative.** Provider text and claimed actions are
   diagnostic context only. Durable rows, receipts, hashes, ledgers, and
   replay results are authoritative.
4. **Bound context before adding intelligence.** Smaller, family-specific
   packets and deterministic filters should precede more agents, judges, or
   elaborate planning.
5. **Unknown remains unknown.** `unobserved` must carry a reason and receive
   no productive credit. It must not be converted into neutral, regression,
   or success by aggregation.
6. **One canonical path.** Direct curator mutation tools use the canonical
   transaction-scoped action store. A parallel shadow writer or alternate
   receipt ledger is not part of this design.

## 5. Target architecture

### 5.1 Identity chain

The implementation should reuse existing identity boundaries and add only
missing links:

| Stage | Required identity/evidence | Owner |
| --- | --- | --- |
| Candidate discovery | candidate ID, target revision/context token, selection strategy, family, reason | selector/work-item layer |
| Packet assembly | packet ID or durable packet manifest, ordered candidate IDs, family, size/token budget, disclosure decision | curator handler |
| Provider attempt | task ID, execution epoch, provider request/attempt ID, usage fields, packet ID | provider/task telemetry |
| Action | deterministic action ID, operation, target IDs, expected tokens, evidence references | curator tool/action store |
| Commit | `curation_action_receipt`, before/after tokens, mutation event, revisions, repair intent | transaction-scoped action store |
| Run result | actual tool ledger, action counts, candidate dispositions, run outcome | curator handler/reporting |
| Quality observation | receipt/action ID, evaluator version, replay snapshot, disposition, reason/evidence completeness | quality evaluator |
| Retrieval feedback | replay query ID or trusted user-query reference, result/rank/read observations | search analytics |

The action ID and receipt identity are the action-level join. Provider attempt
identity remains the run-level join and must not be used as a substitute for
an action ID.

The task handler must create the canonical `curation_runs` row before opening
the agent session, then transition it through the existing run state contract
before any mutation tool can execute. The current direct path creates a
quality run lazily after the session, which is too late for a receipt with a
foreign key to `curation_runs`. The run identity must come from the existing
run/ledger API; this design does not prescribe a new UUID namespace.

### 5.2 Packet construction

The packet builder should produce a bounded manifest containing, at minimum:

- candidate and memory references;
- title and summary, with bounded excerpts only when necessary;
- candidate score, family, selection reason, and revision/context token;
- relevant edge/type summaries rather than entire neighborhoods;
- protection and disclosure decisions;
- packet ID, policy version, and configured token/record limits.

The provider starts with this compact representation. It hydrates full content
only for a specific candidate/action under investigation. Hydration remains
bounded and is attributed to the same packet and provider attempt.

Packets are homogeneous by maintenance family where practical. A merge or
split packet should not also ask the provider to solve unrelated taxonomy,
conflict, and orphan problems. One packet may contain several candidates, but
each proposed action must retain its own identity and evidence.

The initial limits should be configuration, not hard-coded policy. Begin with
a small experiment over candidate counts such as 4, 8, and 12, then choose a
limit from observed useful-work-per-token and latency. No limit is accepted
as a quality threshold merely because it is convenient.

### 5.3 Canonical mutation execution

Every direct mutation tool must route through the transaction-scoped action
store defined by the accepted action-transaction ADR. One action transaction
must atomically cover the domain change, lexical projection, version-aware
repair intent, mutation history/revisions, and initial receipt.

The adapter must preserve the direct agentic contract: the agent can
investigate and act through internal MCP tools, while the harness validates
the sanitized ledger independently. The adapter must not add a second plan
approval stage or infer success from provider prose.

Before execution, the adapter validates the action payload and derives the
canonical action ID according to the identity ADR. It supplies the expected
record/graph tokens required by the action store. Stale, transient, and
fatal action errors are returned to the agent as structured tool results where
the session can safely continue; an action error must not implicitly become a
quality rejection of the run. The exact action-ID derivation is owned by the
accepted identity contract, not by a provider call sequence or tool-call
counter.

After commit, a fresh verification may move an `applied_unverified` receipt to
`verified`. A crash or verification failure remains visible as evidence; it
does not silently become a successful quality result.

### 5.4 Candidate lifecycle

Use the existing `curation_candidate_state` disposition vocabulary. Each
terminal or deferred disposition must include a reason and the target
revision/context token that was evaluated. Packet membership is a lease or
attribution detail, not a new candidate disposition:

```text
pending
  -> claimed/leased -> actioned
                    -> retained
                    -> cooldown
                    -> escalated
```

`actioned` means that an action was durably committed, not that it improved
quality. `retained` is a valid completed evaluation with no mutation.
`cooldown` means the candidate is temporarily suppressed because the same
revision and evidence did not produce useful work. `escalated` means a
specific unresolved reason requires another family or operator review; it is
not a generic retry bucket.

Candidates included in a packet but neither examined nor mutated must not be
marked `retained`, `cooldown`, or `escalated`. On normal completion,
cancellation, timeout, or provider failure, their packet/work-item lease must
be released and their candidate state must remain eligible for future
selection. This prevents partial packets from silently starving candidates.

Cooldown is primarily time-bounded through `cooldown_until`, with an early
reset when materially new evidence arrives: a human edit, a new source record,
a relationship/revision change, or a new trusted retrieval signal. A retry
with identical context does not reset it. Backoff duration is family-specific
configuration; thresholds must be measured before enforcement.

### 5.5 Advisory quality evaluation

Quality evaluation runs asynchronously after a committed action. It consumes
the before/after identity and a bounded replay context. It must persist:

- the action/receipt identity;
- evaluator and policy versions;
- evidence completeness and unobserved reason, if any;
- operation/family-specific outcome;
- before/after retrieval and content measurements;
- whether the result is `productive`, `structural_only`, `neutral`,
  `regressed`, `verified_only`, `unverified`, or `unobserved`.

The productive numerator follows the existing quality contract. Receipts and
`verified_only` outcomes do not receive productive credit. Structural-only
credit is allowed only where the versioned policy defines a structural
contract; it must not be presented as retrieval improvement.

### 5.6 Feedback controller

The controller changes future selection and budget, never the result of the
run that produced the signal:

| Evidence state | Future action |
| --- | --- |
| Strong productive evidence and complete receipts | Increase one bound modestly: packet size, candidate breadth, or family share. |
| Neutral evidence | Keep scope stable; prefer different candidates or wait for new evidence. |
| Repeated regression for one family/cluster | Narrow that family or cool that cluster; retain the evidence for review. |
| High unobserved/incomplete-replay rate | Reduce ambition until observability is repaired; do not count unknown as bad quality. |
| Provider latency/token pressure | Reduce packet size, remove redundant context, or defer low-value families. |
| Receipt/history reconciliation mismatch | Stop new mutation admission for the affected path and preserve evidence; do not delete or rewrite telemetry. |

The controller must be deterministic, versioned, and explainable. A future
policy may use proposed thresholds, but thresholds are not active until a
baseline window and evidence-completeness report exist.

### 5.7 Run-outcome invariant

`CurationRunOutcome.QUALITY_REJECTED` remains readable for historical records,
but it is not an allowed outcome for a new direct `memory-curator` run. The
direct path may report a no-op, partial application, provider failure,
cancellation, or execution/verification failure when those events actually
occur. A quality disposition of `regressed`, `neutral`, `unobserved`, or
`verified_only` is recorded on the action/evidence path and cannot write a
quality-rejected run outcome. Future admission control may defer or narrow
later candidates; it must not rewrite the completed run's outcome.

## 6. Measurement model

### 6.1 Funnel metrics

Report the following separately for `memory-curator` and curation-wide data:

1. candidates selected, leased, evaluated, actioned, retained, escalated, and
   cooled;
2. provider attempts, calls, failures, retries, latency, and token fields;
3. valid ledgers and actual mutation tool calls;
4. committed actions, receipt status, history events, revisions, and repair
   intent;
5. quality dispositions and unobserved reasons; and
6. replayed retrieval/content outcomes and later user/agent read signals.

The dashboard must not combine curation-wide quality with curator-only yield
without naming the population. It must distinguish rolling historical counts
from the newest post-change run.

### 6.2 Productivity metrics

Execution integrity metrics:

- receipt closure rate;
- action-to-receipt reconciliation rate;
- ledger validity rate; and
- provider-attribution completeness.

Productivity metrics:

- productive quality rate among evaluated actions;
- useful-work count per completed run;
- useful-work per provider call;
- useful-work per recorded token, where token fields are authoritative; and
- median and p95 provider latency per packet family.

Safety and churn metrics:

- evidence-reconciliation mismatches;
- unobserved rate by reason;
- stale-precondition and transient retry rates;
- repeated A-to-B-to-A changes;
- mutations to memories with no later retrieval/read evidence; and
- protected or provenance-changing action incidents.

Mutation count remains a diagnostic count, not the productivity numerator.

### 6.3 Retrieval and content evaluation

Use two complementary sources:

1. **Offline replay.** Re-run a fixed, versioned sample of real historical
   search queries against before/after snapshots. Include both queries that
   previously surfaced or read the affected memories and a stable global
   control sample. Measure result-set overlap, rank changes, zero-result
   changes, and payload size. Add labeled relevance metrics only when the
   corpus has a trusted relevance label or human review. A memory without
   suitable historical queries is `unobserved` with a reason such as
   `no_historical_queries`, not an invented synthetic success.
2. **Online observation.** Observe subsequent search-to-read/citation behavior
   for affected memories, with a sufficient wait window. Maintenance reads must
   not themselves inflate access or surfacing signals.

For structural links, evaluate endpoint/type/lineage invariants separately.
Do not claim retrieval improvement from a structural-only result.

## 7. Implementation sequence

### Phase 0: Reconcile the current path

- Trace direct curator mutation tools to their storage implementation.
- Join task/run/attempt, ledger, direct mutation evidence, receipt, history,
  revision, repair, and quality rows for one bounded run.
- Identify whether the current gap is write bypass, identity propagation, or
  dashboard query scope.
- Add reconciliation diagnostics before adding controls.

**Exit condition:** one run can be explained from selection through durable
mutation evidence, or the exact missing boundary is reported explicitly.

### Phase 1: Close canonical mutation evidence

- Create the canonical `curation_runs` row before the direct agent session.
- Route all direct mutation tools through the accepted action-store seam.
- Derive action identity from the validated canonical action payload and
  enforce idempotency and canonical target/precondition tokens.
- Verify receipt/history/revision/repair joins on SQLite and Postgres.
- Preserve the direct agentic ledger contract and run-level outcomes.

**Exit condition:** every committed direct mutation has exactly one durable
receipt and matching history identity, or an explicit persisted failure.

### Phase 2: Bound and attribute work packets

- Persist a compact, bounded packet manifest in the existing run/attempt
  telemetry owner. Do not create a competing packet ledger; add a versioned
  run field or an existing direct-evidence extension only after verifying
  whether one run can contain multiple packets.
- Add per-packet candidate count, family, context size, provider usage, and
  hydration counts.
- Run the packet-size experiment only after Phase 1 provides authoritative
  receipt and quality joins.
- Remove redundant neighborhood expansion and keep full hydration on demand.

**Exit condition:** provider cost and latency can be attributed to a packet,
family, and candidate set, and useful-work metrics have authoritative action
evidence.

### Phase 3: Improve candidate queue hygiene

- Persist disposition reasons and evaluated revision/context tokens.
- Add family-specific cooldown/backoff for unchanged or repeatedly escalated
  candidates.
- Prioritize deterministic high-confidence candidates and cold/never-surfaced
  candidates only when their operation has a clear contract.
- Keep specialist routing inside the canonical `memory-curator` campaign.

**Exit condition:** the queue shows why candidates remain pending/escalated and
identical candidates do not consume repeated provider calls without new
evidence.

### Phase 4: Add asynchronous quality feedback

- Persist replay snapshots and evaluator versions per action.
- Add offline retrieval replay and structural/content-specific evaluations.
- Add online read/citation observations with a delayed attribution window.
- Expose complete dispositions and unobserved reasons in the dashboard.

**Exit condition:** productive credit is derived only from measured evidence;
unknown evidence is visible and receives no credit.

### Phase 5: Introduce conservative ambition control

- Begin with operator-visible recommendations, not automatic changes.
- Enable one bound at a time: packet size, family share, or candidate breadth.
- Use versioned thresholds and compare against a fixed baseline.
- Narrow future work on regression or cost pressure; never quality-reject a
  completed run.

**Exit condition:** each controller decision is explainable from durable
metrics and can be reversed by changing future admission configuration.

## 8. Verification strategy

The implementation should add behavior-level coverage for:

- packet bounds, family separation, and on-demand hydration;
- stable packet/action identity across retry and restart;
- atomic mutation plus receipt/history/revision/repair writes;
- idempotent replay and stale-precondition behavior;
- candidate disposition reasons and cooldown reset on new evidence;
- complete versus unobserved quality evaluation;
- productive credit excluding receipts and `verified_only`;
- quality signals changing future scope without changing the current run
  outcome;
- historical versus newest-run dashboard projections; and
- SQLite/Postgres parity for the canonical action contract.

The existing focused suites are the initial homes for this coverage:
`tests/small/test_curation_action_store.py`,
`tests/small/test_curation_direct_mcp.py`,
`tests/small/test_curator_evidence.py`,
`tests/small/test_curator_cooldown.py`,
`tests/small/test_curation_quality.py`,
`tests/medium/test_curator_direct_mcp.py`,
`tests/medium/test_curator_verified_execution.py`, and
`tests/medium/test_postgres_curation_action_transaction.py`. New tests should
extend these boundaries rather than create parallel harnesses.

Runtime verification should use one bounded curator run and reconcile the
management API, task result, logs, direct evidence, receipts, history,
provider usage, and quality rows together. Health alone is insufficient.

The operational canary should trigger exactly one run, inspect its task result,
then query health, overview, nerd metrics, mutation history, direct evidence,
and quality evidence using the same task ID and execution epoch. A dashboard
with zero receipts is a failed evidence closure check, even if the task says
it completed.

## 9. Risks and deferred decisions

### Risks

- A packet-size limit can lower recall if candidate selection is weak; measure
  useful work, not just latency.
- Online retrieval attribution can be noisy because user queries vary and
  observations arrive after the mutation.
- A controller can create selection bias by repeatedly favoring easy work;
  retain bounded coverage and report untouched candidate families.
- Missing token fields can make cost controls unsafe; add provider usage
  normalization before enforcing token budgets.
- A dashboard can appear healthy while direct mutation evidence is absent;
  reconciliation mismatches must be prominent.

### Defer until the evidence loop works

- multi-agent debate or judge panels;
- cross-workspace automatic merges;
- whole-corpus online re-indexing during curator execution;
- automatic destructive deletion; and
- hard quality thresholds that change current run outcomes.

### Open decisions

1. Use `curation_runs` plus the existing run/attempt/direct-evidence telemetry
   as the canonical packet attribution owner. Add a bounded versioned packet
   field or extension only if one run can contain multiple packets; do not add
   a second receipt or event ledger.
2. Which provider usage fields are authoritative for each configured adapter?
3. Which historical queries have enough trust and volume for replay?
4. Which candidate families have deterministic structural contracts suitable
   for structural-only credit?
5. What baseline window is long enough to set ambition thresholds without
   baking current telemetry gaps into policy?

## 10. Acceptance criteria for this design

This design is ready to implement when:

- the direct mutation evidence gap has a named owner and a reproducible
  diagnosis;
- the action identity and transaction path point to the accepted ADRs rather
  than a competing ledger;
- packet cost can be attributed without relying on provider prose;
- candidate dispositions explain unresolved queue stock;
- quality outcomes explicitly separate execution proof from usefulness; and
- the controller contract states that quality can change future scope but can
  never produce `curation_outcome=quality_rejected` for the current run.
