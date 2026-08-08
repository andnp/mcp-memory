# Curator quality rollout

**Status:** Operator runbook; thresholds marked **Proposed** are not enforced by current code.

This runbook stages direct-MCP curator mutation behind measurable quality evidence. The semantic policy is defined in [Curation Quality and Maintenance-Family Ownership](specs/19-curation-quality-and-family-ownership.md); this document defines the operational gate.

## Current contract

The production curator is direct and agentic: it selects bounded seeds, uses internal MCP tools, validates the sanitized tool ledger, and reports actual tool calls and mutations. Provider prose is not evidence of work. Mutation tools and history/receipts establish execution and store state; they do not establish useful quality.

For every direct mutation, the run must reconcile:

- the direct mutation evidence and before/after revision tokens;
- the mutation-history event and receipt status;
- post-mutation quality evidence; and
- run, provider, and dashboard aggregates.

Quality evidence must be durable. Each mutation has exactly one quality disposition:

| Disposition | Operator meaning | Productive credit |
| --- | --- | --- |
| `productive` | Independent retrieval, content, or durability evidence passed policy. | Yes |
| `structural_only` | A structural/link repair passed its structural contract; no retrieval or content gain is claimed. | Only if the versioned policy explicitly permits it. |
| `neutral` | Quality was observed with no material change. | No. |
| `regressed` | Evidence shows material loss of utility or durable meaning. | No; stop or escalate. |
| `verified_only` | Store postcondition passed, but useful quality was not demonstrated. | No. |
| `unverified` | Store or consistency verification failed. | No. |
| `unobserved` | Quality was not measured; retain a specific reason such as unavailable repository, unavailable search, incomplete replay, no trusted query, or budget skip. | No. |

`verified_only` receipts are not productive quality. Missing evidence must be rolled up as `unobserved`, never as neutral or productive. A zero verification-failure count does not offset provider failures, invalid plans/ledgers, retries, missing quality evidence, or retrieval regressions.

## Rollout stages

1. **Shadow.** Run selection and quality evaluation without enabling unattended mutation. Confirm direct tool-ledger validation, mutation evidence, quality persistence, and provider attribution.
2. **Manual canary.** Enable a bounded operator-invoked run. Review every mutation, evidence disposition, history event, and reconciliation result before another canary.
3. **Small unattended canary.** Enable only a tightly bounded schedule, seed count, and mutation budget. Keep destructive operations disabled unless separately approved.
4. **Expanded rollout.** Increase one bound at a time after the canary window reconciles. Keep the previous bound available for immediate rollback.

The current direct path may report no sampled quality evidence or use a non-durable quality repository. Treat that as a rollout failure, not as a successful no-op. Do not advance until the path persists quality evidence or a durable `unobserved` reason.

## Monitoring and gates

Report counts and rates separately for:

- verified receipts and `verified_only` outcomes;
- productive outcomes;
- structural-only outcomes;
- neutral outcomes;
- regressions;
- unverified outcomes; and
- unobserved outcomes by reason.

Also report mutation-history events, direct-evidence rows, ledger validity, provider calls/tokens, retries, budget skips, no-ops, and reconciliation mismatches. Use `productive` quality evidence—not receipt count—as the numerator for productive yield. Keep neutral separate from regression; unchanged quality is not regression.

### Canary gates

The following are **Proposed operator thresholds**, not current code-enforced policy. Replace them with versioned configuration after a baseline is recorded:

- 100% of applied direct mutations have one durable quality disposition and a matching history/receipt identity.
- 0 evidence-reconciliation mismatches, missing before/after tokens, ledger-invalid runs, or unexplained provider-attribution gaps.
- 0 destructive false positives. Any confirmed loss of durable meaning is an immediate stop.
- Regression rate no higher than the pre-canary baseline and no more than 5% of evaluated mutations; any statistically material regression spike stops the canary.
- At least 80% of evaluated non-structural mutations are `productive` or policy-approved `structural_only`; `verified_only` does not count.
- `unobserved` remains below 10% of applied mutations, with no repeated infrastructure reason. This is a visibility gate, not permission to count unobserved work as neutral.
- Provider failures, retries, token usage, and mutation counts reconcile across task telemetry and dashboard aggregates.

Do not advance on aggregate rates alone. Inspect the individual evidence for every regression, every structural-only result, every unobserved reason, and a sample of productive results.

## Stop conditions

Stop new mutation admission and preserve the evidence when any of these occurs:

- a mutation lacks durable quality evidence or an explicit reason for being unobserved;
- direct evidence, history, receipts, ledger, or dashboard totals disagree;
- a receipt is treated as productive quality;
- a ledger is invalid, before/after identity is missing, or current state no longer matches the recorded postcondition;
- any confirmed destructive false positive, provenance loss, protection violation, or meaningful-date/qualifier loss occurs;
- regression exceeds the approved threshold or repeats on the same operation/family;
- provider attribution, budget accounting, or token accounting cannot be reconciled;
- the quality repository, search path, or history backend is unavailable;
- unattended mutation would continue while rollback availability is unknown.

Stopping means disable admission; it does not delete receipts, quality evidence, mutation history, provider records, or raw ledgers.

## Evidence reconciliation checklist

For each run, and again before promotion:

1. Join direct mutation evidence to the task and execution epoch.
2. Join each applied mutation to exactly one history event and receipt.
3. Check affected IDs, operation, intent, before token, and after token.
4. Require exactly one quality disposition per evaluated mutation, or one durable `unobserved` reason.
5. Recompute productive yield from quality outcomes; exclude receipts and `verified_only`.
6. Compare direct counts and provider usage with task and dashboard aggregates.
7. Review all mismatches before continuing; do not repair rollups by deleting or rewriting evidence.

## Live manual verification

For a bounded operator-invoked canary:

1. Confirm the daemon is running and `admin health` has no new transport or persistence errors.
2. Trigger exactly one forced `memory-curator` task; do not start a second run while the first is pending or running.
3. Inspect `admin task show <task-id>` after completion. Reconcile task status, run result, direct-evidence rows, quality-evidence status, and mutation counts.
4. Inspect the daemon log for errors from the same time window. A completed task is not a clean pass when the transport logged a mutation-evidence persistence failure.
5. Treat `quality_evidence_status=not_applicable` as missing quality proof for a quality canary. Structural-only reviews still require their structural contract and must not be counted as productive retrieval quality.
6. Stop further mutation admission when any task result, evidence store, or daemon-log signal disagrees; preserve the run ID, task ID, execution epoch, and log window for reconciliation.

For rollback actions and evidence preservation, see [Curator quality rollback](curator-quality-rollback.md).
