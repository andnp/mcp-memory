# Curator quality rollback

**Status:** Operator runbook; restore actions remain subject to current protection and revision checks.

Use this procedure when the curator quality rollout breaches a stop condition in [Curator quality rollout](curator-quality-rollout.md) or the semantic requirements in [Curation Quality and Maintenance-Family Ownership](specs/19-curation-quality-and-family-ownership.md).

## Immediate containment

1. Disable the direct curator scheduler, task admission, or deployment feature gate. If there is no dedicated switch, stop the trigger that creates new curator executions. Do not broaden the change to other maintenance families.
2. Let an in-flight session reach its bounded end when safe; otherwise terminate it and record the task, execution epoch, and reason. Do not start replacement runs.
3. Disable unattended destructive operations first. Keep read-only inspection and evidence export available.
4. Record the stop time, rollout stage, code/policy version, affected runs, and the exact stop condition.
5. Freeze the evidence window for investigation: raw tool ledgers, direct mutation evidence, receipts, quality evidence, mutation history, provider usage, and dashboard snapshots.

If direct mutation evidence cannot be persisted, treat the run as unreconciled even if the provider reports success. Keep curator mutation admission disabled until the storage error and the task-level error signal are both verified.

Containment disables future direct mutation. It does not delete or rewrite evidence and does not silently restore records.

## Determine scope

Partition applied actions into:

- **No store change:** rejected, skipped, ledger-invalid, or unverified actions with no committed mutation. Preserve their failure evidence; no restore is required.
- **Verified store change with no productive proof:** `verified_only`, `neutral`, or `unobserved`. Do not call these regressions automatically. Review quality and retrieval impact before deciding whether to restore.
- **Structural-only:** review the structural contract and link/lineage integrity. Restore only when the repair is wrong or harmful.
- **Regressed:** identify the exact mutation event, affected IDs, before/after revisions, and evidence showing lost utility or durable meaning.
- **Unreconciled:** quarantine the run from quality rollups until direct evidence, history, receipts, and aggregates are compared. Do not infer outcomes from counts.

Receipts are execution evidence. They are not a reason by themselves to restore, and they never justify deleting quality or mutation history.

## Back out harmful mutations

Use the authoritative mutation-history/restore path, with operator approval for high-risk merge, split, archive, or destructive changes:

1. Select the affected mutation event and inspect its bounded before/after snapshots, links, protection state, and dependent later events.
2. Confirm the current revision tokens still match the event after-state. If state diverged, stop and route the event for manual review; do not force an overwrite.
3. Restore the complete event where supported. A restore is a new compensating mutation event that references the original event; it does not edit history.
4. Preserve later user changes and retrieval telemetry. Restore semantic state only within the approved operation scope.
5. Re-run post-restore verification and quality evaluation. Persist the result, including `neutral`, `regressed`, `unverified`, or `unobserved` with reason.
6. If automatic compensation is not deterministic, protected, or precondition-safe, leave the original event applied, block further curator mutation, and create operator-review work.

Do not hard-delete affected memories to undo a rollout. Do not remove receipts, quality evidence, provider records, task ledgers, mutation events, or revision snapshots. History must show both the original mutation and the compensating restore.

## Reconciliation after rollback

Before re-enabling any curator mutation:

- every applied event in the incident window has a matching history/receipt identity and revision evidence;
- every mutation has durable quality evidence or an explicit durable `unobserved` reason;
- productive counts are recomputed from `productive` and policy-approved `structural_only` outcomes only;
- verified-only, neutral, regressed, unverified, and unobserved outcomes remain distinct;
- restore events and current revision tokens reconcile without overwriting later changes;
- provider calls, failures, retries, token usage, task results, and dashboard aggregates agree;
- the triggering defect is fixed or the affected operation/family remains disabled; and
- a new manual canary passes the rollout gates.

The requirement for complete evidence and reconciliation is the rollout contract. Any numeric re-enable threshold is **Proposed** unless present in versioned runtime configuration; do not treat the example thresholds in the rollout runbook as code-enforced.

## Re-enable decision

The operator records one of:

- **Remain disabled:** evidence is incomplete, contradictory, or shows unresolved quality harm.
- **Manual-only:** reads and operator-reviewed mutations may continue; unattended admission remains disabled.
- **Resume canary:** only after the defect is addressed, evidence is durable, and the next canary uses reduced bounds and the approved rollback path.

Retain the incident package and all evidence according to the existing operational retention policy. A rollback is complete only when future mutation is contained, harmful changes are either safely restored or explicitly queued for review, and the evidence trail remains intact.
