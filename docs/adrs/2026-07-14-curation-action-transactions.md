## ADR: Transaction-scoped curation action execution

- **Status:** Accepted
- **Date:** 2026-07-14

## Context

CUR-034 and CUR-035 need one backend-neutral contract for applying one
validated curation action. The existing repositories do not provide that
boundary:

- SQLite `RelationalMemoryRepository` uses the thread connection from
  `DatabaseManager.get_connection()` and commits public operations through
  connection context managers.
- Postgres `PostgresRelationalMemoryRepository` opens a pooled connection from
  `SessionManager.open_connection()` and explicitly commits each public
  operation.
- `SessionManager` exposes a connection lease, but no transaction-scoped
  repository or callback API.

Calling those public methods in sequence cannot make a compound action,
lexical projection, reversible history, repair intent, and receipt atomic.
The canonical identity ADR at
`docs/adrs/2026-07-14-curation-canonical-identities.md` supplies the record and
graph tokens used for the precondition checks here.

## Decision

Add a backend-neutral action-store seam. The exact implementation names may
follow module conventions, but the contract is:

```python
class CurationActionStore(Protocol):
    def execute_action(
        self,
        *,
        run_id: UUID,
        action_id: UUID,
        target_ids: Sequence[str],
        expected_tokens: Mapping[str, str],
        apply: Callable[[CurationTransaction], MutationResult],
    ) -> MutationReceipt: ...
```

`CurationTransaction` is the only repository surface available to `apply`.
It provides transaction-scoped reads and domain operations for memories,
tags, workspace associations, links, lineage, lexical projections, and
version-aware repair intent. It does not expose `commit`, `rollback`, a new
connection, or any public repository method that commits independently. The
callback returns the normalized operation result; the action store captures
the before-state before the callback and the after-state after it, so callers
cannot omit required history rows.

The action store owns the following sequence:

1. Validate the run/action identity and normalize, deduplicate, and sort all
   target IDs by their canonical UTF-8 byte sequence.
2. Check `(run_id, action_id)` idempotency before doing domain work. An
   existing receipt is returned unchanged, including `applied_unverified`,
   `verified`, `stale`, or `failed` status. A replay never reapplies a
   mutation.
3. Begin one backend transaction and lock every existing target in the sorted
   order. All existing link endpoints that the callback may change must be in
   `target_ids`; discovering a new existing target during the callback is a
   fatal contract violation, not an opportunity to acquire a lock out of
   order.
4. Re-read current records, relevant links, and `memory_protections` after
   locking. Compare them with the supplied canonical record/graph tokens and
   action preconditions. A stale action exits before any domain write.
5. Capture normalized before snapshots, run `apply` through the transaction
   facade, update the authoritative domain rows and lexical projection, and
   enqueue version-aware repair intent for each changed semantic memory.
6. Capture normalized after snapshots and write the audit rows and compact
   receipt before commit.
7. Commit once. Postcondition verification is a fresh read after this commit;
   it is not part of the action transaction.

For an applied action, these writes are one atomic unit:

- domain memory, tag, workspace, link, and lineage changes;
- the SQLite `memories_fts` or Postgres `memory_search_documents` lexical
  projection;
- a durable version-aware row in the existing `embedding_repair_queue` when
  semantic fields changed;
- one `memory_mutation_events` row;
- the related `memory_record_revisions` and `memory_link_revisions` rows; and
- one `curation_action_receipts` row with `status=applied_unverified`, linked
  to the already-created `curation_runs` row.

`memory_protections` is read and enforced in this transaction; an action never
changes protection state. Protection writers must acquire the same target
lock first so a concurrent protection change cannot race execution.

`curation_runs` is created and terminalized by the run/ledger API, not by the
action callback. The action store verifies that the referenced run exists and
is eligible for action execution, but does not hold a plan-wide transaction or
change the run's terminal state.

The action transaction is per action, not per plan. Earlier committed actions
remain auditable when a later action is stale or fails. A failure at any point
before commit rolls back every domain, projection, repair-intent, history,
revision, and receipt write. Embedding generation and postcommit verification
remain asynchronous; only durable repair intent and the initial receipt belong
inside this transaction.

### Canonical action errors

The repository exposes one base `CurationActionError` with exactly these
terminal categories:

- **`CurationActionStaleError`** — the target is missing, a record or graph
  token differs, a required status/link assertion no longer holds, or an
  expected protection/precondition token changed. It is deterministic and
  must not be retried blindly. The action transaction writes no domain,
  history, repair, or receipt row before raising. The executor may record a
  compact `stale` receipt through the curation ledger in a separate short
  transaction. A protection mode that currently denies an otherwise-current
  action is a fatal policy outcome, not a stale backend error.
- **`CurationActionTransientError`** — a retryable backend condition such as a
  SQLite busy/locked result, Postgres serialization failure, deadlock, lock
  timeout, or connection interruption. The transaction is rolled back before
  the error escapes. The caller retries the same `(run_id, action_id)` with a
  bounded policy; it does not retry an arbitrary callback with a new identity.
- **`CurationActionFatalError`** — malformed canonical input, unsupported
  operation, invalid domain data, violated transaction contract, unexpected
  constraint failure, or another non-retryable implementation/domain error.
  The transaction is rolled back and the original exception is preserved as
  the cause. Policy rejection is represented as a fatal action outcome, not
  as a transient backend error.

Backend drivers must translate only known retryable conditions into
`CurationActionTransientError`; broad exception-string matching is not a
valid implementation. Error payloads include the run/action IDs and bounded
target identifiers, never full memory content or provider transcripts.

### Idempotency and uniqueness

`curation_action_receipts` has a database-enforced, non-null unique constraint
on `(run_id, action_id)`. The pair is the durable idempotency key, not the
operation name, target order, provider request ID, or receipt timestamp.
`memory_mutation_events` records the same logical run/action reference and may
have at most one applied event for that pair; its event is written in the
same transaction as the receipt. A uniqueness race is resolved by rolling
back the losing transaction, reading the committed receipt, and returning it
unchanged. A duplicate with a different canonical action payload is a fatal
identity collision, not a second execution.

Stale or failed outcome receipts recorded by the executor after a rolled-back
action transaction use the same pair and the same uniqueness rule. Conditional
terminalization follows the existing first-terminal-write-wins rule: a later
reconciler or provider callback cannot replace a terminal receipt.

## Backend transaction patterns

These are the required shapes, not new public repository methods.

### SQLite

`DatabaseManager.open_connection()` supplies a dedicated connection for the
action. The implementation takes the writer lock before reading preconditions,
then uses the same connection for every operation:

```python
connection = db_manager.open_connection()
try:
    connection.execute("BEGIN IMMEDIATE")
    for memory_id in sorted_target_ids:
        connection.execute(
            "SELECT id FROM memories WHERE id = ?", (memory_id,)
        )
            # Re-read tokens, apply through CurationTransaction, write the
            # mutation/history/receipt entities plus projection/repair rows,
            # then:
    connection.commit()
except Exception:
    connection.rollback()
    raise
finally:
    connection.close()
```

SQLite's single-writer lock makes row-level `FOR UPDATE` unnecessary. The
sorted reads still matter: they keep the contract deterministic and ensure
that any future SQLite concurrency refinement has the same lock discipline.
The implementation must not call `RelationalMemoryRepository.update_memory`,
`add_link`, or another auto-committing public method from the callback.

### Postgres

`SessionManager.open_connection()` supplies one pooled connection with
`autocommit=False`. Existing target rows are locked one at a time in canonical
sorted order; a database collation-dependent `ORDER BY id` is insufficient to
define this order:

```python
with sessions.open_connection() as connection:
    try:
        with connection.cursor() as cursor:
            for memory_id in sorted_target_ids:
                cursor.execute(
                    "SELECT id FROM memories WHERE id = %s FOR UPDATE",
                    (memory_id,),
                )
            # Re-read tokens/protections, apply through CurationTransaction,
            # write history/revisions/repair/receipt, then:
        connection.commit()
    except Exception:
        connection.rollback()
        raise
```

The same sorted target lock order is mandatory for link, protection, and
future compound-action writers. `FOR UPDATE` is taken before conditional
revision checks; conditional `UPDATE ... WHERE id = ... AND token = ...`
remains a final stale-write guard and must affect exactly one row. A zero-row
update is `CurationActionStaleError`, not success. The implementation must not
call `PostgresRelationalMemoryRepository.update_memory`, `add_link`, or
another method that opens/commits its own connection from the callback.

## Consequences

- CUR-034 and CUR-035 can implement one behavior contract while retaining
  backend-specific SQL and transaction mechanics.
- Stale plans fail closed without partial domain changes; transient failures
  can retry safely; fatal defects remain visible instead of being disguised as
  retries.
- History, lexical search state, repair intent, and receipts cannot drift from
  a committed mutation. A crash after commit but before verification leaves an
  `applied_unverified` receipt for reconciliation.
- SQLite remains the local authoritative store and Postgres remains the shared
  authoritative store. Derivative caches and embedding generation are not
  alternative transaction participants.
- The existing auto-committing public repository methods remain valid for
  current callers, but action execution must use a new internal transaction
  facade until those methods can be safely refactored.

## Required contract coverage

Shared SQLite/Postgres tests must prove:

- reversed concurrent target input still acquires locks in one stable order;
- stale record, graph, missing-target, and protection preconditions write no
  domain/history/repair/receipt state;
- injected failures after domain mutation, projection update, repair intent,
  history, and receipt preparation roll back the entire action;
- `(run_id, action_id)` replay returns the original receipt without duplicate
  memories, links, events, revisions, or repair units;
- concurrent duplicate execution produces one receipt and one applied event;
- semantic mutations update lexical projections and enqueue one
  version-aware repair unit; and
- SQLite and Postgres expose the same `CurationActionError` categories and
  terminal behavior.
