## ADR: Versioned canonical identities for curation

- **Status:** Accepted
- **Date:** 2026-07-14

## Context

The curation harness, planner validator, mutation history, executor, verifier,
restore path, and both storage backends need to agree on whether two states are
the same. The existing shared read-cache validation token is useful evidence,
but it is not that contract: its current `v1` payload covers only record
`status`/`updated_at`, links, and `SUPERSEDES` target summaries. It omits the
semantic fields required to reject stale curation plans and to restore state.
It also must not be confused with a curation identity merely because both use
a `v1:<sha256>` spelling.

## Decision

Use the following identity contract. All identities in this ADR use canonical
JSON bytes and SHA-256. The contract version is `1`; any incompatible change
increments it.

### Canonical JSON

1. Construct a JSON value using the field sets below.
2. Normalize every string to Unicode NFC. Normalize `CRLF` and `CR` to `LF`.
   Do not trim, collapse, or otherwise change whitespace in content, title,
   summary, link context, rationale, or evidence text. Identifiers,
   enumerated values, and tag values are trimmed of leading/trailing Unicode
   whitespace; an empty result is invalid rather than converted to null.
3. Preserve `null` as JSON `null`; do not omit a declared nullable field or
   replace null with an empty string/list/map. Required absent values are
   invalid, not null.
4. Serialize with UTF-8, `ensure_ascii=false`, sorted object keys,
   separators `(',', ':')`, and no trailing newline. Numbers must be finite;
   use JSON integer/decimal values without a locale or exponent rewrite.
   Boolean values remain booleans and are never treated as integers.
5. Lists whose meaning is a set are sorted after normalization. Lists whose
   order is semantic retain their order. Maps are objects with recursively
   sorted keys; duplicate keys are invalid.

The digest representation is lowercase hexadecimal SHA-256 of those exact
UTF-8 bytes, prefixed with `v1:` (for example, `v1:0123...64 hex digits...`).
The prefix is part of the stored token, not part of the bytes being hashed.

### Versioned semantic record snapshot

The canonical record snapshot is an object with exactly these keys:

```json
{
  "schema_version": 1,
  "record": {
    "id": "...",
    "title": "...",
    "content": "...",
    "summary": "...",
    "type": "...",
    "status": "...",
    "tags": ["..."],
    "workspace_ids": ["..."],
    "lineage": {"...": "..."},
    "mutation_metadata": {"...": "..."}
  }
}
```

`title`, `content`, and `summary` are strings or explicit null where the
authoritative model permits null. `tags` and `workspace_ids` are normalized,
deduplicated, and sorted by normalized UTF-8 byte sequence. `type`, `status`,
and `id` use normalized identifier handling. `lineage` contains only the
authoritative lineage fields (including parent/child/sibling and source
mapping identifiers); `mutation_metadata` is an explicit allowlist of
mutation-relevant metadata, recursively canonicalized. Unknown metadata keys
are excluded, not copied opportunistically. A missing optional allowlisted
field is represented according to its declared schema default; a declared
nullable field is represented as null.

The following are excluded: `read_count`, `access_score`,
`last_accessed_at`, `last_surfaced_at`, retrieval/search scores, cache
timestamps, internal row/update timestamps, provider transcripts and token
usage, task leases, and other operational or access telemetry. Exclusion is
intentional: those values must not make a semantic plan stale, and restore
must not erase later access telemetry.

The **record token** is the digest of this snapshot. SQLite, Postgres,
planner validation, mutation history, executor, verifier, and restore must
produce the same bytes for the same logical snapshot. A record token changes
when any included scalar, tag/workspace member, lineage member, or allowlisted
mutation metadata changes; it does not change for excluded telemetry.

### Graph snapshot and graph token

For a focal record, the graph snapshot is:

```json
{
  "schema_version": 1,
  "memory_id": "...",
  "edges": [
    {"source_id": "...", "target_id": "...", "type": "...", "context": "..."}
  ]
}
```

Include every relevant incoming and outgoing edge visible to the authoritative
query, including lineage edges. Normalize edge type as an identifier and
context as a string; do not include database edge IDs or creation/update
timestamps. Deduplicate exact edge objects and sort edges by
`(source_id, target_id, type, context)` after normalization. The **graph
token** is the digest of this graph snapshot. A graph action must use the
graph token in addition to the affected record token; a record-only action
does not acquire graph identity merely from unrelated edges.

### Context and frontier fingerprints

The **frontier fingerprint** is the digest of:

```json
{"schema_version": 1, "family": "...", "strategy": "...", "seed_ids": ["..."]}
```

`seed_ids` are normalized, deduplicated, and sorted. Support-record IDs are
not included: changing support evidence changes context, not frontier
identity. The existing `frontier_key` may use this same value.

The **context fingerprint** is the digest of the exact immutable packet
supplied to the planner:

```json
{
  "schema_version": 1,
  "frontier_fingerprint": "v1:...",
  "seeds": [...],
  "support": [...],
  "record_tokens": {"id": "v1:..."},
  "graph_tokens": {"id": "v1:..."},
  "disclosure": {...},
  "omissions": [...],
  "limits": {...}
}
```

The packet preserves semantic list order where it is presented order (seed
order and support ranking); maps are key-sorted. It includes redaction,
truncation/omission reasons, disclosed field names, revision tokens, and
budget limits because each affects what the planner was allowed to see.
Volatile timestamps, request IDs, provider identity, telemetry counters, and
the run/plan IDs are excluded. A changed packet byte therefore changes the
fingerprint; reordering set-like IDs or map keys does not.

### Deterministic action IDs

The harness assigns `action_id` after typed schema validation. It does not
accept a provider-supplied ID. Define the action-name input as the UTF-8
canonical JSON bytes of:

```json
{
  "schema_version": 1,
  "plan_id": "...",
  "position": 0,
  "operation": "...",
  "target_ids": ["..."],
  "arguments": {...},
  "preconditions": {...},
  "evidence": [...]
}
```

`position` is the zero-based position in the validated action list.
`target_ids` are normalized and sorted; operation arguments, preconditions,
and evidence use their typed schemas, with set-like IDs/assertions sorted.
Exclude `action_id`, rationale, confidence, provider transcript data, and
other advisory/volatile fields. Thus formatting-only retries retain an ID,
while changed operation intent, target, precondition, evidence, plan, schema,
or position does not. Derive the ID as UUIDv5 under a fixed curation action
namespace UUID recorded in the implementation, using the canonical JSON
string (the UTF-8 bytes decoded as UTF-8) as the UUIDv5 name. The namespace
must never be generated per run; changing it requires a new schema version.

### Schema-version behavior and edge coverage

Every consumer validates `schema_version == 1` before hashing or comparing.
Unknown, missing, or malformed versions fail closed as an incompatible
identity; they are not silently downgraded. A schema-version bump creates new
tokens even if the logical fields are unchanged. Stored old tokens remain
readable as opaque values for audit, but cannot satisfy a new precondition
without an explicit migration that recomputes the versioned snapshot.

Implementations and contract tests must cover: empty strings; whitespace-only
and leading/trailing whitespace; tabs and repeated spaces; LF/CRLF/CR;
Unicode composed/decomposed equivalents and non-ASCII; null versus empty
string/list/map; missing optional fields; duplicate and differently ordered
tags, workspace IDs, map keys, edges, and evidence; semantically ordered
lists; empty record/graph/context/frontier sets; duplicate IDs; escaped JSON
characters; large integers and finite decimals; booleans; invalid UTF-8,
NaN/Infinity, unknown keys, malformed tokens, unknown schema versions,
changed semantic fields, changed excluded telemetry, changed edge context,
and action intent changes. SQLite and Postgres must be tested against the
same canonical byte fixtures, not only against equal digest strings.

## Consequences

- Curation and history get one explicit semantic identity contract instead of
  inferring safety from access-cache behavior.
- The current shared read-cache validation token remains a legacy derivative
  cache validator until a separate implementation change adopts this contract;
  this ADR does not change runtime code.
- Canonical snapshots are deliberately strict and may reject malformed data;
  failing closed is safer than making a stale plan appear current.
- Full snapshots belong in mutation history where required; receipts and
  cache entries should store compact tokens, not duplicate content.
