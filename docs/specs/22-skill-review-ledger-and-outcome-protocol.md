# Skill-review ledger and outcome protocol

Status: Active

mcp-memory owns the observation ledger. Devkit and mcp-memory remain independent
applications and communicate only through these versioned MCP wire contracts.

## Read-only ledger v1

`get_skill_review_ledger` accepts `protocol_version: 1`, an exact
`workspace_id`, a bounded `page_size` (1–100), and an optional snapshot-bound
`page_token`. It returns summary-only records plus `ledger_id`, `snapshot_id`,
`total_count`, and `next_page_token`.

The candidate set is exactly `active` memories of type `observation` having the
`skill-observation` tag and the requested workspace. Selection is a direct
relational query, not semantic search. Records are ordered by
`updated_at`, `created_at`, and `id`, ascending. The snapshot is the SHA-256
identity of the complete ordered summary projection; every page carries the
same identity and a cursor from another snapshot is rejected.

The isolated Devkit observer must load every page before invoking the agent.
The complete ledger is injected into the isolated context. Search and read
memory tools are supplemental only, and the observer remains confined to
`skill-manager/skills/`.

## Terminal review writer v2

`commit_skill_review` accepts `protocol_version: 2`. Evidence includes source
hashes, a passed deployment receipt, and the ledger `snapshot_id`. Each
disposition contains a unique `memory_id`, an `outcome`, and an evidence note.
The only persisted outcomes are:

- `done`: the improvement was made, validated, and deployed; the observation
  becomes `archived`.
- `bad`: the observation is invalid or not actionable; it becomes `stale`, and
  the note and evidence metadata are retained.

`untouched` is not a writer disposition. Devkit omits it, leaving the
observation active with no persistence mutation.

The writer validates workspace, observation type, tag, active status, and skill
membership before changing anything. Each request is atomic, idempotent by
`review_run_id`, and rejects conflicting reuse. SQLite and Postgres implement
the same contract. The generic `resolve_skill_observation` actioned/deferred/
verified operation is separate and unchanged.

Devkit writes dispositions only after agent-result validation, source-race
checks, skill validation, deployment checks, and a successful deployment
receipt. An agent decision whose ID is absent from the loaded ledger fails
closed before any disposition write.
