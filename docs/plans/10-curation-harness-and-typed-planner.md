# Direct Agentic Curation

**Status:** Active runtime contract

## Decision

The `memory-curator` campaign uses one simple execution model:

1. Select a small bounded set of seed memories.
2. Open one agentic curator session with internal search, read, relationship, and mutation tools.
3. Let the curator investigate the seeds and make justified mutations directly.
4. Validate the complete sanitized tool-call ledger after the session.
5. Report actual tool results and mutation counts; never infer work from provider prose.

There is no typed plan, retention submission, human-style approval stage, or planner retry loop in the production curator path.

## Runtime boundary

The task handler owns only deterministic setup and cleanup:

- select a claimed review item or direct-sampling frontier
- build the bounded seed context
- enforce the configured mutation budget
- open and close the agentic session
- complete or release the work item
- persist provider usage and the tool ledger

The agent owns the maintenance loop within that bounded session. It may search, read, inspect relationships, update, normalize, split, merge, archive, create links, and remove links using the allowed internal MCP tools. It must not invent memory IDs or claim actions that do not appear in the ledger.

## Ledger validation

The runtime validates the finalized ledger independently of the agent. Validation checks:

- a snapshot exists
- the ledger length matches the total call count
- sequence numbers are contiguous
- every tool is allowed for the curator session
- every entry has a valid status and read/mutation kind
- the recorded kind matches the tool's classification
- successful mutation entries match the mutation counter

Invalid ledgers are reported as `ledger_invalid`. The raw sanitized ledger and actual mutation count remain available for diagnosis, but the run reports zero productive mutations. The runtime does not attempt rollback; storage mutation correctness remains the responsibility of the individual MCP mutation tools and their existing receipts/history.

## Seed selection

Selection remains deterministic and bounded. A claimed `memory_curation_review` item takes precedence; otherwise the handler selects a task-seeded frontier using the configured strategy and support-record ranking. The agent may expand context through search and read tools, but the initial prompt contains only the compact seed set.

## Safety boundary

This design intentionally accepts that the agent has broad mutation freedom. The retained safeguards are narrow and operational:

- internal MCP tools enforce their own argument and storage contracts
- mutation tools write authoritative history and receipts where applicable
- tool calls are tracked independently of provider summaries
- task execution retains single-writer and lease/recovery semantics
- configured session and mutation budgets bound blast radius

Quality evaluation and curator effectiveness are measured from persisted tool activity, receipts, and sampled before/after retrieval checks rather than from plans or model rationale.

## Historical context

Earlier revisions explored a typed planner, plan validation, deterministic action execution, and a separate submission protocol. Those modules and tests are retired from the production curator design. Shared mutation history, receipts, storage preconditions, work-item leases, and runtime recovery remain independent infrastructure.
