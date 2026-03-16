# AGENTS.md

## What this repo is

`mcp-memory` is the source repository for the MCP Memory Server and its runtime.
It is a standalone Model Context Protocol server for persistent AI memory management.
The active product shape is relational-first, backed by one shared global SQLite store, with a thin MCP stdio proxy, a daemon/runtime layer, background maintenance agents, and a small management API/dashboard.

This repo is also the implementation of the memory tools that agents use while working here.
That makes this repository a dogfooding environment: use the memory system heavily, and treat gaps in the memory experience as product feedback, not just workflow friction.

## Core working stance

- Make minimal, targeted changes.
- Prefer direct, factual language.
- Preserve existing behavior unless the task explicitly requires a change.
- Treat docs, specs, memory records, and code as complementary sources of truth.
- When touching behavior, prefer understanding the product intent first, then changing code.

## Version control in this repo

This repository does **not** use GitButler.
Agents may use plain `git` here.

Keep the same behavioral standard either way:

- make small, atomic commits
- commit only the files relevant to the current change
- avoid mixing unrelated workspace changes into one commit

## Standard QA checks

These are the default quality gates for this repo:

1. `ruff`
2. `pyright`
3. `pytest`

Preferred commands:

```bash
uv run ruff check .
uv run pyright
uv run pytest tests/small/
```

When the change touches broader integration/runtime behavior, widen test scope deliberately:

```bash
uv run pytest tests/medium/
uv run pytest tests/large/
```

Guidance:

- Use `tests/small/` by default while iterating.
- Expand to `tests/medium/` or `tests/large/` only when the change justifies it.
- If you change daemon lifecycle, MCP tool wiring, management API behavior, or end-to-end flows, do not stop at linting alone.
- Do **not** run `ruff format` in this repo. If you need auto-fixes, use `uv run ruff check --fix`.

## Memory usage standards

This repo implements the memory product. Agents working here should use the memory tools more heavily than they would in a normal repo.

### Mandatory workflow

1. Start by querying memory for relevant user preferences, coding standards, testing preferences, architecture context, and task-specific context.
2. Treat memory search as summary-first discovery only.
3. Read the full relevant memories before planning or coding.
4. Record thoughts frequently throughout the task.
5. Finish by recording longer reflections with reusable lessons and product feedback.

### Expected memory-tool behavior

- `search_memory_records` is for discovery.
  It returns compact, summary-first results that help identify which memories to inspect.
- `read_memory_record` is for full context.
  Use it to read the complete memory payload, relationships, and lineage.
- `record_thought` is for ongoing system-1 capture.
  Use it aggressively while working in this repo.

### Heavy-use expectation

Because this repository dogfoods the memory system:

- search memory early
- read relevant memories fully
- record thoughts after meaningful searches, discoveries, edits, and verification steps
- record longer end-of-task reflections than you normally would
- treat weak retrieval, missing context, awkward ergonomics, poor summaries, or unhelpful outputs as notable product issues

## Product-feedback expectation

Agents are explicitly empowered to complain when the memory tools are not useful enough.
That is expected behavior in this repo.

If the memory tools are insufficient in any way, say so clearly and concretely.
Do not silently work around the problem and move on.

When memory tooling falls short, agents should:

- describe what was missing or frustrating
- explain how it affected the task
- note whether the issue was search quality, read quality, thought capture, ranking, context retrieval, ergonomics, latency, or workflow fit
- propose one or more concrete improvements to the user

Useful feedback is specific. Good examples:

- search returned summaries that were too vague to decide what to read
- I needed relationship context earlier in the workflow
- thought capture was easy, but retrieval did not surface the thought when it became relevant
- memory search found the right record only after multiple query reformulations
- the tool surface encouraged manual repo spelunking when memory should have carried more of the load

## Planning guidance for this repo

Plan in idea space first.
Use product principles, specs, README, and relevant memories to understand intended behavior before changing implementation details.
This matters especially in this repo because product semantics and tool semantics are easy to accidentally drift apart.

## Repo-specific focus areas

When relevant, pay close attention to:

- relational memory semantics
- MCP tool contracts
- daemon/runtime boundaries
- global vs workspace-scoped behavior
- background task behavior
- provider telemetry and management surfaces
- search ranking and memory retrieval quality

## Completion standard

Before considering work complete:

- the relevant QA checks should pass, or the reason for not running them should be explicit
- docs and behavior descriptions should still match reality
- important task learnings should be captured in memory
- any notable weakness in the memory tooling should be surfaced to the user with a concrete improvement suggestion
