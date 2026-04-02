## ADR: Stats-driven batch selection for agentic maintenance

- **Status:** Accepted
- **Date:** 2026-04-01

## Context

`mcp-memory` already supports multiple candidate-selection strategies for maintenance work through `RouletteProvider` in `src/mcp_memory/core/sampling.py`. That strategy menu is useful, but the current default choice is still roulette-based when a task does not explicitly request a valid strategy.

That randomness was acceptable for the first rollout because it kept candidate exploration broad while preserving deterministic task-queue ordering. It is now the wrong default for curator work:

- the curator task is increasingly aimed at retrieval-quality repair, not generic coverage
- the candidate set already carries cheap live signals that indicate what kind of review is most needed right now
- maintenance operators need more legible selection behavior and better observability than “roulette picked this”
- the rest of the maintenance architecture still benefits from small, reversible steps rather than a cross-task redesign

This repo’s architecture principles also push toward conservative background automation with deterministic fallback paths and explicit, inspectable behavior.

## Decision

We will move agentic maintenance batch selection toward **stats-driven strategy choice**, beginning with a narrow first slice for the curator task only.

For this first slice:

1. **Curator defaults become deterministic and signal-driven.**
   - When `task_name == "memory-curator"` and no explicit valid strategy is requested, strategy selection is based on cheap live candidate-set signals instead of roulette.
   - Example signals include oversized-share, never-surfaced share, cold-tail / never-accessed share, low-support share, and lightweight retrieval-friction proxies.

2. **Existing strategy rankers stay unchanged.**
   - `semantic`, `anomaly`, `cold-storage`, `never-surfaced`, `orphan/low-support`, and `bounded-noise` still rank candidates the same way once selected.
   - This ADR changes *which strategy is chosen by default*, not how each strategy produces a batch.

3. **Other maintenance tasks remain on current behavior.**
   - Deduplicator, taxonomist, and the rest of the maintenance stack keep the current seeded roulette selection behavior for now.

4. **Explicit requests still win.**
   - If a caller explicitly requests a valid allowed strategy, that request is honored.
   - Invalid or disallowed requested strategies still fall back safely.

5. **Selection observability becomes part of the batch contract.**
   - Batch metadata may include compact selection hints such as mode, reason text, and score summaries so operators and tests can see why curator chose a strategy.

## Crawl / Walk / Run rollout

### Crawl

Ship deterministic score-based strategy selection for `memory-curator` only.

- use cheap live candidate-set signals already available in memory records and support counts
- keep selection logic local to sampling so handler churn stays minimal
- preserve existing fallback semantics
- expose compact selection metadata on curator batch responses and task-result payloads when available

### Walk

Expand stats-driven selection to other maintenance tasks where the task’s objectives clearly map to live signals.

- deduplicator can bias toward semantic density, overlap, and cooldown pressure
- taxonomist can bias toward tagging gaps, stale taxonomy, and low-support clusters
- relationship tasks can bias toward bridge or conflict signals

At this stage we should also compare selected strategy, resulting mutations, and maintenance yield so the selector can be evaluated against actual task outcomes rather than intuition alone.

### Run

Promote strategy selection into a shared maintenance policy layer driven by persisted telemetry.

- use historical yield and guardrail outcomes, not just batch-local stats
- allow bounded task-specific policies rather than one global heuristic
- keep deterministic queue semantics intact while making the selected strategy explainable and auditable

## Consequences

### Positive

- curator batch choice becomes more legible and repeatable
- maintenance effort shifts toward the most obvious live retrieval-quality gaps
- observability improves without redesigning the whole maintenance stack
- the change is reversible because it leaves the rankers and non-curator tasks alone

### Negative / trade-offs

- the first slice uses heuristic scoring rather than outcome-trained policy
- live-signal selection can still be imperfect when candidate metadata is sparse or noisy
- curator and non-curator tasks temporarily use different default-selection semantics

### Guardrails

- do not change task-queue fairness or claim ordering
- do not change strategy rankers in this slice
- keep batch metadata backwards-compatible by making new selection fields optional
- widen to other tasks only after verification shows the curator slice is useful and stable