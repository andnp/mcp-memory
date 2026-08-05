---
name: search-readonly-dogfooding
description: Dogfood mcp-memory search with bounded, read-only queries and evidence capture. Use when validating retrieval quality, workspace scope, diagnostics, cache behavior, or released searchkernel integration.
---

# Read-only Search Dogfooding

Use the live memory corpus as the test corpus, but never mutate it during this
workflow. Do not call mutation tools, enqueue maintenance, or approve curation
actions. Search and read only the records needed to support a conclusion.

## Protocol

1. State the question, time window, and intended scope. Omit workspace filters
   for global behavior; provide an explicit workspace filter only when testing
   isolation. Treat the caller workspace as ranking context, not isolation.
2. Run a small baseline query without debug, then repeat with `debug: true`
   when diagnostics are needed. Keep limits bounded and prefer adaptive search
   unless exact cardinality is the subject of the test.
3. Use the returned `memory_ref`, title, and summary to select a few records.
   Read only those records, requesting metadata or relationships only when the
   hypothesis requires them. Use `summary_only` or bounded content chunks for
   large records.
4. Record evidence as a compact table: query, scope/filter, returned refs,
   relevant summaries, diagnostics, degradation/cache signals, and the
   conclusion. Do not copy full memory bodies into reports.
5. For quality or duplicate hypotheses, report a curator-ready signal with the
   candidate refs and rationale. Do not merge, archive, split, or collapse
   results while searching.

## Evidence rules

- A search result is evidence of retrieval, not proof that a memory is correct.
- Distinguish global recall from workspace ranking uplift.
- Treat kernel failures, cache mismatches, stale fallbacks, partial results, and
  provenance warnings as first-class findings.
- Verify claims with a second query or a bounded read when practical.
- If the searchkernel dependency lacks the required released hook, document the
  exact missing API and stop at the integration seam; do not bump dependencies
  or inspect sibling repositories.
