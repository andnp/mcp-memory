---
name: search-readonly-dogfooding
description: Dogfood mcp-memory search with bounded, read-only queries and evidence capture. Use when validating retrieval quality, workspace scope, diagnostics, cache behavior, or released searchkernel integration.
---

# Read-only Search Dogfooding

Use the live memory corpus as the test corpus, but never mutate it during this
workflow. Do not call `record_thought`, mutation tools, enqueue maintenance, or
approve curation actions. Search and read only the records needed to support a
conclusion.

## Protocol

1. State the question, time window, intended scope, runtime version, and
   searchkernel dependency version. Test global behavior without a workspace
   filter; test isolation with an explicit filter. Treat the caller workspace
   as ranking context, not isolation.
2. Run a bounded query matrix covering exact refs/paths/symbols/commits,
   conceptual terms, synonyms, vague terms, multi-term queries, unrelated
   controls, and duplicate candidates. Keep each category small and record the
   exact query strings.
3. Run a baseline without debug, then repeat selected queries with
   `debug: true`. Repeat cold and warm queries when cache behavior or latency
   matters. Keep limits bounded and prefer adaptive search unless exact
   cardinality is the subject of the test.
4. Verify the search-to-read round trip: pass returned `memory_ref` values
   directly to single and batch reads, measure read-conversion for selected
   results, and use `summary_only` or bounded content chunks for large records.
   Request metadata, relationships, or supersession only when the hypothesis
   requires them.
5. Record evidence as a compact table: query, scope/filter, runtime/package
   version, returned refs, relevant summaries, selected reads, diagnostics,
   degradation/cache signals, latency samples, and conclusion. Do not copy
   full memory bodies into reports.
6. For quality or duplicate hypotheses, report a curator-ready signal with
   candidate refs, rationale, confidence, and suggested owner. Do not merge,
   archive, split, collapse, or enqueue results while searching.
7. Separate observed behavior from hypotheses. Classify each actionable gap as
   searchkernel, mcp-memory, curator/maintenance, or deployment/release work.
   If an upstream hook is missing, document the exact API gap and reproducible
   query; an explicitly authorized follow-up may inspect the sibling
   searchkernel checkout and prepare a release handoff, but dogfooding itself
   must remain read-only.
8. After an upstream fix is released, verify the source commit, PyPI version,
   lockfile resolution, and live runtime version before attributing behavior
   changes to the release.

## Evidence rules

- A search result is evidence of retrieval, not proof that a memory is correct.
- Distinguish global recall from workspace ranking uplift.
- Treat kernel failures, cache mismatches, stale fallbacks, partial results, and
  provenance warnings as first-class findings.
- Verify claims with a second query or a bounded read when practical.
- Compare surfaced results with selected reads; retrieval quality is not just
  candidate count.
- Report repeated-query latency as samples or a range, not a single anecdote.
- If the memory transport or search/read tool is unavailable, report the
  outage explicitly and do not convert missing evidence into a negative result.
