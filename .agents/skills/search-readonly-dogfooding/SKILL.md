---
name: search-readonly-dogfooding
description: Dogfood mcp-memory search with bounded, read-only queries and evidence capture. Use when validating retrieval quality, workspace scope, diagnostics, cache behavior, or released searchkernel integration.
---

# Read-only Search Dogfooding

Use the live memory corpus as the test corpus, but never mutate it during this
workflow. Do not call `record_thought`, mutation tools, enqueue maintenance, or
approve curation actions. Search and read only the records needed to support a
conclusion.

The agent may restart the local daemon when it is stale, unhealthy, or
explicitly requested: run `uv run mcp-memory daemon restart`, then verify
`uv run mcp-memory daemon status` reports a running daemon, expected project
binary, and ready runtime before collecting search evidence. Report restart
failures as deployment/outage findings rather than negative search results.

## Protocol

1. State the question, time window, intended scope, runtime version, and
   searchkernel dependency version. Test global behavior without a workspace
   filter; test isolation with an explicit filter. Treat the caller workspace
   as ranking context, not isolation.
2. Run a bounded query matrix covering conceptual terms, synonyms, vague terms,
   multi-term queries, ordinary filename/symbol/commit text, unrelated
   controls, and duplicate candidates. Keep each category small and record the
   exact query strings. Exact memory identifiers are read semantics, not a
   searchkernel search-quality contract.
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
7. After producing the report, launch a read-only triage subagent. It should
   review the evidence, reproduce the highest-impact findings with bounded
   searches or health checks, separate observations from hypotheses, classify
   ownership, identify any release implications, and produce a proposed
   implementation commit sequence with estimated scope. The subagent must not
   mutate the corpus, edit code, or implement the plan; review and approve the
   commit plan before implementation begins.
8. Separate observed behavior from hypotheses. Classify each actionable gap as
   searchkernel, mcp-memory, curator/maintenance, or deployment/release work.
   If an upstream hook is missing, document the exact API gap and reproducible
   query; an explicitly authorized follow-up may inspect the sibling
   searchkernel checkout and prepare a release handoff, but dogfooding itself
   must remain read-only.
9. Treat any searchkernel search-quality code change as upstream release work:
   it requires a new PyPI release, an mcp-memory dependency and lockfile
   update, source/PyPI/runtime verification, and only then attribution of
   behavior changes to the release. Verify the source commit, published PyPI
   version, resolved lockfile version, and live runtime version.

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
