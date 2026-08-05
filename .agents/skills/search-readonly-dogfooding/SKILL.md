---
name: search-readonly-dogfooding
description: Subjectively evaluate mcp-memory search against the live corpus and report what was useful, noisy, missing, or wrong.
---

# Live Search Dogfooding

Use the live memory corpus as the test corpus. This is a hands-on product
evaluation, not a formal benchmark. Search and read records, then explain
whether the results would actually help an agent answer its question.

Never mutate the corpus during this workflow. Do not call `record_thought`,
mutation tools, maintenance enqueue tools, or curation approval tools. Do not
create test memories.

## Delegation

Run the dogfooding pass in a read-only subagent. The lead agent provides the
question and scope, verifies daemon readiness when needed, and relays the
result; it should not perform the search-and-read evaluation itself. The
subagent owns the searches, selected reads, subjective judgments, and Good /
Bad / Ugly report.

## Workflow

1. The lead gives the subagent the questions, time window, scope, runtime
   version, and searchkernel version. Search globally by default. Use an
   explicit workspace filter only when checking ranking context or isolation.
2. The subagent picks a small handful of realistic questions from the current
   task or known corpus topics. Use natural language, synonyms, follow-ups, and
   multi-concept questions. Exact memory identifiers are read semantics, not a
   search-quality target.
3. The subagent searches normally. It inspects the top results and reads enough
   returned refs to judge them. Use single or batch reads with bounded content
   when useful.
4. For each question, ask:
   - Did search find the memory I needed?
   - Were the best results near the top?
   - Were the summaries specific enough to predict their value?
   - What was noisy, missing, stale, duplicated, or surprisingly good?
   - Would I trust these results to answer the question?
5. Try a paraphrase or follow-up when it helps reveal a quality gap. Use
   `debug: true`, health checks, and latency samples to explain surprising
   behavior, not as the main evaluation.
6. Report concrete evidence without copying full memory bodies: exact queries,
   returned refs, short summaries, selected reads, and your subjective
   judgment. Separate observations from hypotheses.
7. Report curator-ready duplicate or quality signals with candidate refs,
   rationale, confidence, and suggested owner. Never merge, archive, split,
   collapse, or enqueue anything while dogfooding.
8. After the report, launch a separate read-only triage subagent. It should
   reproduce the most important findings, distinguish facts from hypotheses,
   classify ownership, identify release implications, and propose an
   implementation commit sequence. It must not mutate the corpus, edit code, or
   implement the plan.

## Report shape

Keep the report short and candid:

- **Good:** what worked and should be preserved.
- **Bad:** gaps, weak ranking, missing context, noisy results, or poor
  search-to-read usefulness.
- **Ugly:** bugs, surprising behavior, repeated failures, or operational issues.
- **Follow-up:** likely owner (`searchkernel`, `mcp-memory`, curator,
  deployment/release), confidence, and the next reproduction or implementation
  step.

Treat a returned result as evidence of retrieval, not proof of correctness.
Use selected reads to support judgments. If search or read is unavailable,
report the outage explicitly instead of treating missing evidence as a quality
failure.

## Runtime and release guardrails

The agent may restart a stale or unhealthy local daemon:

```bash
uv run mcp-memory daemon restart
uv run mcp-memory daemon status
```

Verify the daemon is running, uses the expected project binary, and is ready
before collecting search evidence. Report restart failures as deployment or
outage findings.

Searchkernel must remain domain-neutral. If a search-quality change belongs
upstream, it requires a new PyPI release, an mcp-memory dependency and lockfile
update, source/PyPI/runtime verification, and only then attribution of the
behavior change to that release.
