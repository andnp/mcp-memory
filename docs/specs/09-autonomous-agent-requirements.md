# Specification: Autonomous Agent Requirements

## 1. Overview
The `mcp-memory` daemon runs 9 distinct background agents to maintain the memory graph. To optimize for cost, speed, and privacy, we must be deliberate about which agents require AI, and more importantly, *what class* of AI they require.

Not every task requires a frontier reasoning model (e.g., `gemini-cli` with Gemini 1.5 Pro). Many can be handled by fast, weak local models (e.g., `ollama` with Llama 3 8B) or pure deterministic code.

## 2. Agent Categorization Matrix

| Agent | Task | Needs AI? | Model Tier Required | Structured Output? |
| :--- | :--- | :--- | :--- | :--- |
| **Ingestor** | System 1 → System 2 mapping | Yes | **Medium/Strong** | Strict JSON |
| **Summarizer** | 2-sentence search summaries | Yes | **Weak/Fast** | Text |
| **Graph Linker** | Inferring `DEPENDS_ON` edges | Yes | **Medium** | Strict JSON |
| **Conflict Detector** | Finding contradictions | Yes | **Strong (Reasoning)** | Strict JSON |
| **Defragmenter** | Synthesizing 30 journals into 1 | Yes | **Strong (Context)** | Text + JSON |
| **Taxonomist** | Canonicalizing Tags | Yes | **Weak** | Strict JSON |
| **Fact Checker** | Verifying `ext:` file paths | **No** | *Deterministic Code* | N/A |
| **Project Manager** | Flagging 60-day old plans | **No** | *Deterministic SQL* | N/A |
| **Sweeper** | Purging old telemetry | **No** | *Deterministic SQL* | N/A |

## 3. Deep Dive: AI-Driven Agents

### 3.1 The Ingestor (Medium/Strong)
- **Role**: Reads messy, human-written "thoughts" and decides how to mutate the System 2 graph.
- **Why it needs Medium/Strong AI**: It has to make routing decisions. Does this thought belong in an existing plan? Is it a brand new fact? If it modifies an existing memory, the AI must output the exact JSON payload to append or rewrite the content safely.
- **Fallback**: If only a weak model is available, the Ingestor should default to just creating *new* `observation` memories rather than attempting complex merges.

### 3.2 The Conflict Detector (Strong)
- **Role**: Scans highly similar facts/plans to see if they disagree.
- **Why it needs Strong AI**: Detecting nuance is hard. "Use SQLite for local dev" and "Use Postgres for production" are highly similar in vector space, but *do not* contradict. A weak model will generate constant false positive `CONTRADICTS` flags.
- **Execution**: Should only run when the daemon has access to a reasoning-capable provider (like `copilot-cli` or `gemini-cli`).

### 3.3 The Defragmenter (Strong)
- **Role**: Garbage collection for dense clusters of `journal` memories.
- **Why it needs Strong AI**: It has to read potentially thousands of words of rambling daily logs and synthesize them into a concise, high-signal `reflection` memory. This requires a large context window and strong narrative generation capabilities.

### 3.4 The Summarizer (Weak/Fast)
- **Role**: Writes the 2-sentence summary used by the `search_memories` tool.
- **Why it needs Weak AI**: Summarization is a solved problem for small models. Since this runs every time a memory is created or updated, it must be fast. A local `ollama` model (e.g., Llama 3 8B or Phi-3) is perfect for this.

### 3.5 The Graph Linker (Medium)
- **Role**: Determines if two semantically similar memories should be explicitly linked.
- **Why it needs Medium AI**: It requires basic logical deduction ("Does Memory A depend on Memory B?"), but doesn't require massive context windows.
- **Workflow**: Given Memory A and Memory B, prompt: *"Are these related? Return JSON: {"linked": true, "type": "DEPENDS_ON", "reason": "..."}"*

### 3.6 The Taxonomist (Weak)
- **Role**: Deduplicates tags (`unit-test` vs `testing`).
- **Why it needs Weak AI**: This is basically a synonym-matching task. We can feed the LLM a list of 100 tags and ask it to output a JSON dictionary mapping redundant tags to a canonical tag.

## 4. Deep Dive: Deterministic Agents (No AI)

These agents run pure Python/SQL and should execute frequently, as their compute cost is effectively zero.

### 4.1 The Fact Checker
- **Execution**: A Python script that queries SQLite for all `links` where `target_id` starts with `ext:`. It extracts the file path, prepends the memory's `workspace_id` (the root project directory), and calls `os.path.exists()`. If False, it runs `UPDATE memories SET status = 'degraded'`.

### 4.2 The Project Manager
- **Execution**: A simple SQL query that runs daily: `UPDATE memories SET status = 'stale' WHERE type = 'plan' AND updated_at < date('now', '-60 days') AND status = 'active'`.

### 4.3 The Sweeper
- **Execution**: A simple SQL query that purges old rows from the `tasks` and `system1_journal` tables to prevent database bloat.
