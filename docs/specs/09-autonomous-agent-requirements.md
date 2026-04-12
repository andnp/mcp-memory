# Specification: Autonomous Agent Requirements

**Status:** Vision

This is a target-direction document, not the canonical source of current runtime behavior.
Prefer active specs, runbooks, and the root `README` for implemented behavior.

## 1. Overview
The `mcp-memory` daemon runs 9 background agents to maintain the relational knowledge graph. This document defines the model strength requirements and operational logic for each.

## 2. Agent Roster & AI Tier Matrix

| Agent | Task | AI Tier | Structured? |
| :--- | :--- | :--- | :--- |
| **Ingestor** | System 1 → System 2 promotion | Medium | Yes (JSON) |
| **Summarizer** | 2-sentence search summaries | Weak/Fast | No (Text) |
| **Graph Linker** | Discovering `DEPENDS_ON` edges | Medium | Yes (JSON) |
| **Conflict Detector** | Identifying `CONTRADICTS` claims | Strong | Yes (JSON) |
| **Defragmenter** | Synthesis of Journal clusters | Strong | No (Markdown) |
| **Taxonomist** | Canonicalizing & Deleting Tags | Weak | Yes (JSON) |
| **Fact Checker** | Verifying `ext:` file paths | **None** | Deterministic |
| **Project Manager** | Flagging 60-day old plans | **None** | SQL |
| **Sweeper** | Purging old telemetry | **None** | SQL |

## 3. The Strategy Roulette (Heuristics)
All maintenance agents (Linker, Conflict Detector, Defragmenter, Taxonomist) must rotate through these sampling heuristics to ensure total graph coverage:

- **The "Semantic Cluster" Pass**: (Default) Uses FAISS to find memories with high vector similarity.
- **The "Cold Storage" Pass**: Pulls memories based on `last_accessed_at` ascending.
- **The "Never Surfaced" Pass**: Pulls memories based on `last_surfaced_at` ascending.
- **The "Noise" Pass**: Injects randomness by selecting 10-20 IDs at random.
- **The "Anomaly" Pass**: Pulls memories with extreme character counts (too large or too small).

## 4. Model Tiers & Providers

### 4.1 Strong (Reasoning & Large Context)
- **Providers**: `gemini-cli` (Pro), `copilot-cli`.
- **Use Cases**: Conflict Detection (nuance between "A" and "Not A"), Defragmentation (reading 30 journals).

### 4.2 Medium (Logic & Structure)
- **Providers**: `ollama` (Llama 3 8B), `opencode`.
- **Use Cases**: Ingestion, Graph Linking. Must be capable of strict JSON output.

### 4.3 Weak/Fast (Simple NLP)
- **Providers**: Tiny local models.
- **Use Cases**: Summarization, Tag Normalization.

### 4.4 Premium Call Objective

For premium providers such as `copilot-cli` strong models, the system should assume:

- **1 execution call = 1 premium cost unit**
- runtime duration does not materially change that cost unit

Therefore agent design should optimize for:

- **useful work completed per premium call**
- keeping one premium execution alive across additional compatible work when safe
- letting the agent search, inspect, and adapt inside the same paid run instead of reflexively ending and starting another one

Deterministic prep is justified only when it increases what that same premium execution can finish or avoids a later premium re-entry.

## 5. Agent-Specific Logic

### 5.0 Shared Guardrails For Agentic Memory Mutations
- Prefer focused, durable memories. Small-to-medium records are the default target shape.
- Do not merge, append, or rewrite across different projects, products, or repositories unless the memory is explicitly about their relationship.
- Shared generic vocabulary like `architecture`, `daemon`, `infra`, `testing`, `observability`, `roadmap`, or `migration` is not sufficient evidence for combining records.
- Treat transient progress notes, debugging chatter, and one-off execution state as temporal context. Promote them only when they clearly encode reusable long-term knowledge.
- Mutations are value-driven, not quota-driven. A no-op is correct when a change would reduce coherence or search quality.
- When uncertain, prefer reading more context, splitting, or linking over forcing a canonical merged record.
- When a premium execution is already live, prefer pulling more compatible work into the same session over ending early for architectural neatness alone.

### 5.1 The Taxonomist (Empowered)
Unlike other agents, the Taxonomist is explicitly empowered to **delete** and **modify** the global `tags` table to collapse synonyms (e.g., `testing` + `tests` → `pytest`).

### 5.2 The Fact Checker (Deterministic)
Runs pure Python logic to check `os.path.exists` for external links relative to the `workspace_id`. Marks failed memories as `degraded`.

### 5.3 The Project Manager (Deterministic)
Runs pure SQL to mark untouched plans as `stale`.
