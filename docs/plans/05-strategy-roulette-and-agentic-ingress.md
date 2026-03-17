# Plan: Strategy Roulette & Agentic Ingress Overhaul

## 1. Context
This plan covers the final evolution of the memory maintenance system. It introduces the **Strategy Roulette** for intelligent batch selection and overhauls the **Ingress** system to give the AI full agentic control over how new thoughts are integrated into the memory base.

## 2. Interface Specifications

### 2.1 The Strategy Roulette (Sampling Engine)
Maintenance agents will rotate through these heuristics to ensure total graph coverage.

Entropy should be introduced at the **candidate-selection layer**, not in the durable task queue or in ingest claim finalization. The queue should remain deterministic for fairness and operability, while each maintenance run gets a reproducible strategy decision and seed batch.

Design guardrails:
- **Deterministic safety rails stay deterministic**: `tasks.claim_next()` ordering, ingest claim/delete/release semantics, and CRUD validation do not become random.
- **Entropy is reproducible per run**: strategy selection should use a task-scoped seeded RNG so a given run can be audited and replayed.
- **Selection is bounded**: strategies sample from active candidates and produce an explicit seed batch; they do not make the entire maintenance pass non-deterministic.
- **Observability is mandatory**: task results and audit logs should record `strategy_used`, candidate counts, and selected seed ids.

Initial heuristic menu:

| Strategy | Logic |
| :--- | :--- |
| **Semantic** | Pick a seed memory using task-scoped entropy, then find its N closest neighbors in vector space. |
| **Cold Storage** | `ORDER BY last_accessed_at ASC` (review old data). |
| **Never Surfaced**| `ORDER BY last_surfaced_at ASC` (review invisible data). |
| **Anomaly** | `ORDER BY abs(char_count - median) DESC` (very large or small). |
| **Bounded Noise** | Random sample from under-touched, non-recently-mutated candidates for stochastic discovery without burning runs on obvious noise. |
| **Graph Bridge** | Surface records with strong semantic neighbors but weak explicit link support. |
| **Orphan / Low Support** | Surface active records with little graph support, low read/access support, or no recent surfacing. |
| **Cooldown Escape** | Temporarily deprioritize recently maintained records so the same hot memories do not dominate repeated runs. |
| **Conflict Frontier** | Surface fact/plan candidates with overlapping topics or embeddings but no contradiction links yet. |

Recommended initial assignment by maintenance agent:

| Agent | Default candidate-selection strategies |
| :--- | :--- |
| **Memory Curator** | `anomaly`, `cold-storage`, `never-surfaced`, `orphan/low-support`, `bounded-noise` |
| **Deduplicator** | `semantic`, `anomaly`, `cooldown-escape` |
| **Graph Linker** | `semantic`, `graph-bridge`, `bounded-noise` |
| **Conflict Detector** | `semantic`, `conflict-frontier`, `never-surfaced` |
| **Defragmenter** | `cold-storage`, `semantic`, `orphan/low-support` |
| **Taxonomist** | `cold-storage`, `never-surfaced`, `bounded-noise` |
| **Ingest System1** | Keep FIFO claim ordering; use semantic grouping only inside the already-claimed batch |

Additional variants remain plausible later, especially weighted recency decay, high-centrality / highly-linked records, stale-plan-specific review, contradiction-adjacent review, and workspace-bridge sampling.

### 2.2 Agentic Ingress Overhaul
The Ingestor agent will no longer follow a rigid "group and promotion" loop. Instead:
- **Empowered Suite**: The Ingestor receives the full suite of internal maintenance tools (`merge`, `append`, `create`, `update`, `link`, `archive`).
- **Goal**: Integrate the latest System 1 thoughts while maintaining a tidy, high-signal System 2.
- **Authority**: The Ingestor can refactor existing memories to better accommodate new information.

### 2.3 Claimed Thought Lifecycle
The ingest task now treats a selected thought batch as a task-scoped claim set rather than a loose pending slice.

- **Claim on selection**: the ingest task atomically claims the selected `system1_journal` rows before analysis.
- **Delete on meaningful success**: if the ingest run exits cleanly and performs meaningful memory work, all thoughts claimed by that task are deleted from the journal under the assumption that they informed the resulting memory maintenance.
- **Release on no-op or failure**: if the run performs no meaningful CRUD-style work, or if the handler fails/cancels, the claimed rows are released back to `pending`.
- **Crash recovery**: worker startup releases orphaned claimed thoughts whose owning task is no longer running.
- **Guardrail**: ignore-only/read-only passes are treated as no-op ingest outcomes and do not consume claimed thoughts.

---

## 3. Implementation Tasks

### Task 1: The Sampling Engine
- [ ] Implement `mcp_memory.core.sampling.RouletteProvider`.
- [ ] Implement `get_batch(strategy: str, limit: int)` supporting the initial heuristic menu and graceful fallbacks when embeddings or graph signals are unavailable.
- [ ] Choose strategies with a task-scoped seeded RNG and return `strategy_used` with each batch.
- [ ] Keep `SQLiteTaskQueue.claim_next()` and `System1Journal` claim/delete/release ordering deterministic.
- [ ] Update maintenance handlers and internal batch acquisition to use agent-specific allowed strategy sets.
- [ ] Add task-result / audit metadata for `strategy_used`, `candidate_count`, and selected seed ids.

### Task 2: Agentic Ingress
- [x] Evolve `handle_ingest_system1_task` to use a task-scoped claim/delete/release lifecycle instead of simple pending→processed promotion.
- [x] Keep the provider-backed internal tool loop for discovery (`search`/`read`/`list`) while preserving the existing action adapter for `create`/`append`/`ignore`.
- [x] Add a no-op guardrail: delete claimed thoughts only after meaningful memory mutation; otherwise release them back to `pending`.
- [ ] Expand the Ingestor to a broader mutation-capable internal tool suite (`merge`, `append`, `create`, `update`, `link`, `archive`).
- [ ] Update the Ingestor prompt to emphasize "tidy maintenance" and "refactoring" over simple promotion.
- [ ] Ensure the Ingestor can handle dependencies between multiple thoughts in one batch.

### Task 3: Provider Expansion (Roadmap Epic 5)
- [ ] Formalize `copilot-cli` and `ollama` provider wrappers.
- [ ] Implement capability detection (e.g., "does this local model support JSON output?").

### Task 4: Dashboard & Visibility (Roadmap Epic 6)
- [ ] Add "Audit Logs" view to the dashboard to show agentic mutations.
- [ ] Add "Conflict Inbox" to show unresolved `CONTRADICTS` links.

### Task 5: Testing
- [ ] `tests/small/test_roulette.py`: Verify each strategy pulls correct batches.
- [ ] Verify strategy choice is reproducible for a given task seed and does not affect task queue fairness.
- [x] Add focused journal and worker tests for claim / release / delete / orphan recovery semantics.
- [x] Add ingest tests that verify ignore-only runs release claimed thoughts instead of deleting them.
- [ ] `tests/medium/test_agentic_ingress.py`: Verify that the Ingestor correctly refactors existing memories when new thoughts provide superior structure.
