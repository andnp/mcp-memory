# Plan: Strategy Roulette & Agentic Ingress Overhaul

## 1. Context
This plan covers the final evolution of the memory maintenance system. It introduces the **Strategy Roulette** for intelligent batch selection and overhauls the **Ingress** system to give the AI full agentic control over how new thoughts are integrated into the memory base.

## 2. Interface Specifications

### 2.1 The Strategy Roulette (Sampling Engine)
Maintenance agents will rotate through these heuristics to ensure total graph coverage:

| Strategy | Logic |
| :--- | :--- |
| **Semantic** | Pick a random memory, then find its N closest neighbors in vector space. |
| **Cold Storage** | `ORDER BY last_accessed_at ASC` (review old data). |
| **Never Surfaced**| `ORDER BY last_surfaced_at ASC` (review invisible data). |
| **Anomaly** | `ORDER BY abs(char_count - median) DESC` (very large or small). |
| **Noise** | `ORDER BY RANDOM()` (stochastic discovery). |

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
- [ ] Implement `get_batch(strategy: str, limit: int)` supporting all 5 heuristics.
- [ ] Update `handle_memory_curator_task` to use the `RouletteProvider`.

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
- [x] Add focused journal and worker tests for claim / release / delete / orphan recovery semantics.
- [x] Add ingest tests that verify ignore-only runs release claimed thoughts instead of deleting them.
- [ ] `tests/medium/test_agentic_ingress.py`: Verify that the Ingestor correctly refactors existing memories when new thoughts provide superior structure.
