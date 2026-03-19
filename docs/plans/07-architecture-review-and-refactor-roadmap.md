# Plan: Architecture Review & Refactor Roadmap

## 1. Context

Recent work substantially increased the system's capability surface:
- daemon-backed runtime and thin-client MCP proxying
- management API + dashboard + nerd metrics
- agentic maintenance tools and background agents
- provider routing, budgets, and fallback chains
- entropy/sampling infrastructure
- richer observability and task metadata

The product direction still looks right, but the codebase is now at the point where the main risks are **architectural drift, boundary blur, and duplicated orchestration**, not missing features.

This document started as a first-pass architecture review. It is now updated to reflect the current repository state after several refactor slices already landed, so the roadmap below focuses on **remaining debt**, not work that is already complete.

## 2. Current System Map

The active system has five major layers.

### 2.1 CLI / MCP Proxy Layer
**Primary files:**
- `src/mcp_memory/cli.py`
- `src/mcp_memory/server.py`

**Responsibilities:**
- user-facing CLI commands
- MCP stdio server entrypoints
- daemon start/status/stop/restart commands
- forwarding tool calls into the daemon transport

**Observation:**
This layer is still broadly correct, but `cli.py` remains a large workflow-glue hotspot and still knows too much about bootstrapping, daemon mediation, and local-vs-daemon behavior.

### 2.2 Daemon Lifecycle & Transport Layer
**Primary files:**
- `src/mcp_memory/daemon.py`
- `src/mcp_memory/daemon_app.py`
- `src/mcp_memory/daemon_transport.py`
- `src/mcp_memory/daemon_lifecycle.py`
- `src/mcp_memory/daemon_models.py`

**Responsibilities:**
- daemon startup/reuse/recovery
- lock/metadata lifecycle
- ZMQ transport and request dispatch
- FastAPI management endpoints and hook handling
- binding runtime lifetime to daemon lifetime

**Observation:**
This layer is feature-rich and reasonably robust, but startup/lifespan behavior still composes many concerns at once. The architecture is coherent, yet the ownership boundaries around construction, dispatch, and degradation are still more centralized than ideal.

### 2.3 Runtime Composition Layer
**Primary files:**
- `src/mcp_memory/mcp/runtime.py`
- `src/mcp_memory/context.py`
- `src/mcp_memory/runtime_facades.py`

**Responsibilities:**
- build the runtime
- compose repository, journal, search, task queue, embeddings, providers
- carry config/workspace/runtime-wide dependencies

**Observation:**
`ApplicationContext` remains the broadest dependency carrier in the system. It is operationally convenient, but still too much of a service bag for long-term maintainability.

### 2.4 Background Agent & Task Runtime Layer
**Primary files:**
- `src/mcp_memory/core/agent_runtime.py`
- `src/mcp_memory/core/task_worker.py`
- `src/mcp_memory/core/task_handlers/ingest.py`
- `src/mcp_memory/core/task_handlers/maintenance.py`
- `src/mcp_memory/core/task_handlers/constants.py`
- `src/mcp_memory/core/sampling.py`

**Responsibilities:**
- task claiming/execution/follow-up scheduling
- provider selection and binding
- agentic maintenance execution
- deterministic maintenance fallbacks
- strategy selection and grouping

**Observation:**
This remains the highest-leverage area. The deduplicator path is healthier than before, but `maintenance.py` and `ingest.py` still concentrate orchestration patterns that should be more modular.

### 2.5 Internal API / Management Layer
**Primary files:**
- `src/mcp_memory/mcp/services.py`
- `src/mcp_memory/mcp/internal_services.py`
- `src/mcp_memory/mcp/tools.py`
- `src/mcp_memory/mcp/internal_tools.py`
- `src/mcp_memory/management/service.py`
- `src/mcp_memory/management/*.py`

**Responsibilities:**
- user-facing MCP tool contracts
- internal maintenance MCP tool contracts
- management/dashboard payloads
- stats, diagnostics, task operations, AI conversation visibility

**Observation:**
This area improved materially. `ManagementService` is now more façade-like and delegates to reporting builders, but reporting queries are still schema-coupled and `internal_services.py` remains too mixed by mutation domain.

## 3. Status Update: What Is Already Done

The original roadmap identified several “do now” items that have since landed at least partially.

### 3.1 Provider routing policy extraction is largely complete
The core policy split now lives in:
- `src/mcp_memory/core/provider_policy.py`
- `src/mcp_memory/core/task_policy.py`

`agent_runtime` depends on policy helpers rather than embedding the full routing decision tree inline.

### 3.2 `ManagementService` is no longer the same monolith
Recent refactors split reporting logic into:
- `src/mcp_memory/management/route_audit.py`
- `src/mcp_memory/management/health_reporting.py`
- `src/mcp_memory/management/agent_run_reporting.py`
- `src/mcp_memory/management/overview_reporting.py`
- `src/mcp_memory/management/analytics_reporting.py`

`src/mcp_memory/management/service.py` is still important, but it is now primarily a façade over those builders rather than a 1.3k+ mixed-responsibility blob.

### 3.3 The deduplicator path is now split into clearer seams
Recent slices extracted:
- `src/mcp_memory/core/task_handlers/deduplicator_support.py`
- `src/mcp_memory/core/task_handlers/deduplicator_merge.py`

This reduced the risk and surface area of `handle_deduplicator_task()` and removed one notable private-helper dependency from `internal_services.py`.

## 4. Ranked Architecture Debt List

This ranked list reflects the **current** architecture state and focuses on smells beyond raw LOC counts.

### 4.1 High — `ApplicationContext` is still a broad service bag
**Primary file:** `src/mcp_memory/context.py`

Why it matters:
- too many modules can reach too many other modules through one object
- it weakens boundary discipline even when file boundaries look clean
- some helper/reporting code still fabricates or partially populates context-like state just to reuse policy APIs

Symptoms:
- runtime composition, management reporting, and provider-selection logic still depend on broad ambient context
- testing and refactoring are more expensive than they should be because narrow dependencies are not expressed explicitly

### 4.2 High — background maintenance orchestration is still too concentrated
**Primary files:**
- `src/mcp_memory/core/task_handlers/maintenance.py`
- `src/mcp_memory/core/task_handlers/ingest.py`

Why it matters:
- maintenance handlers still mix policy, work selection, prompting, deterministic fallbacks, and mutation finalization
- adding or changing a maintenance family still tends to touch multiple orchestration layers

What improved:
- deduplicator work is meaningfully better isolated

What still smells:
- graph linker, conflict detector, defragmenter, curator, and shared maintenance helpers still live too close together
- ingest still owns task-specific claiming/finalization logic that wants a smaller service seam

### 4.3 High — reporting/query logic is still schema-coupled
**Primary files:**
- `src/mcp_memory/management/overview_reporting.py`
- `src/mcp_memory/management/analytics_reporting.py`

Why it matters:
- management/reporting logic talks directly to SQLite schema details instead of a dedicated read-model/query layer
- workspace scoping rules and reporting semantics are encoded in multiple places
- schema evolution will be more brittle than necessary

This is a real architectural smell even when the builder split itself is healthy.

### 4.4 Medium — internal MCP maintenance services are too mixed by domain
**Primary file:** `src/mcp_memory/mcp/internal_services.py`

Why it matters:
- read/search/list behavior lives beside ingest-specific mutations and generic maintenance mutations
- internal tool semantics are powerful, but the implementation module is still a kitchen sink
- mutation invariants are harder to audit when multiple domains share one service file

### 4.5 Medium — daemon composition and dispatch are still centralized
**Primary files:**
- `src/mcp_memory/daemon_app.py`
- `src/mcp_memory/daemon_transport.py`

Why it matters:
- lifecycle, worker startup, metadata publication, hook wiring, and request dispatch still gather in a small number of high-authority modules
- the design is coherent, but testability and fault-isolation would improve with narrower composition seams

### 4.6 Medium — extension seams still require too many coordinated edits
**Common touchpoints:**
- `core/task_handlers/constants.py`
- `core/task_policy.py`
- `core/agent_runtime.py`
- management reporting/audit code
- tests

Why it matters:
- adding a new background agent is not hard, but it is not yet a one-obvious-place change
- the architecture still encourages cross-cutting edits for new maintenance capabilities

### 4.7 Low — helper logic is still duplicated in a few places
Examples:
- token/tag normalization heuristics in maintenance-related modules
- reporting bucketing and summary logic split across builder modules

Why it matters:
- mostly a cleanup concern today
- left alone, it will slowly reintroduce drift between maintenance families and reporting surfaces

## 5. Design Principles for the Next Refactor Phase

### 5.1 Separate policy from execution
- **Policy** decides: capability tier, fallback chain, budgets, deterministic bypass, sampling strategy.
- **Execution** performs: claim work, call provider/tool loop, apply mutations, finalize task outcome.

### 5.2 Standardize maintenance-agent orchestration incrementally
The deduplicator split proved that small, family-specific extractions work. Continue applying that pattern to curator, graph/conflict, and ingest rather than attempting one giant framework rewrite.

### 5.3 Keep daemon boundaries explicit
The daemon should remain the operational center of gravity. Runtime construction and daemon-backed operation should be separate on purpose, not just by habit.

### 5.4 Centralize read-model/query semantics for reporting
Builder modules are a good façade layer, but raw reporting queries should live behind a more deliberate read-model/query seam.

### 5.5 Tighten internal API contracts
Internal maintenance tools should continue moving toward a stable internal platform with consistent validation, naming, and mutation semantics.

## 6. Updated Refactor Roadmap

### 6.1 Do Now — Highest-Leverage Remaining Tranche

#### Refactor A: Continue maintenance-family extraction
**Goal:** keep peeling families out of `maintenance.py` using the same behavior-preserving pattern that worked for the deduplicator.

**Best next candidates:**
1. graph linker + conflict detector support
2. curator support
3. defragmentation support

**Expected payoff:**
- smaller maintenance blast radius
- clearer family-level tests
- easier reuse of common orchestration decisions without a premature framework rewrite

#### Refactor B: Introduce a narrow provider-selection input boundary
**Goal:** let provider-selection and route-audit logic depend on a smaller typed input than full `ApplicationContext`.

**Expected payoff:**
- directly attacks the service-bag smell
- cleaner reuse from management/reporting code
- easier policy testing without synthetic context construction

#### Refactor C: Create a reporting query/read-model layer
**Goal:** centralize raw reporting SQL and workspace scoping for overview/analytics builders.

**Expected payoff:**
- less schema-coupled reporting logic
- clearer semantics for dashboard metrics
- easier evolution of management payloads and operational definitions

### 6.2 Do Next — Boundary Cleanup

#### Refactor D: Split `internal_services.py` by mutation domain
**Focus:**
- read/search/list services
- ingest-specific mutation services
- generic maintenance mutation services

#### Refactor E: Extract ingest claim/finalization helpers behind a narrow service seam
**Focus:**
- thought-batch claiming
- cleanup/release/finalization rules
- structured run metadata assembly

#### Refactor F: Test architecture cleanup
**Focus:**
- explicit runtime factories
- stronger provider/CLI isolation
- less dependence on ambient user config and installed tools

### 6.3 Later — Deeper Structural Cleanup

#### Refactor G: Narrow `ApplicationContext`
Split it into smaller service bundles or typed facades once enough consumers can move to narrower inputs.

#### Refactor H: Daemon/runtime ownership cleanup
Make local runtime construction and daemon-backed operation more deliberately separated.

#### Refactor I: Dashboard semantics & alert taxonomy pass
Tighten metric definitions and make operational alerts consistently actionable.

## 7. Immediate Recommendation

The best next implementation tranche is now:

### **Maintenance-family extraction + reporting query-layer cleanup**

That combines the two highest-value remaining themes:
- reduce coordination hotspots in background-agent code
- reduce schema coupling in management/reporting code

Provider policy cleanup is no longer the first-order problem it was in the original draft; the bigger remaining architectural risks are now **broad dependency carriers and mixed orchestration/query seams**.

## 8. Proposed Next Coding Slices

Recommended near-term order:

1. extract graph/conflict support from `maintenance.py`
2. extract curator support from `maintenance.py`
3. introduce a narrower provider-selection input model
4. create a management reporting query layer used by `overview_reporting.py` and `analytics_reporting.py`
5. split `mcp/internal_services.py` by domain

Rationale:
- these are all small, behavior-preserving slices
- each slice tightens one real boundary instead of chasing cosmetics
- each slice reduces future change coupling in an area that still carries meaningful structural debt

## 9. Completion Criteria for This Review Update

This roadmap update is complete when it:
- reflects the current live code rather than the earlier snapshot
- removes already completed refactor items from the “do now” queue
- records the strongest remaining debt in ranked order
- provides a concrete next tranche that favors architecture leverage over new feature surface
