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

This plan captures a first-pass architecture review and a refactor roadmap intended to improve maintainability, policy clarity, and implementation velocity before another large wave of feature work.

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
This layer is mostly thin, but it still owns too much workflow glue around daemon/runtime bootstrapping and local-vs-daemon behavior.

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
This layer is feature-rich and fairly robust, but the startup/lifespan path composes many concerns at once, making it harder to reason about degradation and test setup.

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
`ApplicationContext` has become a large service bag. It is useful operationally, but too many parts of the system can reach too many other parts with no narrow abstraction boundary.

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
This is currently the highest-leverage architectural area. Policy and execution are mixed together, and multiple handlers repeat the same orchestration structure with task-specific variations.

### 2.5 Internal API / Management Layer
**Primary files:**
- `src/mcp_memory/mcp/services.py`
- `src/mcp_memory/mcp/internal_services.py`
- `src/mcp_memory/mcp/tools.py`
- `src/mcp_memory/mcp/internal_tools.py`
- `src/mcp_memory/management/service.py`

**Responsibilities:**
- user-facing MCP tool contracts
- internal maintenance MCP tool contracts
- management/dashboard payloads
- stats, diagnostics, task operations, AI conversation visibility

**Observation:**
The internal API surface is powerful, but contract consistency and layering discipline need tightening. Management logic in particular now carries too much metric and orchestration semantics.

## 3. Highest-Risk Architectural Issues

### 3.1 `ApplicationContext` is too broad
**File:** `src/mcp_memory/context.py`

`ApplicationContext` couples:
- config and workspace identity
- persistence services
- search and embedding services
- provider registry and active providers
- task queue and runtime state

This makes it easy to build features quickly, but it weakens module boundaries and makes testing/refactoring more expensive.

### 3.2 `ManagementService` is too large and semantically overloaded
**File:** `src/mcp_memory/management/service.py`

It currently acts as:
- a metrics engine
- a reporting facade
- a task-control surface
- a memory-query adapter
- a dashboard API backend

This increases the chance of semantic bugs in observability logic and makes it harder to evolve metrics cleanly.

### 3.3 Background task handlers duplicate orchestration structure
**Files:**
- `src/mcp_memory/core/task_handlers/ingest.py`
- `src/mcp_memory/core/task_handlers/maintenance.py`

Common repeated structure includes:
- seed/claim selection
- provider-policy choice
- tool-loop or agentic path execution
- finalization / delete / release / cleanup
- structured run metadata assembly

This is the biggest code-quality hotspot because every policy change risks touching multiple handlers.

### 3.4 Provider routing mixes policy and execution
**File:** `src/mcp_memory/core/agent_runtime.py`

Current routing logic still combines:
- task semantics
- capability preference
- budget checks
- route ordering
- fallback selection
- binding usage context

This should become a more explicit policy object or module so handlers and workers can depend on a simpler interface.

### 3.5 Internal MCP tool contracts are not yet uniform enough
**Files:**
- `src/mcp_memory/mcp/internal_tools.py`
- `src/mcp_memory/mcp/internal_services.py`
- `src/mcp_memory/mcp/services.py`

The surface is useful, but naming, error semantics, and service conventions still feel organically grown rather than deliberately standardized.

### 3.6 Tests are too exposed to host/runtime ambient state
**Relevant areas:** provider tests, runtime tests, daemon/CLI tests

We already saw this concretely via:
- real provider CLI discovery hazards
- daemon command path mismatches
- config drift between repo defaults and live user config

The suite needs stronger fixtures and more explicit runtime construction boundaries.

## 4. Design Principles for the Refactor Phase

### 4.1 Separate policy from execution
- **Policy** decides: capability tier, fallback chain, budgets, deterministic bypass, sampling strategy.
- **Execution** performs: claim work, call provider/tool loop, apply mutations, finalize task outcome.

### 4.2 Standardize maintenance-agent orchestration
The ingest / dedup / curator family should share a framework for:
- selecting work
- entering agentic or deterministic paths
- finalizing mutation state
- emitting structured run metadata

### 4.3 Keep daemon boundaries explicit
The daemon should remain the operational center of gravity. Local/runtime construction should be treated as infrastructure, not a casual convenience layer sprinkled throughout the codebase.

### 4.4 Make observability definitions precise
Every dashboard stat and alert should have crisp semantics.
Examples:
- runnable vs scheduled queue age
- provider failure rate windows
- degraded vs stale vs archived memory states
- candidate_count vs seed_count vs touched_count

### 4.5 Tighten internal API contracts
Internal maintenance tools should feel like a stable internal platform, not just a collection of useful helpers.

## 5. Refactor Roadmap

### 5.1 Do Now — Highest-Leverage Refactor Tranche

#### Refactor A: Shared maintenance-agent execution framework
**Goal:** extract a common orchestration skeleton for ingest, deduplicator, and curator.

**Potential shape:**
- work selection / seed acquisition
- policy resolution
- agentic execution path
- deterministic fallback path
- finalization/cleanup hook
- structured result metadata builder

**Expected payoff:**
- less duplicated control flow
- easier policy changes
- more uniform metrics and tests
- easier future agent additions

#### Refactor B: Provider routing policy extraction
**Goal:** move route/capability/fallback semantics into a dedicated policy module.

**Expected payoff:**
- fewer branching conditionals in runtime glue
- clearer capability-tier semantics
- easier testability of routing behavior
- less handler-level coupling to provider details

#### Refactor C: Normalize task run metadata contracts
**Goal:** standardize fields emitted by all maintenance handlers.

**Expected payoff:**
- simpler management metrics
- fewer one-off UI/dashboard transformations
- easier debugging and comparison across agents

### 5.2 Do Next — Boundary Cleanup

#### Refactor D: Internal MCP tool/service contract cleanup
**Focus:**
- naming consistency
- argument validation consistency
- error contract consistency
- clearer separation of user-facing vs internal surfaces

#### Refactor E: Split `ManagementService`
Break it into narrower components such as:
- task/queue reporting
- memory metrics/reporting
- provider/runtime telemetry
- dashboard composition

#### Refactor F: Test architecture cleanup
**Focus:**
- explicit runtime factories
- safer provider/CLI isolation
- less dependence on ambient user config and installed tools

### 5.3 Later — Deeper Structural Cleanup

#### Refactor G: Narrow `ApplicationContext`
Split it into smaller service bundles or typed facades.

#### Refactor H: Daemon/runtime ownership cleanup
Make local runtime construction and daemon-backed operation more deliberately separated.

#### Refactor I: Dashboard semantics & alert taxonomy pass
Tighten definitions and make operational alerts consistently actionable.

## 6. Immediate Recommendation

The first implementation tranche should be:

### **Background-agent framework + provider policy cleanup**

This is the best leverage point because it touches:
- maintainability
- policy clarity
- future feature velocity
- observability quality
- cost and quality control

It is also the area where recent complexity has accumulated fastest.

## 7. Proposed First Coding Slice

A minimal, behavior-preserving first slice should:

1. introduce a shared maintenance-agent execution abstraction
2. migrate one handler family onto it first
3. extract provider routing policy behind a narrower interface
4. preserve all current external behavior and management payload contracts
5. add explicit tests for policy-vs-execution boundaries

Recommended candidate order:
1. `deduplicator`
2. `memory-curator`
3. `ingest-system1`

Rationale:
- deduplicator already has a relatively crisp seed-selection + execution pattern
- curator is high-value and policy-heavy
- ingest has the most task-specific finalization semantics and should likely move last

## 8. Completion Criteria for the Review Phase

This architecture review tranche is complete when we have:
- this durable written plan
- high-fidelity memories capturing the structural findings
- a clear first refactor target
- agreement that the next coding work should prioritize architectural leverage over new feature surface
