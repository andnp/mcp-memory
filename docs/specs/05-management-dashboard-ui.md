# Specification: Memory Command Center (UI Overhaul)

**Status:** Draft

## 1. Vision: The "Memory Pulse"
Transition from a static "monitoring dashboard" to an interactive **Memory Command Center**. The UI should feel "alive," reflecting the autonomous nature of the background agents and providing a low-friction interface for human thought stashing and memory exploration.

## 2. Core Functional Pillars

### 2.1 The Global Command Bar (P0)
- **Concept:** A prominent, persistent search and input bar (Ctrl+K style).
- **Functionality:**
    - **Search:** Real-time, fuzzy/semantic search through the memory bank.
    - **Stash:** Instantly record a new "thought" into System 1.
    - **Command:** Quickly trigger background agents or administrative actions.

### 2.2 The Agent Audit Stream (P1)
- **Concept:** A live, terminal-like feed of agent reasoning.
- **Content:** Show what each agent is doing in real-time (e.g., "Taxonomist is merging duplicate tags: `testing`, `tests` -> `pytest`").
- **Transparency:** Every agent mutation should be visible and, where possible, show the "Before/After" diff.

### 2.3 Graph Visualization (P1)
- **Concept:** A force-directed graph of the memory base.
- **Visuals:** Nodes are memories, edges are links (`SUPERSEDES`, `DEPENDS_ON`, `CONTRADICTS`).
- **Interactivity:** Click a node to view its full content, lineage, and related memories.

### 2.4 The Conflict Inbox (P2)
- **Concept:** A dedicated area for resolving contradictions.
- **Functionality:** Surfaces memories linked by `CONTRADICTS`. Allows the user to manually merge, archive, or ignore the conflict.

### 2.5 Nerd metrics analytics IA
- The nerd dashboard is now treated as a phased analytics surface instead of one giant undifferentiated panel.
- The concrete backend contract for the first rollout lives in `docs/specs/12-nerd-metrics-analytics.md`.
- Slice 1 focuses on three foundational panels:
    - **Composition** — breakdowns by workspace/project, tag, type, and status
    - **Distributions** — created age, updated age, and content-size buckets
    - **Timelines** — created/updated memory activity and cumulative content bytes over time
- Deeper lifecycle, search-quality, graph-intelligence, and anomaly panels remain later slices layered onto the same `/api/metrics/nerd` contract.

## 3. Tech Stack Overhaul
- **Frontend:** React + Vite (for fast iteration and modern component support).
- **Styling:** Tailwind CSS (for consistent, modern aesthetics).
- **Visuals:** D3.js or `react-force-graph` for the knowledge map.
- **State:** React Query for real-time synchronization with the daemon API.

## 4. User Experience Goals
- **Monospace Aesthetic:** Retain the professional, senior-engineer feel (JetBrains Mono, dark mode).
- **Immediate Feedback:** Stashing a thought should provide instant visual confirmation.
- **Deep Traceability:** Users should be able to trace a memory from its raw System 1 origin to its current System 2 canonical form.

## 5. Implementation Phases
1.  **Phase 1 (Foundation):** Setup Vite + React + Tailwind. Implement the Global Command Bar (Search/Stash).
2.  **Phase 2 (Agentic Visibility):** Build the Agent Audit Stream and the real-time "Pulse" indicators.
3.  **Phase 3 (Analytics Fundamentals):** Ship the first nerd metrics slice from `docs/specs/12-nerd-metrics-analytics.md` behind `/api/metrics/nerd`.
4.  **Phase 4 (Exploration):** Integrate the Graph Visualization, conflict workflows, and deeper analytics drill-downs.
