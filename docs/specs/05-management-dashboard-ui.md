# Specification: Management Dashboard (Web UI)

## 1. Design Philosophy: "The Monospace Command Center"
The UI is a developer-centric, high-density dashboard designed for **Observability**, **Correction**, and **Discovery**. It prioritizes keyboard-first navigation and clear, structured data over generic SaaS aesthetics.

- **Theme**: Dark mode by default.
- **Typography**: Monospaced for all content and logs (JetBrains Mono, Fira Code).
- **Color Palette**:
    - `Fact`: Green
    - `Plan`: Blue
    - `Reflection`: Purple
    - `Conflict`: Red
    - `System 1 (Thought)`: Amber/Yellow

## 2. Core Workflows

### 2.1 Basic CRUD & Metadata
- **Memory Editor**: A robust Markdown editor with side-by-side preview.
- **Tag Management**: Inline tag editing with auto-suggestion based on the global `tags` table.
- **Relationship Editor**: A dedicated panel to add/remove UUID-based links without editing the Markdown body.
- **Bulk Actions**: Select multiple memories to tag, archive, or move to the consolidator.

### 2.2 The Interactive Lineage Graph
- **Force-Directed Visualization**: Nodes colored by type, with edge thickness representing relationship strength.
- **Temporal Slider**: A "Time-Travel" bar at the bottom. Dragging it filters the graph to show the state of knowledge at any point in the project's history.
- **Lineage Tree**: Clicking a node isolates its "Supercession Chain," showing exactly how a fact evolved from its original version to the current "Head."

### 2.3 Agentic Maintenance (The "Black Box" window)
- **The Ingestion Pipeline**: A vertical split view showing raw **System 1 Thoughts** on the left and their promoted **System 2 Memories** on the right.
- **Conflict Inbox**: A "To-Do" list of contradictions detected by the `ConflictDetector` agent. Provides a side-by-side diff for resolution.
- **Agent Audit Log**: A streaming feed of background activity (e.g., *"Consolidator merged 4 fragments into 'Auth Spec' [Undo]"*).

## 3. Analytics & Insights
A high-level view of the health and density of the memory bank.

| Metric | Description |
|:---|:---|
| **Memory Density** | Total S2 Memories vs. Total S1 Thoughts (Ingestion efficiency). |
| **Graph Connectivity** | Average links per memory (Higher is usually better for retrieval). |
| **Influence Leaderboard** | Memories with the highest "In-Degree" (Most linked-to facts). |
| **Top Tags** | A ranked list or cloud of the most frequent semantic categories. |
| **Agent Performance** | Total "Noise" removed (Bytes/Files consolidated by agents). |
| **Storage Stats** | Total SQLite DB size, total vector count, and FAISS index health. |

## 4. Search & Discovery
- **Command Palette (`Cmd+K`)**: Unified search bar for memories, tags, and tools.
- **Hybrid Results**: Real-time search results showing calibrated scores, recency boosts, and relationship context.
- **Filter Bar**: Quick toggles for Type, Status (Active/Archived), and Time Range.

## 5. Technical Implementation (UI)
- **Framework**: React or Vue.js for a responsive, single-page experience.
- **Graph Engine**: `D3.js` or `Cytoscape.js` for high-performance relationship visualization.
- **API**: The dashboard connects to the `mcp-memory` daemon via the localhost FastAPI layer.
