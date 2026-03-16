# Product Principles

## 1. One Global Runtime Authority
`mcp-memory` runs as one global daemon per user environment.

- daemon identity must not vary by workspace
- daemon locks and metadata must be global, not workspace-keyed
- clients connect to one shared runtime authority

## 2. One Global Store
The memory store is global.

- memories, tasks, telemetry, and runtime state live in one shared store
- workspaces are not storage shards
- workspaces must not create independent queues, workers, or background runtimes

## 3. Workspace Is Metadata, Not Isolation
Workspace context is descriptive metadata.

- thoughts record the workspace where they were captured
- memories may accumulate one or more workspace associations over time
- workspace association must not imply tenancy, ownership, or execution isolation

## 4. Only Allowed Workspace Semantics
Workspace context is only valid for the following product behaviors:

1. record the workspace when a thought is captured
2. preserve relevant workspace associations when thoughts are incorporated into memories
3. apply a small search-ranking bonus when the client supplies workspace context
4. filter or segment dashboards and analytics

## 5. Forbidden Workspace Semantics
Workspace context must not be used for:

- daemon identity
- daemon startup locks
- queue partitioning
- worker claim filtering
- recurring task fan-out by workspace
- default management/API isolation
- implicit filesystem-root recovery from a workspace identifier

## 6. Search Behavior
Search is global-first.

- the full corpus remains searchable
- workspace context may upweight relevant memories
- workspace context must not silently hard-filter the result set unless an explicit analytics/admin filter asks for it

## 7. Analytics Behavior
Dashboards, logs, provider usage, and similar telemetry may retain workspace as an optional dimension.

- default operational views should favor global visibility
- workspace filters should remain available for inspection and analytics

## 8. Migration Direction
When existing code conflicts with these principles:

- treat product principles as authoritative
- refactor runtime/orchestration boundaries before refining lower-level behavior
- preserve workspace metadata where it helps ranking or analytics, but remove it as a control-plane boundary