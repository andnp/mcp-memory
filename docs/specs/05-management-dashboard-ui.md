# Specification: Management Dashboard (Web UI)

**Status:** Current runtime exposes a minimal read-only dashboard.

## 1. Current Product Shape

The active dashboard is a localhost operations view for the daemon.

### 1.1 Current Capabilities
- dark, monospace presentation
- daemon-backed overview cards
- recent memories table
- failed tasks table
- memory detail endpoints through the management API
- read-only runtime health and overview endpoints

### 1.2 Current Non-Goals
The current dashboard does **not** yet provide:

- memory editing
- relationship editing
- tag management
- graph visualization
- temporal filtering UI
- conflict resolution UI
- audit log streaming
- broad analytics suite

## 2. Current API Surface

The dashboard consumes the daemon’s localhost API.

### 2.1 Active Endpoints
- `/`
- `/api/health`
- `/api/overview`
- `/api/tasks`
- `/api/memories`
- `/api/memories/{memory_id}`

## 3. Current UX Goals

- low-friction daemon observability
- visibility into recent memories
- visibility into failed tasks
- safe read-only access during early iterations

## 4. Future Expansion Options

If the dashboard grows beyond read-only, the next most plausible additions are:

- memory editor
- explicit relationship authoring
- tag management
- conflict inbox
- agent audit visibility
- richer search and filtering controls

## 5. Deliberate Open Decision

The dashboard is currently a read-only operational tool.
A future product decision is required before building the larger command-center UI described in earlier drafts.
