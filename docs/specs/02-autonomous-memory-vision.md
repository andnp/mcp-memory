# Vision: Autonomous Memory Consolidation

## 1. Overview
The long-term goal is a low-friction memory system where active clients can quickly record context and the daemon gradually turns that raw input into durable, structured relational memory.

## 2. System Model
- **System 1**
  - fast raw thought capture in `system1_journal`
- **System 2**
  - durable relational memory records in SQLite
  - explicit typed links between memories
  - workspace associations for contextual relevance

## 3. Current Reality
The current runtime already supports:

- `record_thought`
- relational memory CRUD/search/read/import
- daemon-owned background task execution
- ingest and summarization workflows
- deterministic maintenance tasks such as fact checking, plan aging, and sweeping

## 4. Future Vision
Potential future improvements include:

- richer ingest merge/append behavior
- graph linking between related memories
- contradiction detection
- defragmentation of older journals into reflections
- tag normalization
- broader provider support
- richer human review tooling in the dashboard

## 5. Provider Direction
The provider layer should remain wrapper-based and local-developer-friendly.

Candidate providers:
- `gemini-cli`
- `copilot-cli`
- `opencode`
- `ollama`

The project should only ship providers that have clear operational behavior and reliable fallback handling.

## 6. Safety Direction
Autonomous mutation should remain conservative.

- durable task persistence
- visible failed-task state
- explicit review surfaces for higher-risk automation
- no hidden or irreversible background behavior by default
