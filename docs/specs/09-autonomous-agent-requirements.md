# Specification: Autonomous Agent Requirements

**Status:** Current runtime supports a focused subset of the full agent vision.

## 1. Current Agent Matrix

| Agent | Implemented | Needs AI? | Notes |
| :--- | :--- | :--- | :--- |
| Ingestor | Yes | Optional | Uses provider when available; falls back to deterministic grouping/create behavior |
| Summarizer | Yes | Optional | Uses provider for summaries when available |
| Fact Checker | Yes | No | Deterministic validation of explicit `ext:` targets |
| Project Manager | Yes | No | Deterministic plan aging |
| Sweeper | Yes | No | Deterministic task/journal cleanup |
| Graph Linker | No | Yes | Future work |
| Conflict Detector | No | Yes | Future work |
| Defragmenter | No | Yes | Future work |
| Taxonomist | No | Yes | Future work |

## 2. Current Provider Support

### Implemented
- `none`
- `gemini-cli`

### Not Yet Implemented
- `copilot-cli`
- `opencode`
- `ollama`

## 3. Current AI Expectations

### Ingestor
- can use an AI provider to propose `create` or `ignore` actions
- falls back to deterministic grouping when provider support is unavailable or invalid

### Summarizer
- can use an AI provider to generate memory summaries
- falls back safely when provider support is unavailable

### Deterministic Agents
- Fact Checker
- Project Manager
- Sweeper

These agents do not require AI in the current design.

## 4. Future Agent Expansion

Future agents should only be added when their operational behavior is clear and testable.

### Candidate Future Agents
- Graph Linker
- Conflict Detector
- Defragmenter
- Taxonomist

### Expected Requirements for Future AI Agents
- explicit structured-output contracts when mutating data
- graceful fallback behavior when the provider is unavailable
- durable task execution through the shared SQLite task queue
- clear management-surface visibility for failures

## 5. Open Scope Decisions

Before expanding the agent roster, the project should decide:

- whether the current five-agent subset is the intended v1 scope
- which providers must be supported for v1
- whether advanced maintenance sampling is required before ship
