# mcp-memory Project Roadmap

This roadmap reflects the current daemon-first, relational-first architecture.

---

## Epic 1: Final Legacy Cleanup
**Goal**: remove the last remnants of the extracted pre-relational architecture.

- delete residual dead runtime/test/docs artifacts
- simplify schema and telemetry to only active concepts
- remove stale naming and leftover compatibility wording

## Epic 2: Spec Alignment
**Goal**: make the docs describe the code we actually ship.

- update outdated specs that still describe indexing-era or wikilink-era behavior
- decide which advanced features remain target scope vs. which should be cut
- keep README, roadmap, and specs consistent with each other

## Epic 3: Search and Retrieval Evolution
**Goal**: either formalize the current simple ranking model or implement the missing advanced pipeline.

- decide whether to keep the current relational ranking model
- optionally add calibrated scoring, richer ranking stages, and authority signals
- document the final ranking contract clearly

## Epic 4: Background Agent Expansion
**Goal**: grow beyond the current core maintenance agents when ready.

- Graph Linker
- Conflict Detector
- Defragmenter
- Taxonomist
- strategy-based maintenance sampling

## Epic 5: Provider Expansion
**Goal**: add additional provider wrappers without growing a generic, leaky config surface.

- `copilot-cli`
- `opencode`
- `ollama`
- graceful capability detection and degraded operation modes

## Epic 6: Dashboard and Human Control
**Goal**: decide how far the management UI should go for v1.

- keep read-only dashboard as the minimal baseline
- optionally add editing, relationship authoring, conflict review, and audit visibility
- add only the UI surface that is backed by real workflow needs

## Epic 7: Runtime Hardening
**Goal**: tighten the daemon lifecycle around the final product direction.

- decide whether HTTP localhost transport remains the intended IPC layer
- or replace it with the spec’d lower-level transport model
- add any remaining lifecycle guarantees required by the chosen direction

## Definition of Done for Pre-Ship Cleanup

- no indexing-era/runtime-era dead code remains
- no stale docs imply deleted features still exist
- config surface is minimal and explicit
- daemon/proxy/runtime behavior is clearly documented
- test suite stays green after each cleanup slice
