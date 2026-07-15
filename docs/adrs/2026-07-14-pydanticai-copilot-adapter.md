## ADR: Wrap the Copilot provider before introducing a PydanticAI model adapter

- **Status:** Accepted for CUR-004
- **Date:** 2026-07-14

## Context

CUR-004 asks whether PydanticAI can become the typed planner boundary without
losing the runtime guarantees already implemented for Copilot. The current
provider path is not a plain model call: `InstrumentedAIProvider` performs
admission checks, assigns request IDs, attaches an observer to each provider
attempt, persists `ai_conversations`, records task execution attempts, handles
cancellation, and accounts for provider calls. `CopilotSDKAgenticProvider`
also supplies the bounded internal MCP tool allowlist and returns an
`AgenticRunResult`.

PydanticAI's current APIs provide useful planner primitives: `Agent` accepts a
typed `output_type`, `run()` returns a typed `AgentRunResult`, `Agent.iter()`
and `run_stream_events()` expose execution events, `all_messages()` exposes a
transcript, `RunUsage` exposes request/token usage, and `UsageLimits` can cap
requests and tool calls. These APIs do not automatically understand this
repository's admission ledger, attempt rows, cancellation finalization,
Copilot session lifecycle, or premium-request accounting. A custom PydanticAI
model would therefore be a second provider integration, not a transparent
replacement for the existing wrapper.

## Decision

**Wrap the existing instrumented Copilot provider for the first PydanticAI
planner integration. Defer a native PydanticAI `Model` implementation.**

The adapter must be planner-only and have this contract:

```text
plan(
    context_packet: CurationContextPacket,
    *,
    run_id: str,
    cancellation: CancellationScope,
) -> PlannerExecutionEnvelope[CurationPlan]
```

The implementation will:

1. receive an already bounded, disclosure-checked context packet;
2. call the existing `InstrumentedAIProvider` (or its agentic wrapper), never
   the raw Copilot SDK, so admission decisions and skipped calls remain
   authoritative;
3. preserve the provider request ID, attempt number, start/heartbeat/finish
   events, cancellation status, failure classification, retry delay, raw
   transcript, and parsed response in the existing usage/conversation stores;
4. keep the Copilot MCP server's explicit `allowed_tool_names` allowlist as the
   tool boundary. The planner cannot receive mutation, work-item lifecycle, or
   task-completion tools;
5. validate the provider's JSON result into `CurationPlan` and return the
   typed plan plus an envelope containing provider metadata, usage, transcript
   reference, and validation outcome. Provider prose or claimed action counts
   are not execution evidence;
6. propagate `asyncio.CancelledError` unchanged after finalizing the running
   conversation and attempt as `cancelled/provider_cancelled`, matching the
   current wrapper; and
7. treat each admitted Copilot model call as one premium request using the
   existing provider accounting. PydanticAI `RunUsage.requests` and token
   counts may be recorded as supplemental telemetry, but must not replace the
   Copilot accounting until Copilot supplies a documented premium-unit field.

PydanticAI `UsageLimits` may be used as an additional local guard when the
adapter owns tool orchestration, but it is not the admission boundary and its
request count must not be counted a second time.

## Capability assessment

| Requirement | Result with the wrapper contract | Reason |
| --- | --- | --- |
| Admission events/skips | **Preserved** | Admission stays in `InstrumentedAIProvider` before the SDK call. |
| Cancellation | **Preserved with a required test** | The adapter propagates task cancellation and uses the existing finalization path; the SDK session/task must be verified not to outlive the request. |
| Attempt telemetry | **Preserved** | Existing observer events remain the source for attempt start, heartbeat, and finish rows. |
| Transcript capture | **Preserved** | Existing `ai_conversations` capture remains authoritative; PydanticAI messages are optional supplemental data. |
| Premium-request accounting | **Preserved** | Existing call rows define premium usage; PydanticAI usage is not substituted for them. |
| Bounded tool use | **Preserved** | Copilot's explicit MCP tool allowlist remains in force and the adapter exposes no mutation tools. |
| Typed results | **Supported** | Validate the JSON payload with the curation Pydantic model; PydanticAI's `output_type` can be used later without changing the envelope. |

## Blocker to a native PydanticAI model

Do not implement a custom Copilot `Model` until a spike proves all of the
following against the pinned SDK: cancellation closes the Copilot session and
does not leave a shielded `send_and_wait` task running; every model/tool turn
can be mapped to the existing attempt and conversation records; and the SDK
exposes a stable premium-request identifier or an explicitly documented rule
for deriving one. PydanticAI's `RunUsage` is not evidence of Copilot premium
units, and its event/transcript APIs do not replace the repository's observer
callbacks.

## Consequences

- CUR-057 can proceed later behind the provider-neutral planner protocol
  without changing production routing now.
- The first implementation has one authoritative admission, cancellation,
  telemetry, transcript, and billing path instead of two competing paths.
- PydanticAI remains useful for typed validation and future bounded tool
  orchestration, but a native model adapter is explicitly deferred until the
  Copilot lifecycle and premium accounting contract are proven.
- No dependency or production code change is part of CUR-004.

## Verification evidence

- Inspected `CopilotSDKProvider`, `CopilotSDKAgenticProvider`, provider
  observer events, `InstrumentedAIProvider`, provider usage/conversation
  repositories, and the provider admission policy.
- Compared the current PydanticAI Agent, `Agent.iter()`, streaming-events,
  `RunUsage`, `UsageLimits`, typed output, and tool APIs with the official
  documentation: <https://ai.pydantic.dev/agents/> and
  <https://ai.pydantic.dev/tools/>.
- No live provider spike was run; this ADR adds no dependency and makes no
  production change. A native-model spike is the explicit prerequisite above.
