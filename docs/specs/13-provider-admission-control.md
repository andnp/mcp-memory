# Architecture Decision Record: Provider Admission Control

**Status:** Accepted incremental direction
**Date:** 2026-03-20

## 1. Context

Provider routing in `mcp-memory` currently combines three related but distinct concerns:

- provider execution
- attempt telemetry
- admission control for rate limits and temporary unavailability

This coupling has been workable for the initial provider-routing rollout, but it makes later policy work harder than necessary. Daily profile budgets, model burst limits, and failover eligibility are all part of one logical admission decision, yet they have historically surfaced through wrapper-local booleans and exception shape.

That design has three costs:

1. routing code receives only a coarse `budget_available()` signal instead of a structured reason
2. provider wrappers own policy that should be portable across routing and dashboard surfaces
3. retry/failover behavior is harder to explain because policy outcomes are not represented directly

## 2. Decision

Introduce a dedicated provider-admission layer.

The first slice is intentionally small:

- create a `core/provider_admission.py` module that evaluates rate-limit admission decisions
- represent the result as a structured `ProviderAdmissionDecision`
- let provider wrappers expose `admission_decision()` in addition to the compatibility `budget_available()` path
- update routing to prefer structured admission decisions when available

This ADR does **not** require a full orchestration rewrite. Provider wrappers may still record usage, and route execution may still use the existing same-run failover behavior. The goal of this slice is to establish a stable policy seam.

## 3. Current Scope of Admission Control

The admission layer is currently responsible for:

- profile-scoped daily call budgets
- model-scoped burst limits

The decision surface is designed to expand later to include:

- upstream backoff windows
- provider health degradation
- operator disablement
- richer routing-policy reasons for dashboard/operator reporting

## 4. Consequences

### Positive

- routing can reason about provider availability without encoding policy details inline
- dashboards and logs can surface clearer denial reasons later without another architectural pivot
- the codebase gains a clean place to answer the question “why was this provider unavailable?”
- future work can decide retry-accounting semantics in one policy-oriented subsystem rather than across wrappers and task handlers

### Negative

- one more module exists in a runtime area that is already moderately fragmented
- provider wrappers still participate in policy enforcement for now because the broader execution pipeline has not yet moved to a full admission-controller service

## 5. Non-Goals for This Slice

This ADR does **not** yet:

- introduce a central long-lived admission service object
- rewrite same-run failover into a new executor abstraction
- redefine attempt-accounting semantics for internal subprocess retries
- replace all compatibility use of `budget_available()` at once

## 6. Follow-On Direction

If the architecture continues in this direction, the next good slices are:

1. make route planning consume structured admission outcomes everywhere
2. move more unavailability reasons into the admission layer
3. surface admission-denial taxonomy in dashboard/provider telemetry
4. separate attempt-ledger semantics from wrapper-local execution details