# Circuit-Breaker-Half-Open-Probe Routing Guide

![Circuit-breaker half-open probe flow](../../assets/circuit-breaker-half-open-probe.gif)

Use the `circuit-breaker-half-open-probe` strategy when recovering providers
should receive **limited probe traffic** instead of a full traffic stampede.

## When to use it

- Provider circuits reopen after a recovery window and you want healthy
  backends to keep the majority of GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
  Kimi K2 traffic.
- Half-open / recovering providers should only accept a small concurrent probe
  budget (`NEXUS_CIRCUIT_HALF_OPEN_PROBE_BUDGET`).
- You need LiteLLM/Portkey-style half-open probe control at decide time.

## How it works

1. Filter domain-eligible catalog candidates.
2. Prefer fully closed (healthy) providers by quality.
3. When no healthy provider remains, measure live in-flight load across
   half-open providers via `InflightStats`.
4. Allow a recovery probe while that load stays under
   `NEXUS_CIRCUIT_HALF_OPEN_PROBE_BUDGET`; otherwise note budget exhaustion and
   still return a deterministic half-open fallback.
5. If nothing is healthy or half-open, fall back to highest-quality eligible.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=circuit-breaker-half-open-probe
export NEXUS_CIRCUIT_HALF_OPEN_PROBE_BUDGET=2
```

Or per request:

```http
X-Router-Strategy: circuit-breaker-half-open-probe
```

## Demo

See the offline routing walkthrough in [`assets/demo.gif`](../../assets/demo.gif).

## Suggested repo metadata

- **Description:** Multi-provider LLM router with pluggable strategies, safety
  guardrails, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 catalog.
- **Topics:** `llm-router`, `litellm`, `openai`, `anthropic`, `gemini`, `routing`, `python`
