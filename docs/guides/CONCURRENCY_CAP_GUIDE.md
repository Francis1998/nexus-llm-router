# Concurrency-Cap Routing Guide

Use the `concurrency-cap` strategy when you want quality-first routing that
stops sending fresh primary traffic to providers that have already reached a
configured live in-flight limit.

## When to use it

- A provider has strict parallel-request quotas and should be skipped before it
  hits a hard provider-side throttle.
- GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, and Kimi K2 traffic should keep
  preferring the highest-quality available model without piling onto an already
  saturated provider.
- You want a deterministic saturation guard that reacts to current router state
  faster than latency or error-rate signals can accumulate.

## How it works

1. Filter to domain-eligible catalog candidates, falling back to the full catalog
   only when no candidate advertises the requested domain.
2. Read each candidate provider's live in-flight count from `InflightStats`.
3. Skip providers whose count is at or above `NEXUS_CONCURRENCY_CAP`.
4. Select the highest-quality remaining candidate; ties prefer lower estimated
   request cost, then model name.
5. Order fallbacks so providers still below the cap are attempted before capped
   providers.
6. If every eligible provider is capped, route to the least-loaded eligible
   provider and record the all-capped fallback in the rationale.

The router increments a provider's in-flight counter immediately before provider
dispatch and decrements it in cleanup after success, failure, or timeout. The
strategy therefore uses live concurrency, not historical latency, to avoid
saturated providers.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=concurrency-cap
export NEXUS_CONCURRENCY_CAP=8
```

Or per request:

```http
X-Router-Strategy: concurrency-cap
```

`NEXUS_CONCURRENCY_CAP` is the maximum live attempts per provider before that
provider is skipped for primary selection. The value must be at least `1`.

## Demo

![Concurrency-cap routing demo](../../assets/concurrency-cap.gif)
