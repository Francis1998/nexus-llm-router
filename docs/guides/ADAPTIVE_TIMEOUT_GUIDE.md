# Adaptive-Timeout Routing Guide

Use the `adaptive-timeout` strategy when you want timeout-aware routing that adapts to request urgency, recent provider p95 latency, and provider error signals.

## When to use it

- Realtime traffic needs faster models when recent provider latency is high or the deadline is tight.
- Batch traffic can accept slower, higher-quality models when the adaptive budget has enough headroom.
- You want GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, and Kimi K2 routing to react to both p95 latency and recent failures without a hard fail at decide time.
- Cold-start traffic should still admit every provider until observations accrue.

## How it works

1. Filter to domain-eligible catalog candidates; realtime requests prefer realtime-capable candidates when available.
2. Use `NEXUS_LATENCY_SLA_MS` as the realtime base timeout budget; batch requests receive a wider derived budget.
3. Read each provider's rolling p95 from `LatencyStats` (no observations -> `0.0ms`).
4. Tighten the budget around the fastest observed provider when recent p95 latency shows pressure.
5. Read each provider's rolling success rate from `SuccessStats` (no observations -> `1.0`) and inflate p95 into a risk-adjusted latency when recent errors are present.
6. Pick the highest-quality candidate whose risk-adjusted latency fits the adaptive timeout budget.
7. If none fit, fall back to the lowest risk-adjusted latency candidate; the engine still uses the normal fallback chain.

The router records latency on completed provider calls and records success/failure on provider outcomes, so the budget adapts from live traffic.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=adaptive-timeout
export NEXUS_LATENCY_SLA_MS=750
```

Or per request:

```http
X-Router-Strategy: adaptive-timeout
```

## Demo

See the offline routing walkthrough in [`assets/demo.gif`](../../assets/demo.gif).
