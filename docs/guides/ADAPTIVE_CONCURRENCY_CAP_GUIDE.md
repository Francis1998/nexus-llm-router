# Adaptive-Concurrency-Cap Routing Guide

Use `adaptive-concurrency-cap` when per-provider in-flight limits should shrink
for unhealthy or slow backends so GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 traffic drains toward healthier providers under load.

![Adaptive concurrency cap demo](../../assets/adaptive-concurrency-cap.gif)

## When to use it

- Static `concurrency-cap` is too coarse when one provider is error-prone or slow.
- You want quality-first routing among providers that still have adaptive headroom.
- Rolling success rate and p95 latency should tighten caps automatically.

## How it works

1. Compute each provider's effective cap:
   `max(min_cap, int(base_cap * success_rate * min(1, latency_ms / p95)))`.
2. Skip providers whose live in-flight count is at or above the effective cap.
3. Prefer the highest health score (success rate × latency factor), then quality.
4. When every provider is saturated, fall back to the least load/cap ratio.

Unlike `concurrency-cap`, caps are dynamic per provider. Unlike `least-busy`,
providers above their adaptive cap are excluded from primary selection.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=adaptive-concurrency-cap
export NEXUS_ADAPTIVE_CONCURRENCY_BASE_CAP=8
export NEXUS_ADAPTIVE_CONCURRENCY_MIN_CAP=1
export NEXUS_ADAPTIVE_CONCURRENCY_LATENCY_MS=2000.0
```

Or per request:

```http
X-Router-Strategy: adaptive-concurrency-cap
```
