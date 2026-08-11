# Adaptive-Timeout-Hedge Routing Guide

Use `adaptive-timeout-hedge` when quality-first routing should move to a faster
provider only after live latency shows a meaningful relative slowdown.

## When to use it

- GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic should normally
  select the highest-quality domain-eligible model.
- Provider latency differs over time, so a fixed millisecond hedge threshold
  would be too aggressive during generally slow periods or too loose during
  healthy periods.
- You want a provider-level hedge without crossing a region boundary.

## How it works

1. Select the highest-quality domain-eligible model as the top choice.
2. Read positive rolling provider p95 observations from shared `LatencyStats`.
3. Derive the threshold as
   `fastest eligible observed p95 * NEXUS_ADAPTIVE_TIMEOUT_HEDGE_RATIO`.
4. If the top choice exceeds that threshold, select the observed alternative
   with the lowest p95.
5. Otherwise keep the quality leader.

Providers with no observations report `0.0ms` and are not treated as fast hedge
targets. This prevents cold providers from attracting traffic based on missing
data.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=adaptive-timeout-hedge
export NEXUS_ADAPTIVE_TIMEOUT_HEDGE_RATIO=1.5
```

Or per request:

```http
X-Router-Strategy: adaptive-timeout-hedge
```

## How it differs

- `adaptive-timeout` filters candidates against an urgency-aware timeout budget
  and inflates latency using provider error risk.
- `multi-region-latency-hedge` compares primary-region p50 to a fixed
  millisecond threshold and moves only to a secondary region.
- `adaptive-timeout-hedge` compares provider p95 values across the eligible pool
  and derives its threshold from the fastest current observation.

The ratio must be at least `1.0`. Raise it to preserve quality through larger
latency differences; lower it toward `1.0` to hedge more readily.
