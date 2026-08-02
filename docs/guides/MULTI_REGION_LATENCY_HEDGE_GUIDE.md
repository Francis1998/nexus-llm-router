# Multi-Region-Latency-Hedge Routing Guide

Use the `multi-region-latency-hedge` strategy when LiteLLM/OpenRouter-style traffic
needs **primary-region quality preference** with an automatic latency hedge into a
secondary region during hot-spot events.

## When to use it

- GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic should stay in a
  preferred residency region (`request.region`) for compliance and cache warmth.
- Occasional primary-region latency spikes should fail over to a faster secondary
  region without abandoning quality-first routing during normal conditions.
- `latency-slo-shed` sheds individual slow providers; this strategy hedges across
  **regions** when the primary pool's p50 crosses `NEXUS_LATENCY_HEDGE_MS`.

## How it works

1. Filter domain-eligible catalog candidates.
2. Treat `request.region` (default `global`) as the **primary** region.
3. Pick the highest-quality primary-region candidate.
4. Read that provider's rolling **p50** from shared `LatencyStats`.
5. When p50 exceeds `NEXUS_LATENCY_HEDGE_MS` (default `500`) **and** at least one
   secondary-region candidate exists, hedge to the lowest-p50 secondary model.
6. Otherwise stay on the primary quality preference.

Providers with no observations yet report `0.0` p50 and keep primary routing.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=multi-region-latency-hedge
export NEXUS_LATENCY_HEDGE_MS=500
```

Or per request:

```http
X-Router-Strategy: multi-region-latency-hedge
```

Include a region affinity on the request body when possible.

## Demo

See the offline routing walkthrough in [`assets/demo.gif`](../../assets/demo.gif).
