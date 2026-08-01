# Latency-SLO-Shed Routing Guide

Use the `latency-slo-shed` strategy when LiteLLM/OpenRouter-style traffic needs
a **latency service-level objective** with explicit shedding: providers whose
rolling p95 exceeds the SLO are deprioritized whenever faster alternatives exist.

## When to use it

- Peak traffic should not keep routing to slow-but-premium providers when a
  faster model can serve the prompt.
- You want GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 routing to shed
  over-SLO backends instead of always competing on quality.
- `latency-budget` (`NEXUS_LATENCY_SLA_MS`, default `750`) is too tight; you
  need a softer SLO gate (default `2000ms`) focused on shedding, not hard caps.

## How it works

1. Filter domain-eligible catalog candidates.
2. Read each provider's rolling p95 from shared `LatencyStats`.
3. When at least one candidate is **under** `NEXUS_LATENCY_SLO_MS`, shed every
   over-SLO provider and pick the highest-quality remaining model (ties break
   toward lower p95).
4. When **every** provider is over the SLO, fall back to the lowest-p95 eligible
   model so routing stays deterministic.

Providers with no observations yet report `0.0` p95 and are treated as within
the SLO (cold start routes to the best model).

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=latency-slo-shed
export NEXUS_LATENCY_SLO_MS=2000
```

Or per request:

```http
X-Router-Strategy: latency-slo-shed
```

## Demo

See the offline routing walkthrough in [`assets/demo.gif`](../../assets/demo.gif).
