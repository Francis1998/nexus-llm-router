# Health/Cost/Latency Routing Guide

Use the `health-cost-latency` strategy when you want a LiteLLM- or Portkey-style
ternary blend across provider reliability, spend, and speed without a separate
quality axis or circuit-breaker hard gate.

## When to use it

- You want rolling provider success rate, estimated request cost, and p95
  latency in one tunable score.
- GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, and Kimi K2 should trade off health,
  cost, and latency explicitly.
- Cold-start traffic should remain safe: missing success stats count as `1.0`,
  and missing p95 latency stats are normalized without divide-by-zero failures.

## How it works

1. Filter to domain-eligible catalog candidates, or the whole catalog if no model
   advertises the domain.
2. Score each candidate with normalized components:
   - provider success rate from `SuccessStats` (`1.0` when no data exists);
   - inverse normalized estimated request cost;
   - inverse normalized rolling provider p95 latency from `LatencyStats`.
3. Select the highest weighted blend and order fallbacks by score, then health,
   cost, and latency.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=health-cost-latency
export NEXUS_HCL_HEALTH_WEIGHT=0.4
export NEXUS_HCL_COST_WEIGHT=0.3
export NEXUS_HCL_LATENCY_WEIGHT=0.3
```

Or per request:

```http
X-Router-Strategy: health-cost-latency
```

Weights are normalized to sum to one, so only ratios matter. All-zero weights
fall back to pure health (success rate).

## Demo

![Health/cost/latency routing demo](../../assets/health-cost-latency.gif)
