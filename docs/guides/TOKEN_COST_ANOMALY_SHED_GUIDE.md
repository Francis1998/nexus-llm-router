# Token-Cost-Anomaly-Shed Routing Guide

Use the `token-cost-anomaly-shed` strategy when LiteLLM/OpenRouter-style traffic
needs **spend guardrails** that react to sudden per-token cost spikes without a
hard per-request ceiling.

## When to use it

- Provider SKUs repriced mid-flight and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
  Kimi K2 traffic should shed away from anomalously expensive projections.
- Finance wants a rolling baseline rather than a static `budget-aware` ceiling.
- You already record completion spend and want anomaly shedding on top of
  quality-first routing.

## How it works

1. Filter domain-eligible catalog candidates.
2. Read the rolling mean cost-per-1k-tokens from shared `CostAnomalyStats`
   (populated by the engine after successful completions).
3. Rank by quality and inspect the top candidate's projected cost/1k for the
   current prompt/output token estimate.
4. When projected cost/1k exceeds `mean * NEXUS_TOKEN_COST_ANOMALY_RATIO`
   (default `2.0`), shed to the cheapest **healthy** alternative below that
   cost.
5. When no cheaper healthy option exists, fall back to pure quality ranking.

Cold start (empty stats) routes to the highest-quality eligible model.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=token-cost-anomaly-shed
export NEXUS_TOKEN_COST_ANOMALY_RATIO=2.0
```

Or per request:

```http
X-Router-Strategy: token-cost-anomaly-shed
```

## Demo

See the offline routing walkthrough in [`assets/demo.gif`](../../assets/demo.gif).
