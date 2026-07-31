# Soft-Family-Budget Routing Guide

Use the `soft-family-budget` strategy when OpenRouter/LiteLLM-style multi-provider
traffic needs a **rolling soft spend budget per provider family** so hot families
are deprioritized before hard caps trip, while routing stays deterministic.

## When to use it

- Finance tracks rolling hourly/daily spend per provider family.
- You want GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic to drift
  away from a family that is running hot without blocking it entirely.
- `provider-family-cost-ceiling` is too rigid: you need observed spend, not just
  per-request estimates.
- You need a fail-soft path: when every family is over budget, Nexus still routes
  to the cheapest other family.

## How it works

1. Filter domain-eligible catalog candidates.
2. Read each family's rolling observed spend from shared `FamilySpendWindow`
   (recorded by the engine after successful completions).
3. Among families **under** `NEXUS_SOFT_FAMILY_BUDGET_USD`, pick the
   highest-quality eligible model (ties break toward lower estimated cost).
4. When every family is over the soft budget, deprioritize the hottest family and
   fall back to the cheapest eligible model from another family.

The rolling window length defaults to one hour via
`NEXUS_SOFT_FAMILY_BUDGET_WINDOW_SECONDS`.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=soft-family-budget
export NEXUS_SOFT_FAMILY_BUDGET_USD=5.0
export NEXUS_SOFT_FAMILY_BUDGET_WINDOW_SECONDS=3600
```

Or per request:

```http
X-Router-Strategy: soft-family-budget
```

## Demo

See the offline routing walkthrough in [`assets/demo.gif`](../../assets/demo.gif).
