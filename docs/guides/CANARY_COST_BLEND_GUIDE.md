# Canary-Cost-Blend Routing Guide

Use the `canary-cost-blend` strategy when LiteLLM/OpenRouter-style traffic needs
**cost-first routing** with a small explore slice that samples the next-cheaper
healthy tier without the complexity-tier coupling of `canary-tier-blend`.

## When to use it

- Production should default to the cheapest healthy provider, but finance wants
  occasional samples of the next cost rung for quality/latency comparison.
- GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 catalogs need deterministic
  cost exploration without progressive SKU rollout (`canary`).
- You want health-gated spend minimization plus replayable explore bucketing.

## How it works

1. Filter domain-eligible catalog candidates.
2. Keep only models whose provider circuit is closed (healthy); when every
   circuit is open the strategy still considers the full eligible pool.
3. Sort healthy candidates by estimated request cost (ties break toward higher
   quality).
4. **Default path:** pick the cheapest healthy model.
5. **Explore slice:** when `sha256(request_id)` falls under
   `NEXUS_CANARY_COST_BLEND_PERCENT` (default `10`), pick the **second-cheapest**
   healthy model (next-cheaper tier). If only one healthy option exists, return
   it with an explicit single-tier note.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=canary-cost-blend
export NEXUS_CANARY_COST_BLEND_PERCENT=10
```

Or per request:

```http
X-Router-Strategy: canary-cost-blend
```

## Demo

See the offline routing walkthrough in [`assets/demo.gif`](../../assets/demo.gif).
