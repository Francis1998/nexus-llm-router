# Provider-Hourly-Cost-Ceiling Routing Guide

Use `provider-hourly-cost-ceiling` when GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2 spend must stay under a rolling hourly estimated budget
per provider — distinct from per-request `provider-family-cost-ceiling`.

![Demo](../../assets/provider-hourly-cost-ceiling.gif)

## When to use it

- Providers have hourly spend caps you want enforced at decide time.
- Per-request family ceilings are too coarse for bursty multi-model traffic.
- Over-ceiling providers should be skipped while under-ceiling ones stay quality-first.

## How it works

1. Filter the catalog to domain-eligible models.
2. Skip providers whose rolling hourly estimated spend exceeds the ceiling.
3. Among under-ceiling providers, pick highest quality (cost tie-break).
4. When every provider is over ceiling, fall back to the cheapest eligible model.
5. The engine records spend into the shared hourly window on successful responses.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=provider-hourly-cost-ceiling
export NEXUS_PROVIDER_HOURLY_COST_CEILING_USD=5.0
```

Or per request:

```http
X-Router-Strategy: provider-hourly-cost-ceiling
```

## Tuning notes

- Lower the ceiling for tighter hourly spend control.
- Raise it when providers share bursty traffic and short overages are acceptable.
- Spend windows are local to each router process unless shared externally.
