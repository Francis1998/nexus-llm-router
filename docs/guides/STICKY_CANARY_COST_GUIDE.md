# Sticky-Canary-Cost Routing Guide

![Sticky canary cost flow](../../assets/sticky-canary-cost.gif)

Use the `sticky-canary-cost` strategy when tenants need sticky model affinity
with a small canary slice that blends toward cheaper healthy models.

## When to use it

- GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic needs
  per-tenant stickiness for billing, quotas, and prompt-cache warmth.
- Operators also want a controlled cost canary
  (`NEXUS_STICKY_CANARY_COST_PERCENT`, default `10`) that samples cheaper SKUs
  without abandoning sticky affinity for the majority of requests.
- Distinct from `sticky-tenant-hash` (no cost canary) and `canary-cost-blend`
  (no tenant stickiness).

## How it works

1. Resolve a tenant sticky key from `metadata.tenant_id` (then `user_id` /
   `sticky_key` / `session_id` fallbacks).
2. Consistent-hash the sticky key onto a domain-eligible primary model.
3. Off-slice traffic sticks to that primary with healthy ring failover.
4. On the deterministic `request_id` explore slice, blend toward the cheapest
   healthy model that is strictly cheaper than the sticky primary. If none
   exists, keep the sticky primary.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=sticky-canary-cost
export NEXUS_STICKY_CANARY_COST_PERCENT=10
```

Or per request:

```http
X-Router-Strategy: sticky-canary-cost
```

Optional tenant metadata:

```json
{ "metadata": { "tenant_id": "acme" } }
```

## Tuning notes

- Lower the percent (for example `5`) when sticky affinity should dominate.
- Raise the percent (for example `25`) when you want faster cost sampling.
- Pair with `canary-cost-blend` for pure cost exploration without stickiness,
  or `budget-aware` for hard per-request ceilings.

## Suggested repo metadata

- **Description:** Multi-provider LLM router with pluggable strategies, safety
  guardrails, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 catalog.
- **Topics:** `llm-router`, `litellm`, `openai`, `anthropic`, `gemini`, `routing`, `python`
