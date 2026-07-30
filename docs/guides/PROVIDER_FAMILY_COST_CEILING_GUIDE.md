# Provider-Family Cost-Ceiling Routing Guide

Use the `provider-family-cost-ceiling` strategy when OpenRouter/LiteLLM-style
multi-provider traffic needs a hard estimated-cost ceiling **per provider
family** (openai / anthropic / google / moonshot) so spend stays within family
budgets while still maximizing quality.

## When to use it

- Finance assigns separate budgets to OpenAI, Anthropic, Google, and Moonshot.
- You want the highest-quality domain-eligible model among GPT-5.5, Claude Sonnet
  4.6, Gemini 3.x, and Kimi K2 that still fits its family's ceiling.
- Global `budget-aware` is too coarse: one family should be tightened without
  starving the others.
- You need a fail-soft path: if nothing fits any family ceiling, Nexus still
  routes to the cheapest eligible model across families.

## How it works

1. Filter domain-eligible catalog candidates (relax to the full catalog if none
   match).
2. Estimate per-request cost from `prompt_tokens_estimate` and `max_tokens`.
3. Look up each candidate's family ceiling (per-family override, else
   `NEXUS_PROVIDER_FAMILY_COST_CEILING_USD`).
4. Keep candidates whose estimate is within their family ceiling.
5. Among those, pick the highest `quality_score` (ties break toward lower cost).
6. If no family has an affordable eligible model, fall back across families to
   the cheapest eligible candidate.

Catalog priors cover GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 under
the openai, anthropic, google, and moonshot families.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=provider-family-cost-ceiling
export NEXUS_PROVIDER_FAMILY_COST_CEILING_USD=0.05
```

Or per request:

```http
X-Router-Strategy: provider-family-cost-ceiling
```

`NEXUS_PROVIDER_FAMILY_COST_CEILING_USD` is the default ceiling applied to every
provider family (non-negative USD). Constructor-level `family_ceilings_usd`
overrides can tighten or relax one family without changing the default.

## Demo

See the offline routing walkthrough in [`assets/demo.gif`](../../assets/demo.gif).
