# Soft-Rate-Limit Routing Guide

Use the `soft-rate-limit` strategy when provider gateways are returning 429s or
rate-limit messages before a provider is unhealthy enough to trip the circuit
breaker.

## When to use it

- LiteLLM, Portkey, OpenRouter, or native provider traffic is close to quota and
  should move away from providers with fresh 429 pressure.
- You want GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, and Kimi K2 traffic to remain
  quality-first while backing off providers that recently said "too many
  requests".
- You prefer a soft signal over a hard outage rule: providers with recent rate
  limits stay in the fallback chain and can recover as successful attempts age
  out older 429 observations.

## How it works

1. Filter to domain-eligible catalog candidates.
2. Prefer candidates whose provider circuit is currently healthy.
3. Read each provider's bounded `RateLimitStats` window.
4. Order candidates by fewer recent rate-limit hits, then lower rate-limit rate.
5. Break ties by higher `quality_score`, lower estimated request cost, and model
   name.

The engine records a positive rate-limit observation when a provider exception
looks like HTTP 429, `rate limit`, `rate_limit`, `rate-limit`, or `too many
requests`. Successful provider attempts record a non-rate-limit observation, so
old pressure naturally ages out of the rolling window.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=soft-rate-limit
```

No additional `NEXUS_*` setting is required.

Or per request:

```http
X-Router-Strategy: soft-rate-limit
```

## Demo

![Soft-rate-limit routing demo](../../assets/soft-rate-limit.gif)
