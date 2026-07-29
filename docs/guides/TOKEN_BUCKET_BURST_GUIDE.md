# Token-Bucket-Burst Routing Guide

Use the `token-bucket-burst` strategy when provider gateways expose bursty
token-bucket quotas and you want Nexus to steer GPT-5.5, Claude Sonnet 4.6,
Gemini 3.x, and Kimi K2 traffic toward providers that still have burst capacity.

## When to use it

- LiteLLM, Portkey, or OpenRouter-style provider pools refill quota over time and
  should accept short bursts without immediately hard-blocking.
- You want quality-first routing among providers that still have at least one
  token available in their local bucket.
- Every provider is momentarily empty and you still need deterministic fallback:
  pick the highest remaining token fraction, then the cheapest eligible model.

## How it works

1. Filter to domain-eligible catalog candidates.
2. Refill each provider's shared token bucket at `NEXUS_TOKEN_BUCKET_REFILL_PER_SEC`
   up to `NEXUS_TOKEN_BUCKET_CAPACITY`.
3. Prefer candidates whose provider has at least one available token.
4. Break ties by higher `quality_score`, then lower estimated request cost.
5. When every bucket is empty, fall back to the highest remaining token fraction,
   then to the cheapest eligible model.
6. Consume one token from the chosen provider's bucket on every decision.

Fallback chains attempt providers with available burst tokens first, then order by
remaining fraction, quality, and cost.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=token-bucket-burst
export NEXUS_TOKEN_BUCKET_CAPACITY=10
export NEXUS_TOKEN_BUCKET_REFILL_PER_SEC=1.0
```

Or per request:

```http
X-Router-Strategy: token-bucket-burst
```

`NEXUS_TOKEN_BUCKET_CAPACITY` is the per-provider burst ceiling (minimum `1`).
`NEXUS_TOKEN_BUCKET_REFILL_PER_SEC` is how many tokens refill each second and must
be positive.

## Demo

![Token-bucket-burst routing demo](../../assets/token-bucket-burst.gif)
