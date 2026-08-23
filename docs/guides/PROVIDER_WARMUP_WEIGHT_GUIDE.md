# Provider-Warmup-Weight Routing Guide

Use `provider-warmup-weight` to bias selection toward providers carrying warm
recent successful traffic while staying quality-aware for GPT-5.5 / Claude
Sonnet 4.6 / Gemini 3.x / Kimi K2.

## When to use it

- Fleets that want to keep sending traffic to providers that are already
  "warm" (recently handling successful requests) instead of always defaulting
  to the single highest-quality model.
- Gateways recovering from a cold start or failover where re-warming a
  recovered provider gradually, rather than instantly, is preferable.
- Workloads that want a request-level override to hint at warmup state (for
  example, from an external load balancer) without waiting for local stats.

## How it works

1. Filter domain-eligible candidates through provider circuit health.
2. Resolve each candidate provider's warmup score in `[0.0, 1.0]`:
   - From a per-request `metadata.provider_warmup_score` override (a mapping
     of provider name to score), when present and numeric.
   - Otherwise from a shared `WarmupStats` rolling window of recent
     successful traffic.
3. Blend quality and warmup: `(1 - blend) * quality + blend * warmup`, where
   `blend` is `NEXUS_PROVIDER_WARMUP_BLEND` (default `0.3`).
4. Select the candidate with the highest blended score, using cost as a
   tie-break.
5. Providers with no recorded observations register a neutral warmup score
   of `0.0`, so a true cold start still favors quality until traffic warms a
   provider up.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=provider-warmup-weight
export NEXUS_PROVIDER_WARMUP_BLEND=0.3
```

Or select it per request:

```http
X-Router-Strategy: provider-warmup-weight
```
