# Provider-Warmup-Weight Routing Guide

Use `provider-warmup-weight` to bias selection toward providers that are
already warm — according to a per-request signal — while staying
quality-aware for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

![Provider warmup weight demo](../../assets/provider-warmup-weight.gif)

## When to use it

- Fleets that want to keep sending traffic to providers that an external
  system (a load balancer, a canary controller, a warm-pool manager) has
  already flagged as warm, instead of always defaulting to the single
  highest-quality model.
- Gateways recovering from a cold start or failover where re-warming a
  recovered provider gradually, rather than instantly, is preferable.
- Workloads that want a request-level hint at warmup state, computed
  upstream, without requiring the router to maintain its own rolling stats.

## How it works

1. Filter domain-eligible candidates through provider circuit health.
2. Resolve each candidate provider's warmup score in `[0.0, 1.0]` directly
   from the request: `metadata.provider_warmup` is a mapping of provider
   name to a score. A provider missing from the mapping — or a request that
   omits the mapping entirely — defaults to a neutral `0.5`.
3. Blend quality and warmup: `(1 - blend) * quality + blend * warmup`, where
   `blend` is `NEXUS_PROVIDER_WARMUP_BLEND` (default `0.3`).
4. Select the candidate with the highest blended score, using cost as a
   tie-break.
5. Out-of-range overrides are clamped to `[0.0, 1.0]`; malformed
   (non-numeric) overrides fall back to the neutral `0.5` default.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=provider-warmup-weight
export NEXUS_PROVIDER_WARMUP_BLEND=0.3
```

Or select it per request:

```http
X-Router-Strategy: provider-warmup-weight
```

```json
{
  "metadata": {
    "provider_warmup": {
      "openai": 0.9,
      "anthropic": 0.4
    }
  }
}
```

Providers omitted from `provider_warmup` (here, any provider besides
`openai` and `anthropic`) are treated as a neutral `0.5`.
