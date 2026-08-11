# Provider-Quota-Fair-Share Routing Guide

Use `provider-quota-fair-share` when recent traffic should be distributed evenly
across eligible providers instead of repeatedly converging on the highest-quality
backend.

## When to use it

- GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic shares provider
  quota pools and one provider should not absorb every request.
- You want a bounded local signal that reacts faster than provider billing or
  rate-limit telemetry.
- Quality should remain the tie-breaker among providers below their fair share.

## How it works

1. Filter the catalog to models eligible for the request domain.
2. Compute equal provider share as `1 / eligible provider count`.
3. Measure each provider's selections over the most recent
   `NEXUS_PROVIDER_QUOTA_LOOKBACK` decisions.
4. Shed providers at or above equal share while an under-share provider exists.
5. Select the highest-quality model in the preferred pool, breaking ties by
   estimated request cost.
6. Record the selected provider and order fallbacks by lower recent share.

At cold start, all shares are zero and the highest-quality eligible model wins.
When shares are exactly balanced, routing also falls back to quality-first
selection.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=provider-quota-fair-share
export NEXUS_PROVIDER_QUOTA_LOOKBACK=100
```

Or select it per request:

```http
X-Router-Strategy: provider-quota-fair-share
```

## Tuning notes

- Lower the lookback to react quickly to changing traffic, at the cost of more
  frequent provider rotation.
- Raise it to smooth short bursts and enforce fairness over a longer horizon.
- The window is local to each router process; use shared external telemetry when
  fairness must be coordinated across replicas.
