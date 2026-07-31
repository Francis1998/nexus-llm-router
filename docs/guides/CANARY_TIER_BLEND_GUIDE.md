# Canary-Tier-Blend Routing Guide

Use the `canary-tier-blend` strategy when progressive canary delivery must respect
**complexity-tier affinity** so frontier validation does not starve mid-tier traffic.

## When to use it

- You are rolling out a new frontier SKU (`NEXUS_CANARY_MODEL`) but want stable
  traffic to stay on tier-appropriate models.
- Canary slices should prefer the canary when it matches the inferred tier.
- Off-slice traffic should follow tier affinity, then quality.
- You need health-gated canary pausing like the standalone `canary` strategy.

## How it works

1. Map `TaskSignals.complexity_score` to a target tier (frontier / mid / economy).
2. Bucket traffic with the same stable `request_id` hash as `canary`.
3. On the canary slice when the canary provider is healthy:
   - Prefer the canary when it matches the target tier.
   - Otherwise still route to the canary.
4. Off the canary slice, or when the canary is unhealthy:
   - Prefer the highest-quality domain-eligible model in the target tier.
   - Fall back to the highest-quality eligible model when no tier match exists.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=canary-tier-blend
export NEXUS_CANARY_STABLE_MODEL=gpt-4.1-mini
export NEXUS_CANARY_MODEL=gpt-5.5
export NEXUS_CANARY_WEIGHT=0.1
```

Or per request:

```http
X-Router-Strategy: canary-tier-blend
```

## Demo

See the offline routing walkthrough in [`assets/demo.gif`](../../assets/demo.gif).
