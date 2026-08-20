# Tenant-Fair-Queue Routing Guide

Use `tenant-fair-queue` when noisy tenants should not monopolize premium model
capacity. The strategy derives a deficit from recent per-tenant request counts,
giving quieter tenants quality priority while moving above-share tenants onto a
healthy cost-efficient relief lane for GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2 traffic.

## When to use it

- Multiple tenants share one router process and request rates differ sharply.
- Quieter tenants should retain access to the highest-quality healthy model.
- Busy tenants may use a cheaper healthy route until recent request shares
  rebalance.

## How it works

1. Resolve tenant identity from `metadata.tenant_id`, metadata user/sticky keys,
   `user_id`, or `session_id`.
2. Count each active tenant's requests over the last
   `NEXUS_TENANT_FAIR_QUEUE_LOOKBACK` routing decisions.
3. Compute equal fair share as `recent requests / active tenants`.
4. Tenants at or below fair share receive the highest-quality healthy route.
5. Tenants above fair share use the cheapest healthy relief lane.
6. Record the tenant after the decision so the bounded window continuously
   rebalances.

This differs from `queue-depth-fairness`, which reacts to provider in-flight
depth, and `provider-quota-fair-share`, which balances provider selections. The
window is process-local; use shared state when fairness must span replicas.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=tenant-fair-queue
export NEXUS_TENANT_FAIR_QUEUE_LOOKBACK=100
```

Or select it per request:

```http
X-Router-Strategy: tenant-fair-queue
```
