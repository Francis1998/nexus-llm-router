# Tenant-Priority-Lanes Routing Guide

Use `tenant-priority-lanes` to map tenants into high, normal, or low service
lanes. Under provider health pressure or recent lane-quota pressure, high tenants
prefer the fastest observed healthy route, normal tenants preserve quality, and
low tenants move to a cost-efficient relief route for GPT-5.5 / Claude Sonnet
4.6 / Gemini 3.x / Kimi K2 traffic.

## When to use it

- Critical interactive tenants need the fastest healthy capacity during an
  outage.
- Standard tenants should retain quality-first routing.
- Batch or best-effort tenants may shift to cheaper capacity under pressure.

## How it works

1. Resolve tenant identity from metadata, `user_id`, or `session_id`.
2. Honor optional `metadata.priority_lane`; otherwise map configured high and
   low tenant ids, with every other tenant assigned to normal.
3. Filter domain-eligible candidates through provider circuit health.
4. Treat missing healthy providers or a lane reaching its soft recent quota as
   capacity pressure.
5. Under pressure, high uses the lowest observed provider p95, normal remains
   quality-first, and low minimizes estimated request cost.
6. Record the lane in a bounded process-local window.

These are routing preferences, not queue preemption: requests already in flight
are not interrupted. Unknown per-request lane values safely fall back to the
configured tenant mapping.

## Quick start

Pydantic list settings use JSON syntax:

```bash
export NEXUS_DEFAULT_STRATEGY=tenant-priority-lanes
export NEXUS_TENANT_PRIORITY_HIGH_TENANTS='["interactive-premium"]'
export NEXUS_TENANT_PRIORITY_LOW_TENANTS='["offline-batch"]'
export NEXUS_TENANT_PRIORITY_LANE_LOOKBACK=100
export NEXUS_TENANT_PRIORITY_HIGH_QUOTA=100
export NEXUS_TENANT_PRIORITY_NORMAL_QUOTA=60
export NEXUS_TENANT_PRIORITY_LOW_QUOTA=30
```

Or select it per request:

```http
X-Router-Strategy: tenant-priority-lanes
```

```json
{"metadata": {"tenant_id": "acme", "priority_lane": "high"}}
```
