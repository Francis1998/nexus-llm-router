# Tenant-Quota-Burst Routing Guide

Use `tenant-quota-burst` when each tenant needs a steady request quota plus a
small, controlled burst band for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 traffic.

## When to use it

- Normal traffic should retain quality-first model selection.
- Short demand spikes should be admitted without immediately rejecting tenants.
- Burst traffic may use cheaper capacity, but a hard request ceiling must stop
  unbounded overload.

## How it works

1. Resolve tenant identity from `metadata.tenant_id`, user, then session.
2. Count admitted requests in a rolling
   `NEXUS_TENANT_QUOTA_BURST_WINDOW_SECONDS` window.
3. Through `NEXUS_TENANT_QUOTA_BURST_SOFT`, select the highest-quality
   domain-compatible model.
4. Above the soft quota, admit requests through
   `NEXUS_TENANT_QUOTA_BURST_HARD` and shed them to the cheapest compatible
   fallback.
5. At the hard quota, raise `TenantQuotaBurstExceededError` before dispatch.

Rejected requests do not consume an additional slot. Tenant windows are
isolated, and old slots become available as their timestamps leave the rolling
window. The in-memory state is process-local.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=tenant-quota-burst
export NEXUS_TENANT_QUOTA_BURST_SOFT=60
export NEXUS_TENANT_QUOTA_BURST_HARD=75
export NEXUS_TENANT_QUOTA_BURST_WINDOW_SECONDS=60.0
```

Or per request:

```http
X-Router-Strategy: tenant-quota-burst
```

This combines Envoy-style burst headroom with a LiteLLM-style rolling budget
window: the soft quota represents steady capacity, while the hard quota bounds
the temporary burst.
