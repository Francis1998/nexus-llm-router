# Tenant-Soft-Isolation Routing Guide

Use `tenant-soft-isolation` to soft-isolate noisy tenants onto spare,
lower-cost capacity instead of frontier models for GPT-5.5 / Claude Sonnet
4.6 / Gemini 3.x / Kimi K2, without ever rejecting a request.

## When to use it

- Multi-tenant gateways that want fair-use protection (Portkey/Helicone
  style) so one noisy tenant cannot monopolize frontier-quality routing.
- Fleets that prefer graceful demotion over hard request rejection when a
  tenant bursts past a soft rate.
- Deployments that already key requests by `metadata.tenant_id` (or fall
  back to `user_id` / `session_id`).

## How it works

1. Resolve the tenant key from `metadata.tenant_id`, then
   `metadata.user_id`, `metadata.sticky_key`, the top-level `user_id`, and
   finally `session_id`.
2. Track each tenant's rolling request rate in `TenantSoftIsolationStats`
   over a 60-second window.
3. While the tenant's rate stays at or below
   `NEXUS_TENANT_SOFT_ISOLATION_RPM` (default `60`), route quality-first
   among healthy domain-eligible candidates.
4. Once the tenant's rate exceeds the soft ceiling, route to the cheapest
   healthy domain-compatible model — spare capacity — instead of rejecting
   the request.
5. Provider circuit health is always respected; each tenant is tracked
   independently so one noisy tenant does not affect another.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=tenant-soft-isolation
export NEXUS_TENANT_SOFT_ISOLATION_RPM=60
```

Or select it per request:

```http
X-Router-Strategy: tenant-soft-isolation
```

```json
{
  "metadata": {
    "tenant_id": "acme-corp"
  }
}
```
