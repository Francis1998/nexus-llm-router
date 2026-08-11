# Token-Bucket-Tenant Routing Guide

Use `token-bucket-tenant` when every tenant should receive an independent burst
budget and over-budget requests should remain available on the cheapest eligible
model.

## When to use it

- GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic is shared by
  multiple customers with different request bursts.
- One noisy tenant must not consume another tenant's local routing budget.
- Over-budget traffic should be cost-shed instead of rejected.

## How it works

1. Resolve tenant identity from `metadata.tenant_id`, then metadata user/sticky
   keys, top-level `user_id`, and finally `session_id`.
2. Maintain one local token bucket per resolved tenant.
3. Refill at `NEXUS_TOKEN_BUCKET_TENANT_RATE` quota tokens per second with a
   one-second burst capacity (minimum one request).
4. Consume one quota token per routing decision.
5. While budget remains, select the highest-quality domain-eligible model.
6. When the bucket is empty, shed to the cheapest domain-eligible model and
   order fallbacks by cost.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=token-bucket-tenant
export NEXUS_TOKEN_BUCKET_TENANT_RATE=5.0
```

Or per request:

```http
X-Router-Strategy: token-bucket-tenant
```

Include a stable tenant identifier:

```json
{
  "metadata": {"tenant_id": "acme"},
  "messages": [{"role": "user", "content": "Summarize this report."}]
}
```

## How it differs

`token-bucket-burst` keeps provider-side buckets and moves traffic toward
providers with burst quota. `token-bucket-tenant` keeps customer-scoped buckets
and changes model economics after a tenant exhausts its budget.

Buckets are local to each router process. Use a shared rate-limit store when
tenant budgets must be enforced globally across replicas.
