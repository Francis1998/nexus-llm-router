# Sticky-Tenant-Hash Routing Guide

Use the `sticky-tenant-hash` strategy when LiteLLM/OpenRouter-style traffic needs
**per-tenant routing affinity** rather than per-conversation session stickiness.

## When to use it

- B2B tenants (GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2) should stay
  on a stable primary model for billing, quotas, and provider prompt-cache hits.
- `sticky-session` is too narrow — it hashes only `session_id` for multi-turn
  chats, while tenants may open many sessions.
- You need deterministic failover when a tenant's sticky primary provider is
  unhealthy.

## How it works

1. Resolve a tenant sticky key in order: `metadata.tenant_id`,
   `metadata.user_id`, `metadata.sticky_key`, top-level `user_id`, then
   `session_id`.
2. Filter domain-eligible catalog candidates and sort deterministically by model
   name.
3. Hash the sticky key onto a primary bucket (same `sha256` scheme as
   `sticky-session`).
4. Walk the ring starting at the primary bucket and pick the first **healthy**
   provider.
5. When every provider is unhealthy, return the primary sticky model anyway.

### Difference from `sticky-session`

| Strategy | Sticky key | Use case |
|----------|------------|----------|
| `sticky-session` | `session_id` only | Multi-turn conversation affinity |
| `sticky-tenant-hash` | `metadata.tenant_id` (with fallbacks) | Per-customer / per-tenant affinity |

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=sticky-tenant-hash
```

Per-request metadata:

```json
{
  "metadata": { "tenant_id": "acme-corp" }
}
```

Or via header strategy override:

```http
X-Router-Strategy: sticky-tenant-hash
```

## Demo

See the offline routing walkthrough in [`assets/demo.gif`](../../assets/demo.gif).
