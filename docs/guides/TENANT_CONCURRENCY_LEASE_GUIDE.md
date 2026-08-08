# Tenant-Concurrency-Lease Routing Guide

![Tenant concurrency lease flow](../../assets/tenant-concurrency-lease.gif)

Use the `tenant-concurrency-lease` strategy when each tenant should receive a
fair share of provider concurrency instead of a single noisy tenant saturating
a backend.

## When to use it

- Multi-tenant gateways need per-tenant in-flight leases
  (`NEXUS_TENANT_CONCURRENCY_LEASE`, default `8`).
- GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic should keep
  quality-first selection while respecting tenant headroom.
- You want LiteLLM/Portkey-style tenant fairness using live `InflightStats`.

## How it works

1. Resolve tenant key from `metadata.tenant_id` (then `user_id` /
   `sticky_key` / `session_id`).
2. Filter domain-eligible catalog candidates.
3. Skip providers whose tenant-scoped in-flight load is at or above the lease.
4. Prefer highest quality under the lease; if every provider is saturated,
   fall back to the least-loaded tenant/provider pair.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=tenant-concurrency-lease
export NEXUS_TENANT_CONCURRENCY_LEASE=8
```

Or per request:

```http
X-Router-Strategy: tenant-concurrency-lease
```

## Demo

See the offline routing walkthrough in [`assets/demo.gif`](../../assets/demo.gif).

## Suggested repo metadata

- **Description:** Multi-provider LLM router with pluggable strategies, safety
  guardrails, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 catalog.
- **Topics:** `llm-router`, `litellm`, `openai`, `anthropic`, `gemini`, `routing`, `python`
