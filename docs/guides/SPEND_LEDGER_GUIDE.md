# Spend Ledger Guide

![Spend ledger demo](../../assets/demo/spend-ledger.gif)

Durable SQLite spend tracking with `GET /v1/spend` aggregation. Closes the gap
versus LiteLLM virtual-key `/spend` APIs for GPT-6 Astra / Claude Opus 5.5 /
Gemini 4 Argon / Kimi K3.1 Pro fleets.

## Why

In-memory budget counters reset on process restart. Portfolio demos and
multi-tenant gateways need a durable spend log that survives new process
instances pointing at the same database path.

## How it works

1. On each successful routed completion, `NexusRouter` appends a row to
   `SpendLedger` (SQLite).
2. Rows capture `request_id`, tenant (`user_id`), provider, model, cost,
   token counts, and `api_key_id` (the key fingerprint, or the virtual key id
   in `virtual_keys` mode; never the secret). Ledgers created before the
   column existed gain it in place, with `NULL` in older rows.
3. `summary(tenant=..., provider=..., model=..., since=..., until=...,
   api_key_id=...)` aggregates with optional half-open time window
   (`recorded_at >= since` and `recorded_at < until`); `key_spend(api_key_id,
   since=...)` returns one key's spend and request count, which virtual-key
   budgets are checked against (see `VIRTUAL_KEYS_GUIDE.md`).
4. `GET /v1/spend` returns totals plus breakdowns by tenant / provider / model.
   In `virtual_keys` mode a virtual key sees only its own requests.
5. `POST /v1/spend` accepts manual records for demos or offline ingestion
   (master key only in `virtual_keys` mode).
6. Pair with `TenantSpendQuotaEnforcer` for fail-closed UTC calendar-month
   USD caps (see `TENANT_SPEND_QUOTA_GUIDE.md`).

## Configuration

```bash
NEXUS_SPEND_LEDGER_PATH=migrations/spend-ledger.sqlite3
```

## Example

```bash
curl -s http://localhost:8000/v1/spend | jq
curl -s -X POST http://localhost:8000/v1/spend \
  -H 'content-type: application/json' \
  -d '{
    "request_id": "demo-1",
    "tenant": "acme",
    "provider": "openai",
    "model": "gpt-5.5",
    "cost_usd": 0.12
  }'
```

## Gap vs LiteLLM / Portkey

| Capability | LiteLLM virtual keys | Nexus spend ledger |
| --- | --- | --- |
| Durable spend log | Yes | Yes (SQLite) |
| `/spend` summary API | Yes | `GET /v1/spend` |
| Record on success | Yes | Wired in router `_respond` |
| Manual record API | Yes | `POST /v1/spend` |
